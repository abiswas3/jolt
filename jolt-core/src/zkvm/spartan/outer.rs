use allocative::Allocative;
use ark_std::Zero;
use rayon::prelude::*;
use std::sync::Arc;
use std::usize;
use tracer::instruction::Cycle;

use crate::field::{FMAdd, JoltField, MontgomeryReduce};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::poly::multiquadratic_poly::MultiquadraticPolynomial;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly_generalised::{GruenSplitEqPolynomialGeneral, SumCheckMode};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::streaming_schedule::StreamingSchedule;
use crate::subprotocols::sumcheck_prover::{
    SumcheckInstanceProver, UniSkipFirstRoundInstanceProver,
};
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::subprotocols::univariate_skip::{build_uniskip_first_round_poly, UniSkipState};
use crate::transcripts::Transcript;
use crate::utils::accumulation::Acc8S;
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::{
    constraints::{
        OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DEGREE,
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
    },
    evaluation::R1CSEval,
    inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS},
};
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltSharedPreprocessing;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;

#[cfg(test)]
use crate::zkvm::r1cs::constraints::{R1CS_CONSTRAINTS_FIRST_GROUP, R1CS_CONSTRAINTS_SECOND_GROUP};
#[cfg(test)]
use crate::zkvm::r1cs::inputs::JoltR1CSInputs;

/// Degree bound of the sumcheck round polynomials for [`OuterRemainingSumcheckVerifier`].
const OUTER_REMAINING_DEGREE_BOUND: usize = 3;
//const WINDOW_WIDTH: usize = 3;
const INFINITY: usize = 2; // 2 represents ∞ in base-3

// Spartan Outer sumcheck
// (with univariate-skip first round on Z, and no Cz term given all eq conditional constraints)
//
// We define a univariate in Z first-round polynomial
//   s1(Y) := L(τ_high, Y) · Σ_{x_out ∈ {0,1}^{m_out}} Σ_{x_in ∈ {0,1}^{m_in}}
//              E_out(r_out, x_out) · E_in(r_in, x_in) ·
//              [ Az(x_out, x_in, Y) · Bz(x_out, x_in, Y) ],
// where L(τ_high, Y) is the Lagrange basis polynomial over the univariate-skip
// base domain evaluated at τ_high, and Az(·,·,Y), Bz(·,·,Y) are the
// per-row univariate polynomials in Y induced by the R1CS row (split into two
// internal groups in code, but algebraically composing to Az·Bz at Y).
// The prover sends s1(Y) via univariate-skip by evaluating t1(Y) := Σ Σ E_out·E_in·(Az·Bz)
// on an extended grid Y ∈ {−D..D} outside the base window, interpolating t1,
// multiplying by L(τ_high, Y) to obtain s1, and the verifier samples r0.
//
// Subsequent outer rounds bind the cycle variables r_tail = (r1, r2, …) using
// a streaming first cycle-bit round followed by linear-time rounds:
//   • Streaming round (after r0): compute
//       t(0)  = Σ_{x_out} E_out · Σ_{x_in} E_in · (Az(0)·Bz(0))
//       t(∞)  = Σ_{x_out} E_out · Σ_{x_in} E_in · ((Az(1)−Az(0))·(Bz(1)−Bz(0)))
//     send a cubic built from these endpoints, and bind cached coefficients by r1.
//   • Remaining rounds: reuse bound coefficients to compute the same endpoints
//     in linear time for each subsequent bit and bind by r_i.
//
// Final check (verifier): with r = [r0 || r_tail] and outer binding order from
// the top, evaluate Eq_τ(τ, r) and verify
//   Eq_τ(τ, r) · (Az(r) · Bz(r)).

/// Uni-skip instance for Spartan outer sumcheck, computing the first-round polynomial only.
#[derive(Allocative)]
pub struct OuterUniSkipInstanceProver<F: JoltField> {
    tau: Vec<F::Challenge>,
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; OUTER_UNIVARIATE_SKIP_DEGREE],
}

impl<F: JoltField> OuterUniSkipInstanceProver<F> {
    #[tracing::instrument(skip_all, name = "OuterUniSkipInstanceProver::gen")]
    pub fn gen<PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, PCS>,
        tau: &[F::Challenge],
    ) -> Self {
        let (preprocessing, _, trace, _program_io, _final_mem) = state_manager.get_prover_data();

        let tau_low = &tau[0..tau.len() - 1];

        let extended =
            Self::compute_univariate_skip_extended_evals(&preprocessing.shared, trace, tau_low);

        let instance = Self {
            tau: tau.to_vec(),
            extended_evals: extended,
        };

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("OuterUniSkipInstance", &instance);
        instance
    }

    /// Compute the extended evaluations of the univariate skip polynomial, i.e.
    ///
    /// t_1(y) = \sum_{x_out} eq(tau_out, x_out) * \sum_{x_in} eq(tau_in, x_in) * Az(x_out, x_in, y) * Bz(x_out, x_in, y)
    ///
    /// for all y in the extended domain {−D..D} outside the base window
    /// (inside the base window, we have t_1(y) = 0)
    ///
    /// Note that the last of the x_in variables corresponds to the group index of the constraints
    /// (since we split the constraints in half, and y ranges over the number of constraints in each group)
    ///
    /// So we actually need to be careful and compute
    ///
    /// \sum_{x_in'} eq(tau_in, (x_in', 0)) * Az(x_out, x_in', 0, y) * Bz(x_out, x_in', 0, y)
    ///     + eq(tau_in, (x_in', 1)) * Az(x_out, x_in', 1, y) * Bz(x_out, x_in', 1, y)
    fn compute_univariate_skip_extended_evals(
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        tau_low: &[F::Challenge],
    ) -> [F; OUTER_UNIVARIATE_SKIP_DEGREE] {
        let m = tau_low.len() / 2;
        let (tau_out, tau_in) = tau_low.split_at(m);
        // Compute the split eq polynomial, one scaled by R^2 in order to balance against
        // Montgomery (not Barrett) reduction later on in 8-limb signed accumulation
        // of e_in * (az * bz)
        let (E_out, E_in) = rayon::join(
            || EqPolynomial::evals_with_scaling(tau_out, Some(F::MONTGOMERY_R_SQUARE)),
            || EqPolynomial::evals(tau_in),
        );

        let num_x_out_vals = E_out.len();
        let num_x_in_vals = E_in.len();
        assert!(
            num_x_in_vals >= 2,
            "univariate skip expects at least 2 x_in values (last bit is group index)"
        );
        // The last x_in bit is the group selector: even indices -> group 0, odd -> group 1
        let num_x_in_half = num_x_in_vals >> 1;

        let num_parallel_chunks = core::cmp::min(
            num_x_out_vals,
            rayon::current_num_threads().next_power_of_two() * 8,
        );
        let x_out_chunk_size = if num_x_out_vals > 0 {
            core::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0
        };
        let iter_num_x_in_vars = num_x_in_vals.log_2();
        let iter_num_x_in_prime_vars = iter_num_x_in_vars - 1; // ignore last bit (group index)

        //TODO: (ari) there could be cacche optimisations here.
        (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let mut acc_unreduced: [F::Unreduced<9>; OUTER_UNIVARIATE_SKIP_DEGREE] =
                    [F::Unreduced::<9>::zero(); OUTER_UNIVARIATE_SKIP_DEGREE];

                for x_out_val in x_out_start..x_out_end {
                    let mut inner_acc: [Acc8S<F>; OUTER_UNIVARIATE_SKIP_DEGREE] =
                        [Acc8S::<F>::zero(); OUTER_UNIVARIATE_SKIP_DEGREE];
                    for x_in_prime in 0..num_x_in_half {
                        // Materialize row once for both groups (ignores last bit)
                        let base_step_idx = (x_out_val << iter_num_x_in_prime_vars) | x_in_prime;
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, base_step_idx);

                        // Group 0 (even index)
                        let x_in_even = x_in_prime << 1;
                        let e_in_even = E_in[x_in_even];

                        let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
                        for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                            let prod_s192 = eval.extended_azbz_product_first_group(j);
                            inner_acc[j].fmadd(&e_in_even, &prod_s192);
                        }

                        // Group 1 (odd index) using same row inputs
                        let x_in_odd = x_in_even + 1;
                        let e_in_odd = E_in[x_in_odd];

                        for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                            let prod_s192 = eval.extended_azbz_product_second_group(j);
                            inner_acc[j].fmadd(&e_in_odd, &prod_s192);
                        }
                    }
                    let e_out = E_out[x_out_val];
                    for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                        let reduced = inner_acc[j].montgomery_reduce();
                        acc_unreduced[j] += e_out.mul_unreduced::<9>(reduced);
                    }
                }
                acc_unreduced
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); OUTER_UNIVARIATE_SKIP_DEGREE],
                |mut a, b| {
                    for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                        a[j] += b[j];
                    }
                    a
                },
            )
            .map(F::from_montgomery_reduce::<9>)
    }
}

impl<F: JoltField, T: Transcript> UniSkipFirstRoundInstanceProver<F, T>
    for OuterUniSkipInstanceProver<F>
{
    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "OuterUniSkipInstanceProver::compute_poly")]
    fn compute_poly(&mut self) -> UniPoly<F> {
        // Load extended univariate-skip evaluations from prover state
        let extended_evals = &self.extended_evals;

        let tau_high = self.tau[self.tau.len() - 1];

        // Compute the univariate-skip first round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        build_uniskip_first_round_poly::<
            F,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            OUTER_UNIVARIATE_SKIP_DEGREE,
            OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        >(None, extended_evals, tau_high)
    }
}

/// SumcheckInstance for Spartan outer rounds after the univariate-skip first round.
/// Round 0 in this instance corresponds to the "streaming" round; subsequent rounds
/// use the remaining linear-time algorithm over cycle variables.
#[derive(Allocative)]
pub struct OuterRemainingSumcheckProver<F: JoltField, S: StreamingSchedule> {
    #[allocative(skip)]
    preprocess: Arc<JoltSharedPreprocessing>,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    //split_eq_poly: GruenSplitEqPolynomial<F>,
    split_eq_poly_gen: GruenSplitEqPolynomialGeneral<F>,
    az: Option<DensePolynomial<F>>,
    bz: Option<DensePolynomial<F>>,
    t_prime_poly: Option<MultiquadraticPolynomial<F>>, // multiquadratic polynomial used to answer queries in a streaming window
    r_grid: Option<Vec<F>>, // hadamard product of (1 - r_j, r_j) for bound variables so far to help with streaming
    /// The first round evals (t0, t_inf) computed from a streaming pass over the trace
    #[allocative(skip)]
    params: OuterRemainingSumcheckParams<F>,
    lagrange_evals_r0: [F; 10],
    schedule: S,
}

impl<F: JoltField, S: StreamingSchedule> OuterRemainingSumcheckProver<F, S> {
    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::gen")]
    pub fn gen<PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, PCS>,
        num_cycles_bits: usize,
        uni: &UniSkipState<F>,
        schedule: S,
    ) -> Self {
        let (preprocessing, _, trace, _program_io, _final_mem) = state_manager.get_prover_data();

        let outer_params = OuterRemainingSumcheckParams::new(num_cycles_bits, uni);

        // L_{-5}(r0) ,..., L_{4}(r0)
        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&uni.r0);

        let tau_high = uni.tau[uni.tau.len() - 1]; // r0 variable
        let tau_low = &uni.tau[..uni.tau.len() - 1]; // includes group index

        // compute eq(tau_hi, r0) where r0 is the challenge from first round
        let lagrange_tau_r0: F = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&uni.r0, &tau_high);

        // Note: tau_lo contains both time vars and group index
        // the scaling factor simply multiplyes everyhing
        // with \eq(tau_hi, r_0)
        // internally this stores a Vec<F> of size 2^{\log T + 1}
        //let split_eq_poly: GruenSplitEqPolynomial<F> =
        //    GruenSplitEqPolynomial::<F>::new_with_scaling(
        //        tau_low,
        //        BindingOrder::LowToHigh,
        //        Some(lagrange_tau_r0),
        //    );

        let sumcheck_mode = if schedule.is_streaming(0) {
            SumCheckMode::STREAMING
        } else {
            SumCheckMode::LINEAR
        };
        let split_eq_poly_gen: GruenSplitEqPolynomialGeneral<F> =
            GruenSplitEqPolynomialGeneral::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
                schedule.num_unbound_vars(0),
                sumcheck_mode, // THIS should be based on window schedule
            );

        //let (t0, t_inf, az_bound, bz_bound) = Self::compute_first_quadratic_evals_and_bound_polys(
        //    &preprocessing.shared,
        //    trace,
        //    &lagrange_evals_r,
        //    &split_eq_poly,
        //);

        Self {
            //split_eq_poly,
            split_eq_poly_gen,
            preprocess: Arc::new(preprocessing.shared.clone()),
            trace: Arc::new(trace.to_vec()),
            az: None,
            bz: None,
            t_prime_poly: None,
            r_grid: Some(vec![F::one()]),
            params: outer_params,
            lagrange_evals_r0: lagrange_evals_r,
            schedule,
        }
    }

    /// Compute the window lenght give current round_idx
    pub fn get_window_with(&self, round_idx: usize) -> usize {
        round_idx
    }

    // only valid for the initial window
    // given a degree 1 polynomial in dim variables
    // as evaluations over {0,1}^dim
    // return evaluations over {0,1,inf}^dim
    pub fn extrapolate_multivariate_1_to_2(
        input: &[F],      // initial buffer (size 2^dim)
        output: &mut [F], // final buffer (size 3^dim)
        tmp: &mut [F],    // scratch buffer, also (size 3^dim)
        dim: usize,
    ) {
        // Fill output by expanding one dimension at a time.
        // We treat slices of increasing "arity"

        // Copy the initial evaluations into the start of either
        // tmp or output, depending on parity of dim.
        // We'll alternate between tmp and output as we expand dimensions.
        let (mut cur, mut next) = if dim % 2 == 1 {
            tmp[..input.len()].copy_from_slice(input);
            (tmp, output)
        } else {
            output[..input.len()].copy_from_slice(input);
            (output, tmp)
        };

        let mut in_stride = 1usize;
        let in_size = 1 << dim;
        let mut out_stride = 1usize;
        let out_size = 3usize.pow(dim as u32);
        let mut blocks = 1 << (dim - 1);

        // sanity checks
        assert_eq!(cur.len(), out_size);
        assert_eq!(next.len(), out_size);
        assert_eq!(input.len(), in_size);

        // start from the smallest subcubes and expand dimension by dimension
        for _ in 0..dim {
            for b in 0..blocks {
                let in_off = b * 2 * in_stride;
                let out_off = b * 3 * out_stride;

                for j in 0..in_stride {
                    // 1d extrapolate
                    let f0 = cur[in_off + j];
                    let f1 = cur[in_off + in_stride + j];
                    next[out_off + j] = f0;
                    next[out_off + out_stride + j] = f1;
                    next[out_off + 2 * out_stride + j] = f1 - f0;
                }
            }
            // swap buffers
            std::mem::swap(&mut cur, &mut next);
            in_stride *= 3;
            out_stride *= 3;
            blocks /= 2;
        }
    }

    // gets the evaluations of az(x, {0,1}^log(jlen), r)
    // where x is determined by the bit decomposition of offset
    // and r is log(klen) variables
    // this is used both in window computation (jlen is window size)
    // and in converting to linear time (offset is 0, log(jlen) is the number of unbound variables)
    fn build_grids(
        &self,
        grid_az: &mut Vec<F>,
        grid_bz: &mut Vec<F>,
        jlen: usize,
        klen: usize,
        offset: usize,
    ) {
        let preprocess = &self.preprocess;
        let trace = &self.trace;
        let lagrange_evals_r = &self.lagrange_evals_r0;
        let r_grid = self.r_grid.as_ref().unwrap();
        for j in 0..jlen {
            for k in 0..klen {
                let full_idx = offset + j * klen + k;
                let current_step_idx = full_idx >> 1;
                let selector = (full_idx & 1) == 1;
                let (az, bz) = self.get_az_bz_at_curr_timestep(
                    current_step_idx,
                    selector,
                    preprocess,
                    trace,
                    lagrange_evals_r,
                );
                if klen > 1 {
                    grid_az[j] += az * r_grid[k];
                    grid_bz[j] += bz * r_grid[k];
                } else {
                    grid_az[j] = az;
                    grid_bz[j] = bz;
                }
            }
        }
    }

    fn update_r_grid(&mut self, r_j: F::Challenge) {
        // Another function that builds self.r_grid()
        // EXAMPLE:
        // Initially len = 1 and r_grid = [1]
        // then receive r_1
        // next = [(1-r1), r1]
        // then receive r_2
        // next is of size 4
        // next = [(1-r1)(1-r2), r1 (1-r2), (1-r1)r2, r1r2]
        let r_grid = self.r_grid.as_mut().unwrap();
        let len = r_grid.len();
        let mut next = Vec::with_capacity(2 * len);
        for (_i, v) in r_grid.iter().enumerate() {
            next.push(*v * (F::one() - r_j));
        }
        for (_i, v) in r_grid.iter().enumerate() {
            next.push(*v * r_j);
        }
        *r_grid = next;
    }

    // returns the grid of evaluations on {0,1,inf}^window_size
    // touches each cycle of the trace exactly once and in order!
    fn get_grid_gen(&mut self, window_size: usize) {
        let split_eq_poly = &self.split_eq_poly_gen;
        // helper constants
        let three_pow_dim = 3_usize.pow(window_size as u32);
        let jlen = 1 << window_size;
        let klen = 1 << (split_eq_poly.num_challenges());
        // intermediate buffers
        let mut buff_a = vec![F::zero(); three_pow_dim];
        let mut buff_b = vec![F::zero(); three_pow_dim];
        let mut tmp = vec![F::zero(); three_pow_dim];
        // output
        let mut res = vec![F::zero(); three_pow_dim];
        // main logic
        for (out_idx, out_val) in split_eq_poly.E_out_current().iter().enumerate() {
            for (in_idx, in_val) in split_eq_poly.E_in_current().iter().enumerate() {
                let i = out_idx * split_eq_poly.E_in_current_len() + in_idx;
                let mut grid_a = vec![F::zero(); jlen];
                let mut grid_b = vec![F::zero(); jlen];
                self.build_grids(&mut grid_a, &mut grid_b, jlen, klen, i * jlen * klen);
                // extrapolate grid_a and grid_b from {0,1}^window_size to {0,1,inf}^window_size
                Self::extrapolate_multivariate_1_to_2(&grid_a, &mut buff_a, &mut tmp, window_size);
                Self::extrapolate_multivariate_1_to_2(&grid_b, &mut buff_b, &mut tmp, window_size);
                let aux = out_val.clone() * in_val.clone();
                for idx in 0..three_pow_dim {
                    res[idx] += buff_a[idx] * buff_b[idx] * aux;
                }
            }
        }
        self.t_prime_poly = Some(MultiquadraticPolynomial::new(window_size, res));
    }

    // single pass over the trace to compute az and bz for linear time prover
    fn stream_to_linear_time(&mut self) {
        let split_eq_poly = &self.split_eq_poly_gen;
        // helper constants
        let jlen = 1 << (split_eq_poly.get_num_vars() - split_eq_poly.num_challenges());
        let klen = 1 << split_eq_poly.num_challenges();
        let mut ret_az = vec![F::zero(); jlen];
        let mut ret_bz = vec![F::zero(); jlen];
        self.build_grids(&mut ret_az, &mut ret_bz, jlen, klen, 0);
        self.az = Some(DensePolynomial::new(ret_az));
        self.bz = Some(DensePolynomial::new(ret_bz));
    }

    fn get_az_bz_at_curr_timestep(
        &self,
        current_step_idx: usize,
        selector: bool,
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        lagrange_evals_r: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
    ) -> (F, F) {
        let row_inputs = R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);
        let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
        // This binds r0
        if !selector {
            let az0 = eval.az_at_r_first_group(lagrange_evals_r);
            let bz0 = eval.bz_at_r_first_group(lagrange_evals_r);
            (az0, bz0)
        } else {
            let az1 = eval.az_at_r_second_group(lagrange_evals_r);
            let bz1 = eval.bz_at_r_second_group(lagrange_evals_r);
            (az1, bz1)
        }
    }

    /// Compute the polynomial for each of the remaining rounds, using the
    /// linear-time algorithm with split-eq optimizations.
    ///
    /// At this point, we have computed the `bound_coeffs` for the current round.
    /// We need to compute:
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// (az_bound[x_out, x_in, 0] * bz_bound[x_out, x_in, 0] - cz_bound[x_out, x_in, 0])`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// az_bound[x_out, x_in, ∞] * bz_bound[x_out, x_in, ∞]`
    ///
    /// (ordering of indices is MSB to LSB, so x_out is the MSB and x_in is the LSB)
    #[inline]
    fn remaining_quadratic_evals(&self) -> (F, F) {
        let eq_poly = &self.split_eq_poly_gen;

        let n = self.az.as_ref().expect("az should be initialized").len();
        let az = self.az.as_ref().expect("az should be initialized");
        let bz = self.bz.as_ref().expect("bz should be initialized");

        debug_assert_eq!(n, bz.len());
        if eq_poly.E_in_current_len() == 1 {
            // groups are pairs (0,1)
            let groups = n / 2;
            let (t0_unr, tinf_unr) = (0..groups)
                .into_par_iter()
                .map(|g| {
                    let az0 = az[2 * g];
                    let az1 = az[2 * g + 1];
                    let bz0 = bz[2 * g];
                    let bz1 = bz[2 * g + 1];
                    let eq = eq_poly.E_out_current()[g];
                    let p0 = az0 * bz0;
                    let slope = (az1 - az0) * (bz1 - bz0);
                    let t0_unr = eq.mul_unreduced::<9>(p0);
                    let tinf_unr = eq.mul_unreduced::<9>(slope);
                    (t0_unr, tinf_unr)
                })
                .reduce(
                    || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                    |a, b| (a.0 + b.0, a.1 + b.1),
                );
            (
                F::from_montgomery_reduce::<9>(t0_unr),
                F::from_montgomery_reduce::<9>(tinf_unr),
            )
        } else {
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
            let x1_len = eq_poly.E_in_current_len();
            let x2_len = eq_poly.E_out_current_len();
            let (sum0_unr, suminf_unr) = (0..x2_len)
                .into_par_iter()
                .map(|x2| {
                    let mut inner0_unr = F::Unreduced::<9>::zero();
                    let mut inner_inf_unr = F::Unreduced::<9>::zero();
                    for x1 in 0..x1_len {
                        let g = (x2 << num_x1_bits) | x1;
                        let az0 = az[2 * g];
                        let az1 = az[2 * g + 1];
                        let bz0 = bz[2 * g];
                        let bz1 = bz[2 * g + 1];
                        let e_in = eq_poly.E_in_current()[x1];
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);
                        inner0_unr += e_in.mul_unreduced::<9>(p0);
                        inner_inf_unr += e_in.mul_unreduced::<9>(slope);
                    }
                    let e_out = eq_poly.E_out_current()[x2];
                    let inner0_red = F::from_montgomery_reduce::<9>(inner0_unr);
                    let inner_inf_red = F::from_montgomery_reduce::<9>(inner_inf_unr);
                    let t0_unr = e_out.mul_unreduced::<9>(inner0_red);
                    let tinf_unr = e_out.mul_unreduced::<9>(inner_inf_red);
                    (t0_unr, tinf_unr)
                })
                .reduce(
                    || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                    |a, b| (a.0 + b.0, a.1 + b.1),
                );
            (
                F::from_montgomery_reduce::<9>(sum0_unr),
                F::from_montgomery_reduce::<9>(suminf_unr),
            )
        }
    }
    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let az = self.az.as_ref().expect("az should be initialized");
        let bz = self.bz.as_ref().expect("bz should be initialized");

        let az0 = if !az.is_empty() { az[0] } else { F::zero() };
        let bz0 = if !bz.is_empty() { bz[0] } else { F::zero() };
        [az0, bz0]
    }
}

impl<F: JoltField, T: Transcript, S: StreamingSchedule> SumcheckInstanceProver<F, T>
    for OuterRemainingSumcheckProver<F, S>
{
    fn degree(&self) -> usize {
        OUTER_REMAINING_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingSumcheckProver::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let (t0, t_inf) = if self.schedule.is_streaming(round) {
            let num_unbound_vars = self.schedule.num_unbound_vars(round);

            if self.schedule.is_window_start(round) {
                // NOTE: Important that this get updated first
                // As this re-computes E_out and E_in
                self.split_eq_poly_gen
                    .recompute_eq_polys_for_streaming(num_unbound_vars);
                self.get_grid_gen(num_unbound_vars);
            }
            // Use the multiquadratic polynomial to compute the message
            let t_prime_poly = self
                .t_prime_poly
                .as_ref()
                .expect("t_prime_poly should be initialized");
            let E_active = self.split_eq_poly_gen.E_active_current();
            let t_prime_0 = t_prime_poly.project_to_first_variable(E_active, 0);
            let t_prime_inf = t_prime_poly.project_to_first_variable(E_active, INFINITY);

            //if round == 0 {
            //    let (t0_expected, t_inf_expected) = self.first_round_evals;
            //    assert_eq!(t0_expected, t_prime_0);
            //    assert_eq!(t_inf_expected, t_prime_inf);
            //} else {
            //    let (t0_expected, t_inf_expected) = self.remaining_quadratic_evals();
            //    assert_eq!(t0_expected, t_prime_0, "t0 mismatch at round {}", round);
            //    assert_eq!(
            //        t_inf_expected, t_prime_inf,
            //        "t_inf mismatch at round {}",
            //        round
            //    );
            //}

            (t_prime_0, t_prime_inf)
        } else {
            // LINEAR PHASE
            if self.schedule.is_first_linear(round) {
                self.split_eq_poly_gen.recompute_eq_polys_for_linear();
                // This is just a placeholder for now
                self.stream_to_linear_time();
                //let (_t0, _t_inf) = self.compute_az_bz_for_linear_sumcheck();
            }
            // For now, just use quadratic evals
            let (t0, t_inf) = self.remaining_quadratic_evals();
            (t0, t_inf)
        };
        //let evals = self
        //    .split_eq_poly
        //    .gruen_evals_deg_3(t0, t_inf, previous_claim);
        //
        let evals = self
            .split_eq_poly_gen
            .gruen_evals_deg_3(t0, t_inf, previous_claim);
        //assert_eq!(evals_grid[0], evals[0], "Gruen should also work");
        //self.stream_to_linear_time();
        vec![evals[0], evals[1], evals[2]]
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        // This should only happen in the Linear schedule from now on
        //self.split_eq_poly.bind(r_j);

        // NEW API
        if self.schedule.is_streaming(round) {
            let t_prime_poly = self
                .t_prime_poly
                .as_mut()
                .expect("t_prime_poly should be initialized");
            t_prime_poly.bind(r_j, BindingOrder::LowToHigh);
            self.update_r_grid(r_j);

            self.split_eq_poly_gen.bind(r_j, SumCheckMode::STREAMING);
        } else {
            self.split_eq_poly_gen.bind(r_j, SumCheckMode::LINEAR);
            rayon::join(
                || {
                    self.az
                        .as_mut()
                        .expect("az should be initialised")
                        .bind_parallel(r_j, BindingOrder::LowToHigh)
                },
                || {
                    self.bz
                        .as_mut()
                        .expect("bz should be initialised")
                        .bind_parallel(r_j, BindingOrder::LowToHigh)
                },
            );
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.get_opening_point(sumcheck_challenges);

        // Append Az, Bz claims and corresponding opening point
        let claims = self.final_sumcheck_evals();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[0],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[1],
        );

        // Handle witness openings at r_cycle (use consistent split length)
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.params.num_cycles_bits);

        // Compute claimed witness evals and append virtual openings for all R1CS inputs
        let claimed_witness_evals =
            R1CSEval::compute_claimed_inputs(&self.preprocess, &self.trace, r_cycle);

        #[cfg(test)]
        {
            // Recompute Az,Bz at the final opening point USING ONLY the claimed witness MLEs z(r_cycle),
            // then compare to the prover's final Az,Bz claims. This validates the consistency wiring
            // between the outer sumcheck and the witness openings.

            // Prover's final Az,Bz claims (after all bindings)
            let claims = self.final_sumcheck_evals();

            // Extract streaming-round challenge r_stream from the opening point tail (after r_cycle)
            let (_, rx_tail) = opening_point.r.split_at(self.params.num_cycles_bits);
            let r_stream = rx_tail[0];

            // Build z(r_cycle) vector extended with a trailing 1 for the constant column
            let const_col = JoltR1CSInputs::num_inputs();
            let mut z_cycle_ext = claimed_witness_evals.to_vec();
            z_cycle_ext.push(F::one());

            // Lagrange weights over the univariate-skip base domain at r0
            let w = LagrangePolynomial::<F>::evals::<F::Challenge, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(
                &self.params.r0_uniskip,
            );

            // Group 0 fused Az,Bz via dot product of LC with z(r_cycle)
            let mut az_g0 = F::zero();
            let mut bz_g0 = F::zero();
            for i in 0..R1CS_CONSTRAINTS_FIRST_GROUP.len() {
                let lc_a = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.a;
                let lc_b = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.b;
                az_g0 += w[i] * lc_a.dot_eq_ry::<F>(&z_cycle_ext, const_col);
                bz_g0 += w[i] * lc_b.dot_eq_ry::<F>(&z_cycle_ext, const_col);
            }

            // Group 1 fused Az,Bz (use same Lagrange weights order as construction)
            let mut az_g1 = F::zero();
            let mut bz_g1 = F::zero();
            let g2_len = core::cmp::min(
                R1CS_CONSTRAINTS_SECOND_GROUP.len(),
                OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            );
            for i in 0..g2_len {
                let lc_a = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.a;
                let lc_b = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.b;
                az_g1 += w[i] * lc_a.dot_eq_ry::<F>(&z_cycle_ext, const_col);
                bz_g1 += w[i] * lc_b.dot_eq_ry::<F>(&z_cycle_ext, const_col);
            }

            // Bind by r_stream to match the outer streaming combination used for final Az,Bz
            let az_final = az_g0 + r_stream * (az_g1 - az_g0);
            let bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0);

            assert_eq!(
                az_final, claims[0],
                "Az final eval mismatch vs claims from evaluating R1CS inputs at r_cycle: recomputed={} claimed={}",
                az_final, claims[0]
            );
            assert_eq!(
                bz_final, claims[1],
                "Bz final eval mismatch vs claims from evaluating R1CS inputs at r_cycle: recomputed={} claimed={}",
                bz_final, claims[1]
            );
        }

        for (i, input) in ALL_R1CS_INPUTS.iter().enumerate() {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
                claimed_witness_evals[i],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct OuterRemainingSumcheckVerifier<F: JoltField> {
    params: OuterRemainingSumcheckParams<F>,
}

impl<F: JoltField> OuterRemainingSumcheckVerifier<F> {
    pub fn new(num_cycles_bits: usize, uni: &UniSkipState<F>) -> Self {
        let params = OuterRemainingSumcheckParams::new(num_cycles_bits, uni);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for OuterRemainingSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        OUTER_REMAINING_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (_, claim_Az) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);

        let tau = &self.params.tau;
        let tau_high = &tau[tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.params.r0_uniskip);
        let tau_low = &tau[..tau.len() - 1];
        let r_tail_reversed: Vec<F::Challenge> =
            sumcheck_challenges.iter().rev().copied().collect();
        let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);
        tau_high_bound_r0 * tau_bound_r_tail_reversed * claim_Az * claim_Bz
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.get_opening_point(sumcheck_challenges);

        // Populate Az, Bz openings at the full outer opening point
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );

        // Append witness openings at r_cycle (no claims at verifier) for all R1CS inputs
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.params.num_cycles_bits);
        ALL_R1CS_INPUTS.iter().for_each(|input| {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
            );
        });
    }
}

struct OuterRemainingSumcheckParams<F: JoltField> {
    /// Number of cycle bits for splitting opening points (consistent across prover/verifier)
    /// Total number of rounds is `1 + num_cycles_bits`
    num_cycles_bits: usize,
    /// The tau vector (length `2 + num_cycles_bits`, sampled at the beginning for Lagrange + eq poly)
    tau: Vec<F::Challenge>,
    /// The univariate-skip first round challenge
    r0_uniskip: F::Challenge,
    /// Claim after the univariate-skip first round, updated every round
    input_claim: F,
}

impl<F: JoltField> OuterRemainingSumcheckParams<F> {
    fn new(num_cycles_bits: usize, uni: &UniSkipState<F>) -> Self {
        Self {
            num_cycles_bits,
            tau: uni.tau.clone(),
            r0_uniskip: uni.r0,
            input_claim: uni.claim_after_first,
        }
    }
    fn num_rounds(&self) -> usize {
        1 + self.num_cycles_bits
    }

    fn get_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_tail = sumcheck_challenges;
        let r_full = [&[self.r0_uniskip], r_tail].concat();
        OpeningPoint::<LITTLE_ENDIAN, F>::new(r_full).match_endianness()
    }
}
