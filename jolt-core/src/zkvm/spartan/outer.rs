use allocative::Allocative;
use ark_std::Zero;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

use crate::field::{FMAdd, JoltField, MontgomeryReduce};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
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
use crate::utils::thread::unsafe_allocate_zero_vec;
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
const WINDOW_WIDTH: usize = 2;

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
pub struct OuterRemainingSumcheckProver<F: JoltField> {
    #[allocative(skip)]
    preprocess: Arc<JoltSharedPreprocessing>,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    split_eq_poly: GruenSplitEqPolynomial<F>,
    // We still have these but their sizes will change
    az: Option<DensePolynomial<F>>,
    bz: Option<DensePolynomial<F>>,
    s_reduced: Option<Vec<F>>,
    /// The first round evals (t0, t_inf) computed from a streaming pass over the trace
    first_round_evals: (F, F),
    #[allocative(skip)]
    params: OuterRemainingSumcheckParams<F>,
}

impl<F: JoltField> OuterRemainingSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::gen")]
    pub fn gen<PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, PCS>,
        num_cycles_bits: usize,
        uni: &UniSkipState<F>,
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
        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
            );

        let (t0, t_inf, az_bound, bz_bound, s_reduced) =
            Self::compute_first_quadratic_evals_and_bound_polys_new(
                &preprocessing.shared,
                trace,
                &lagrange_evals_r,
                &split_eq_poly,
            );

        Self {
            split_eq_poly,
            preprocess: Arc::new(preprocessing.shared.clone()),
            trace: Arc::new(trace.to_vec()),
            az: az_bound,
            bz: bz_bound,
            s_reduced,
            first_round_evals: (t0, t_inf),
            params: outer_params,
        }
    }

    /// Compute the window lenght give current round_idx
    pub fn get_window_with(&self, round_idx: usize) -> usize {
        round_idx
    }

    /// I need to recompute the trace -- This looks very similar to what I am currently doing.
    pub fn re_compute_multivariate_polynomial(&self) -> (F, F) {
        (F::zero(), F::zero())
    }

    #[inline]
    fn _compute_first_quadratic_evals_and_bound_polys(
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        lagrange_evals_r: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        split_eq_poly: &GruenSplitEqPolynomial<F>,
    ) -> (F, F, DensePolynomial<F>, DensePolynomial<F>) {
        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();
        let groups_exact = num_x_out_vals
            .checked_mul(num_x_in_vals)
            .expect("overflow computing groups_exact");

        // Preallocate interleaved buffers once ([lo, hi] per entry)
        let mut az_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);
        let mut bz_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);

        // Parallel over x_out groups using exact-sized mutable chunks, with per-worker fold
        // NOTE: The parallelism is only over the outer evals, and this prevents any false sharing.
        // Each unit of work is simpy (2\sqrt{T}, 2\sqrt{T}).
        let (t0_acc_unr, t_inf_acc_unr) = az_bound
            .par_chunks_exact_mut(2 * num_x_in_vals)
            .zip(bz_bound.par_chunks_exact_mut(2 * num_x_in_vals))
            .enumerate()
            .fold(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |(mut acc0, mut acci), (x_out_val, (az_chunk, bz_chunk))| {
                    let mut inner_sum0 = F::Unreduced::<9>::zero();
                    let mut inner_sum_inf = F::Unreduced::<9>::zero();
                    for x_in_val in 0..num_x_in_vals {
                        // current_step_idx = (x_out_val || x_in_val)
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);
                        let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
                        // This binds r0
                        let az0 = eval.az_at_r_first_group(lagrange_evals_r);
                        let bz0 = eval.bz_at_r_first_group(lagrange_evals_r);
                        let az1 = eval.az_at_r_second_group(lagrange_evals_r);
                        let bz1 = eval.bz_at_r_second_group(lagrange_evals_r);
                        let p0 = az0 * bz0;
                        // Let A(x) and B(x) be the univariate polynomials the prover
                        // needs to send in round 1 where x is group index variable.
                        // A(x) = qx + p
                        // B(x) = tx + s
                        // A(x)B(x) = tq x^2 + (tp + sq)x + ps
                        // The leading coeff is
                        // A(1) - A(0) * B(1) - B(0) = (p + q - p) * (t + s - s) = tq
                        let slope = (az1 - az0) * (bz1 - bz0);
                        let e_in = split_eq_poly.E_in_current()[x_in_val];
                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sum_inf += e_in.mul_unreduced::<9>(slope);
                        let off = 2 * x_in_val;
                        az_chunk[off] = az0;
                        az_chunk[off + 1] = az1;
                        bz_chunk[off] = bz0;
                        bz_chunk[off + 1] = bz1;
                    }
                    let e_out = split_eq_poly.E_out_current()[x_out_val];
                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
                    let reduced_inf = F::from_montgomery_reduce::<9>(inner_sum_inf);
                    acc0 += e_out.mul_unreduced::<9>(reduced0);
                    acci += e_out.mul_unreduced::<9>(reduced_inf);
                    (acc0, acci)
                },
            )
            .reduce(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        (
            F::from_montgomery_reduce::<9>(t0_acc_unr),
            F::from_montgomery_reduce::<9>(t_inf_acc_unr),
            DensePolynomial::new(az_bound),
            DensePolynomial::new(bz_bound),
        )
    }
    fn compute_first_quadratic_evals_and_bound_polys_new(
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        lagrange_evals_r: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        split_eq_poly: &GruenSplitEqPolynomial<F>,
    ) -> (
        F,
        F,
        Option<DensePolynomial<F>>,
        Option<DensePolynomial<F>>,
        Option<Vec<F>>,
    ) {
        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();
        let groups_exact = num_x_out_vals
            .checked_mul(num_x_in_vals)
            .expect("overflow computing groups_exact");

        let mut az_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);
        let mut bz_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);

        const S_SIZE: usize = 3_usize.pow(WINDOW_WIDTH as u32);

        let (t0_acc_unr, t_inf_acc_unr, s) = az_bound
            .par_chunks_exact_mut(2 * num_x_in_vals)
            .zip(bz_bound.par_chunks_exact_mut(2 * num_x_in_vals))
            .enumerate()
            .fold(
                || {
                    (
                        F::Unreduced::<9>::zero(),
                        F::Unreduced::<9>::zero(),
                        [F::Unreduced::<9>::zero(); S_SIZE],
                    )
                },
                |(mut acc0, mut acci, mut s_acc), (x_out_val, (az_chunk, bz_chunk))| {
                    let mut inner_sum0 = F::Unreduced::<9>::zero();
                    let mut inner_sum_inf = F::Unreduced::<9>::zero();

                    for x_in_val in 0..num_x_in_vals {
                        let x_in_bit0 = x_in_val & 1;

                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);
                        let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                        let az0 = eval.az_at_r_first_group(lagrange_evals_r);
                        let bz0 = eval.bz_at_r_first_group(lagrange_evals_r);
                        let az1 = eval.az_at_r_second_group(lagrange_evals_r);
                        let bz1 = eval.bz_at_r_second_group(lagrange_evals_r);

                        let p0 = az0 * bz0;
                        let p1 = az1 * bz1;
                        let p_inf = (az1 - az0) * (bz1 - bz0);

                        let e_in = split_eq_poly.E_in_current()[x_in_val];
                        let e_out = split_eq_poly.E_out_current()[x_out_val];
                        let eq_weight = e_in * e_out;

                        s_acc[0 * 3 + x_in_bit0] += eq_weight.mul_unreduced::<9>(p0);
                        s_acc[1 * 3 + x_in_bit0] += eq_weight.mul_unreduced::<9>(p1);
                        s_acc[2 * 3 + x_in_bit0] += eq_weight.mul_unreduced::<9>(p_inf);

                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sum_inf += e_in.mul_unreduced::<9>(p_inf);

                        let off = 2 * x_in_val;
                        az_chunk[off] = az0;
                        az_chunk[off + 1] = az1;
                        bz_chunk[off] = bz0;
                        bz_chunk[off + 1] = bz1;
                    }

                    let e_out = split_eq_poly.E_out_current()[x_out_val];
                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
                    let reduced_inf = F::from_montgomery_reduce::<9>(inner_sum_inf);
                    acc0 += e_out.mul_unreduced::<9>(reduced0);
                    acci += e_out.mul_unreduced::<9>(reduced_inf);

                    (acc0, acci, s_acc)
                },
            )
            .reduce(
                || {
                    (
                        F::Unreduced::<9>::zero(),
                        F::Unreduced::<9>::zero(),
                        [F::Unreduced::<9>::zero(); S_SIZE],
                    )
                },
                |(a0, ai, mut s1), (b0, bi, s2)| {
                    for i in 0..S_SIZE {
                        s1[i] += s2[i];
                    }
                    (a0 + b0, ai + bi, s1)
                },
            );

        // Reduce S
        let mut s_reduced = [F::zero(); S_SIZE];
        for i in 0..S_SIZE {
            s_reduced[i] = F::from_montgomery_reduce::<9>(s[i]);
        }

        // Verify s_reduced by recomputing from az_bound and bz_bound
        let mut s_verify = [F::zero(); S_SIZE];
        for x_out_val in 0..num_x_out_vals {
            for x_in_val in 0..num_x_in_vals {
                let x_in_bit0 = x_in_val & 1;
                let e_in = split_eq_poly.E_in_current()[x_in_val];
                let e_out = split_eq_poly.E_out_current()[x_out_val];
                let eq_weight = e_in * e_out;

                let base_idx = x_out_val * (2 * num_x_in_vals) + 2 * x_in_val;

                let az0 = az_bound[base_idx];
                let az1 = az_bound[base_idx + 1];
                let bz0 = bz_bound[base_idx];
                let bz1 = bz_bound[base_idx + 1];

                let p0 = az0 * bz0;
                let p1 = az1 * bz1;
                let p_inf = (az1 - az0) * (bz1 - bz0);

                s_verify[0 * 3 + x_in_bit0] += eq_weight * p0;
                s_verify[1 * 3 + x_in_bit0] += eq_weight * p1;
                s_verify[2 * 3 + x_in_bit0] += eq_weight * p_inf;
            }
        }

        // Assert each element of s_reduced matches
        for i in 0..S_SIZE {
            assert_eq!(s_reduced[i], s_verify[i], "s_reduced[{}] mismatch", i);
        }

        // Marginalize to get t(0), t(∞)
        let t0_from_window = s_reduced[0 * 3 + 0] + s_reduced[0 * 3 + 1];
        let t_inf_from_window = s_reduced[2 * 3 + 0] + s_reduced[2 * 3 + 1];

        // Assert t(0) and t(∞)
        assert_eq!(
            F::from_montgomery_reduce::<9>(t0_acc_unr),
            t0_from_window,
            "t0 mismatch"
        );
        assert_eq!(
            F::from_montgomery_reduce::<9>(t_inf_acc_unr),
            t_inf_from_window,
            "t_inf mismatch"
        );

        (
            //F::from_montgomery_reduce::<9>(t0_acc_unr),
            t0_from_window,
            //F::from_montgomery_reduce::<9>(t_inf_acc_unr),
            t_inf_from_window,
            Some(DensePolynomial::new(az_bound)),
            Some(DensePolynomial::new(bz_bound)),
            Some(s_reduced.to_vec()),
        )
    }
    // No special binding path needed; az/bz hold interleaved [lo,hi] ready for binding

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
        let eq_poly = &self.split_eq_poly;

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
    fn remaining_quadratic_evals_new(&mut self) -> (F, F) {
        let s_ext = self.s_reduced.as_ref().expect("s_reduced should exist");
        let current_size = s_ext.len();

        if current_size == 1 {
            println!("Handle S recomputation");
            return (F::zero(), F::zero());
        }

        let eq_poly = &self.split_eq_poly;
        let num_vars = (current_size as f64).log(3.0).round() as usize;
        let other_vars_size = 3_usize.pow((num_vars - 1) as u32);

        // Apply E_out weights when marginalizing
        let mut t0_unr = F::Unreduced::<9>::zero();
        let mut t_inf_unr = F::Unreduced::<9>::zero();

        for i in 0..other_vars_size {
            // Get eq weight
            let eq_weight = if num_vars == 1 {
                eq_poly.E_out_current()[0] // Single variable case
            } else {
                eq_poly.E_in_current()[i] // Multiple variables
            };

            let val_0 = s_ext[i];
            let val_inf = s_ext[2 * other_vars_size + i];

            t0_unr += eq_weight.mul_unreduced::<9>(val_0);
            t_inf_unr += eq_weight.mul_unreduced::<9>(val_inf);
        }

        (
            F::from_montgomery_reduce::<9>(t0_unr),
            F::from_montgomery_reduce::<9>(t_inf_unr),
        )
    }
    fn bind_extended_grid(&mut self, r: F::Challenge) {
        let s_ext = self.s_reduced.as_ref().expect("s_reduced should exist");
        let current_size = s_ext.len();
        let num_vars = (current_size as f64).log(3.0).round() as usize;

        if num_vars == 0 {
            return; // Already fully bound
        }

        let new_size = 3_usize.pow((num_vars - 1) as u32);
        let mut new_s_ext = vec![F::zero(); new_size];

        for new_idx in 0..new_size {
            let idx_0 = new_idx;
            let idx_1 = new_idx + new_size;
            let idx_inf = new_idx + 2 * new_size;

            let val_0 = s_ext[idx_0];
            let val_1 = s_ext[idx_1];
            let val_inf = s_ext[idx_inf];

            // f(r) = val_0 * (1-r) + val_1 * r + val_inf * r(r-1)
            new_s_ext[new_idx] = val_0 * (F::one() - r) + val_1 * r + val_inf * r * (r - F::one());
        }

        self.s_reduced = Some(new_s_ext);
    }

    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let az = self.az.as_ref().expect("az should be initialized");
        let bz = self.bz.as_ref().expect("bz should be initialized");

        let az0 = if !az.is_empty() { az[0] } else { F::zero() };
        let bz0 = if !bz.is_empty() { bz[0] } else { F::zero() };
        [az0, bz0]
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for OuterRemainingSumcheckProver<F> {
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
        let (t0, t_inf) = if round == 0 {
            self.first_round_evals
        } else {
            if self.params.windows.contains(&round) {
                println!("Start of window: {:?}; recompute", round);
                let (t0, t_inf) = self.remaining_quadratic_evals();
                (t0, t_inf)
            } else {
                let last_window_start = *self.params.windows.last().unwrap();
                let last_window_end = last_window_start + WINDOW_WIDTH - 2;
                if round > last_window_end {
                    println!("Linear sumcheck : TODO! {:?}", round);
                    self.remaining_quadratic_evals()
                } else {
                    // Case 3: In-between
                    println!("I am in the middle: {:?}", round);
                    let (t0, t_inf) = self.remaining_quadratic_evals();
                    let (debug_0, debug_inf) = self.remaining_quadratic_evals_new();
                    println!("{:?}, {:?}", t0, debug_0);
                    println!("{:?}, {:?}", t_inf, debug_inf);
                    (t0, t_inf)
                }
            }
        };
        let evals = self
            .split_eq_poly
            .gruen_evals_deg_3(t0, t_inf, previous_claim);
        vec![evals[0], evals[1], evals[2]]
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
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
        // Bind the extended grid for S
        self.bind_extended_grid(r_j);

        // Bind eq_poly for next round
        self.split_eq_poly.bind(r_j);
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
    // windows
    windows: Vec<usize>,
}

impl<F: JoltField> OuterRemainingSumcheckParams<F> {
    fn new(num_cycles_bits: usize, uni: &UniSkipState<F>) -> Self {
        let first_half_end: usize = (num_cycles_bits + 1) / 2;
        let windows = (0..first_half_end).step_by(WINDOW_WIDTH).collect(); // Keep 0-indexed: [0, 2, 4, ...]
        Self {
            num_cycles_bits,
            tau: uni.tau.clone(),
            r0_uniskip: uni.r0,
            input_claim: uni.claim_after_first,
            windows,
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
