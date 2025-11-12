use allocative::Allocative;
use ark_std::Zero;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};
use std::usize;
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
use crate::poly::split_eq_poly_generalised::GruenSplitEqPolynomialGeneral;
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
const WINDOW_WIDTH: usize = 3;
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
pub struct OuterRemainingSumcheckProver<F: JoltField> {
    #[allocative(skip)]
    preprocess: Arc<JoltSharedPreprocessing>,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    split_eq_poly: GruenSplitEqPolynomial<F>,
    split_eq_poly_gen: GruenSplitEqPolynomialGeneral<F>,
    // We still have these but their sizes will change
    az: Option<DensePolynomial<F>>,
    bz: Option<DensePolynomial<F>>,
    t_prime_grid: RwLock<Option<Vec<F>>>, // Interior mutability for just this field
    /// The first round evals (t0, t_inf) computed from a streaming pass over the trace
    first_round_evals: (F, F),
    #[allocative(skip)]
    params: OuterRemainingSumcheckParams<F>,
    lagrange_evals_r0: [F; 10],
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

        let split_eq_poly_gen: GruenSplitEqPolynomialGeneral<F> =
            GruenSplitEqPolynomialGeneral::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
                WINDOW_WIDTH,
            );

        let (t0, t_inf, az_bound, bz_bound) = Self::compute_first_quadratic_evals_and_bound_polys(
            &preprocessing.shared,
            trace,
            &lagrange_evals_r,
            &split_eq_poly,
        );

        Self {
            split_eq_poly,
            split_eq_poly_gen,
            preprocess: Arc::new(preprocessing.shared.clone()),
            trace: Arc::new(trace.to_vec()),
            az: Some(az_bound), // TB deleted
            bz: Some(bz_bound), // To be deleted
            t_prime_grid: RwLock::new(None),
            first_round_evals: (t0, t_inf),
            params: outer_params,
            lagrange_evals_r0: lagrange_evals_r,
        }
    }

    /// Compute the window lenght give current round_idx
    pub fn get_window_with(&self, round_idx: usize) -> usize {
        round_idx
    }

    fn digitize(&self, mut i: usize, b: usize, digits: usize) -> Vec<u32> {
        let mut ans = vec![0u32; digits];
        for idx in (0..digits).rev() {
            ans[idx] = (i % b) as u32;
            i /= b;
        }
        ans
    }

    fn bind_first_variable_in_place(&self, r: F::Challenge, w: usize) {
        let mut grid_guard = self.t_prime_grid.write().unwrap();
        if let Some(ref mut t_grid) = *grid_guard {
            let new_size = 3_usize.pow((w - 1) as u32);
            let mut t_grid_bound = vec![F::zero(); new_size];

            // For each point in the new (w-1)-dimensional grid
            for new_idx in 0..new_size {
                // Since first coord is most significant and changes every new_size elements:
                let eval_at_0 = t_grid[new_idx]; // z0 = 0
                let eval_at_1 = t_grid[new_idx + new_size]; // z0 = 1
                let eval_at_inf = t_grid[new_idx + 2 * new_size]; // z0 = ∞

                // Interpolate and evaluate at r
                let one = F::one();
                t_grid_bound[new_idx] =
                    eval_at_0 * (one - r) + eval_at_1 * r + eval_at_inf * r * (r - one);
            }

            *t_grid = t_grid_bound;
        }
    }
    /// returns the grid of evaluations on 3^window_size
    fn get_grid_aux(
        &self,
        split_eq_poly: &GruenSplitEqPolynomialGeneral<F>,
        window_size: usize,
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        lagrange_evals_r: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
    ) {
        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let num_x_in_vars = num_x_in_vals.log_2();
        let num_x_out_vars = num_x_out_vals.log_2();

        let num_rs = split_eq_poly.num_challenges();
        let num_active_evals = (0..window_size).fold(1, |acc, _| acc * 3);
        let mut ans: Vec<F> = vec![F::zero(); num_active_evals];

        for a_idx in 0..num_active_evals {
            let z_vec = self.digitize(a_idx, 3, window_size);
            let mut accumulator = F::zero(); // Unreduced accumulator
            for (out_idx, out_val) in split_eq_poly.E_out_current().iter().enumerate() {
                let x_out_vec = self.digitize(out_idx, 2, num_x_out_vars);
                for (in_idx, in_val) in split_eq_poly.E_in_current().iter().enumerate() {
                    let x_in_vec = self.digitize(in_idx, 2, num_x_in_vars);

                    for r_idx in 0..split_eq_poly.num_challenges().max(1) {
                        let r_vec = if split_eq_poly.num_challenges() > 0 {
                            self.digitize(r_idx, 2, split_eq_poly.num_challenges())
                        } else {
                            vec![] // Empty vec when no challenges
                        };
                        // a_idx is a 3^windown size
                        // time_step_idx = out_idx_in_bits || in_idx_bits || a_idx_in_bits ||
                        // r_idx_in_bits
                        let mut inf_indices: Vec<usize> = Vec::new();
                        for z_idx in 0..z_vec.len() {
                            if z_vec[z_idx] == 2 {
                                inf_indices.push(z_idx);
                            }
                        }

                        let num_infs = inf_indices.len();
                        let mut A_mess = F::zero();
                        let mut B_mess = F::zero();
                        for f_in in 0..(1 << num_infs) {
                            let f_vec = self.digitize(f_in, 2, num_infs);
                            let mut new_z_vec = Vec::new();
                            let mut curr_f_index = 0;
                            for z_in in 0..z_vec.len() {
                                if z_vec[z_in] == INFINITY as u32 {
                                    new_z_vec.push(f_vec[curr_f_index]);
                                    curr_f_index += 1;
                                } else {
                                    new_z_vec.push(z_vec[z_in]);
                                }
                            }

                            let (index_vec, selector) = if split_eq_poly.num_challenges() > 0 {
                                // Concatenate: x_out_vec || x_in_vec || new_z_vec || r_vec[1:]
                                let mut idx_vec = Vec::new();
                                idx_vec.extend_from_slice(&x_out_vec);
                                idx_vec.extend_from_slice(&x_in_vec);
                                idx_vec.extend_from_slice(&new_z_vec);
                                idx_vec.extend_from_slice(&r_vec[1..]);
                                if r_vec[0] == 0 {
                                    (idx_vec, false)
                                } else {
                                    (idx_vec, true)
                                }
                            } else {
                                // Concatenate: x_out_vec || x_in_vec || new_z_vec[1:]
                                let mut idx_vec = Vec::new();
                                idx_vec.extend_from_slice(&x_out_vec);
                                idx_vec.extend_from_slice(&x_in_vec);
                                idx_vec.extend_from_slice(&new_z_vec[1..]);

                                if new_z_vec[0] == 0 {
                                    (idx_vec, false)
                                } else {
                                    (idx_vec, true)
                                }
                            };

                            // Convert binary vector to single index
                            let current_step_idx = index_vec
                                .iter()
                                .fold(0usize, |acc, &bit| (acc << 1) | (bit as usize));
                            //println!("  out_idx={}, in_idx={}", out_idx, in_idx);
                            //println!(
                            //    "    x_out_vec={:?} (should represent bits for w_out)",
                            //    x_out_vec
                            //);
                            //println!(
                            //    "    x_in_vec={:?} (should represent bits for w_in)",
                            //    x_in_vec
                            //);
                            //println!(
                            //    "    z_vec[1..]={:?} (should represent window vars)",
                            //    &z_vec[1..]
                            //);
                            //println!("    Full idx_vec={:?}", index_vec);
                            //println!("    Trace index={}", current_step_idx);
                            //println!("    E_out[{}] represents what polynomial eval?", out_idx);
                            //println!("    E_in[{}] represents what polynomial eval?", in_idx);
                            let (az, bz) = self.get_az_bz_at_curr_timestep(
                                current_step_idx,
                                selector,
                                preprocess,
                                trace,
                                lagrange_evals_r,
                            );
                            //println!("Current time idx: {current_step_idx} Selector: {selector}");
                            //println!("Az: {:?}", az);
                            let num_zeros = f_vec.iter().filter(|&&v| v == 0).count();
                            if num_zeros % 2 == 0 {
                                // Positive contribution
                                A_mess = A_mess + az;
                                B_mess = B_mess + bz;
                            } else {
                                // Negative contribution
                                A_mess = A_mess - az;
                                B_mess = B_mess - bz;
                            }
                        }

                        // Also need to handle eq polynomial for r if there are challenges
                        let eq_r = if split_eq_poly.num_challenges() > 0 {
                            // Compute eq(r_vec, challenges)
                            //self.eq(&r_vec, split_eq_poly.get_challenges())
                            todo!("hAndle the binding business")
                        } else {
                            F::one()
                        };

                        let product = A_mess * B_mess * eq_r * out_val * in_val;
                        accumulator = accumulator + product;
                    }
                }
            }
            ans[a_idx] = accumulator;
        }
        *self.t_prime_grid.write().unwrap() = Some(ans);
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

    fn compute_first_quadratic_evals_and_bound_polys(
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
            //.par_chunks_exact_mut(2 * num_x_in_vals)
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
                        // A(inf)B(inf) = (A(1) - A(0)) * (B(1) - B(0)) = (p + q - p) * (t + s - s) = tq
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

    fn project_to_single_var(
        &self,
        t_grid: &[F],
        E_active: &[F],
        w: usize,
        first_coord_val: usize,
    ) -> F {
        let offset = first_coord_val * 3_usize.pow((w - 1) as u32);

        E_active
            .iter()
            .enumerate()
            .map(|(eq_active_idx, eq_active_val)| {
                let mut index = offset;
                let mut temp = eq_active_idx;
                let mut power = 1;
                for _ in 0..(w - 1) {
                    if temp & 1 == 1 {
                        index += power;
                    }
                    power *= 3;
                    temp >>= 1;
                }
                t_grid[index] * *eq_active_val
            })
            .sum()
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
            self.get_grid_aux(
                &self.split_eq_poly_gen,
                WINDOW_WIDTH,
                &self.preprocess,
                &self.trace,
                &self.lagrange_evals_r0,
            );

            println!("Computing prover message for round={}", round);
            let tmp = self.t_prime_grid.read().unwrap();
            let t_prime_grid = tmp.as_ref().expect("t_grid be initialised by now");
            let E_active = self.split_eq_poly_gen.E_active_current();
            // Sum over all Z2 values when Z1=0
            // NOTE: (ari) i have this code that generalises this for generic window sizes
            // this is for debug purposes only!!!
            //let t_prime_0 = t_grid[0] * E_active[0]
            //    + t_grid[1] * E_active[1]
            //    + t_grid[3] * E_active[2]
            //    + t_grid[4] * E_active[3];
            //
            //let t_prime_inf = t_grid[18] * E_active[0]
            //    + t_grid[19] * E_active[1]
            //    + t_grid[21] * E_active[2]
            //    + t_grid[22] * E_active[3];
            let t_prime_0 = self.project_to_single_var(t_prime_grid, E_active, WINDOW_WIDTH, 0);
            let t_prime_inf =
                self.project_to_single_var(t_prime_grid, E_active, WINDOW_WIDTH, INFINITY);

            let (t0, t_inf) = self.first_round_evals;
            assert_eq!(t0, t_prime_0);
            assert_eq!(t_inf, t_prime_inf);
            println!("Round {} prover message success!", round);
            (t_prime_0, t_prime_inf)
        } else {
            // JUST DEBUGGING
            if round == 1 {
                println!("Computing prover message for round={}", round);
                let t_grid = self.t_prime_grid.read().unwrap();
                let grid_ref = t_grid
                    .as_ref()
                    .expect("t_grid be initialised by now, and shrunk in size");
                //let t_prime_0 = grid_ref[0]; // THIS WORKS WHEN the WIDTH was 2.
                //let E_active = &self.split_eq_poly_gen.E_active_current();
                //let t_prime_0 =
                //    self.project_to_single_var(grid_ref, E_active, WINDOW_WIDTH - round, 0);
                //let _t_prime_inf =
                //    self.project_to_single_var(grid_ref, E_active, WINDOW_WIDTH - round, INFINITY);
                println!("After binding, grid check:");
                println!("grid[0] (z1=0,z2=0) = {:?}", grid_ref[0]);
                println!("grid[1] (z1=0,z2=1) = {:?}", grid_ref[1]);
                println!("grid[3] (z1=1,z2=0) = {:?}", grid_ref[3]);
                println!("grid[4] (z1=1,z2=1) = {:?}", grid_ref[4]);

                // Manually check the interpolation for grid[0]
                let w_idx = 14;
                let t_prime_0 = grid_ref[0] * (F::one() - self.split_eq_poly_gen.w[w_idx])
                    + grid_ref[1] * self.split_eq_poly_gen.w[w_idx];

                let (t0, t_inf) = self.remaining_quadratic_evals();
                assert_eq!(t0, t_prime_0, "t0 != t_prime_0");
                (t0, t_inf)
            } else {
                // All other rounds for now
                let (t0, t_inf) = self.remaining_quadratic_evals();
                (t0, t_inf)
            }
        };

        let evals = self
            .split_eq_poly
            .gruen_evals_deg_3(t0, t_inf, previous_claim);
        if round == 0 {
            let evals_grid = self
                .split_eq_poly_gen
                .gruen_evals_deg_3(t0, t_inf, previous_claim);
            assert_eq!(evals_grid[0], evals[0]);
        }
        vec![evals[0], evals[1], evals[2]]
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
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
        // Testing special bind
        // in the first round it'll make t_grid_smaller from 27 to 9
        // and then i also need to take off the w0 challenge from eq_poly.
        // in the next round z1 will be the active var and z2 is the
        if round == 0 {
            println!("Binding r_{:?}", round);
            self.bind_first_variable_in_place(r_j, WINDOW_WIDTH - round);
            self.split_eq_poly_gen.bind(r_j);
        }
        // Bind eq_poly for next round
        self.split_eq_poly.bind(r_j);
        if round == 1 || self.params.windows.contains(&round) {
            println!(
                "Time to re-create stream data structure for round: {}",
                round
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
    // windows
    windows: Vec<usize>,
}

impl<F: JoltField> OuterRemainingSumcheckParams<F> {
    fn new(num_cycles_bits: usize, uni: &UniSkipState<F>) -> Self {
        let first_half_end: usize = (num_cycles_bits + 1) / 2;
        let windows = (1..first_half_end).step_by(WINDOW_WIDTH).collect(); // If window size 2 Keep 0-indexed: [1, 3, 5, ...]
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
