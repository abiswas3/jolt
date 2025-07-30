use super::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck, SumcheckInstanceProof};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::unsafe_allocate_zero_vec,
        transcript::{AppendToTranscript, Transcript},
    },
};
use rayon::prelude::*;

pub struct ShoutProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    core_piop_claims: ShoutSumcheckClaims<F>,
    ra_claim_prime: F,
}

struct ShoutProverState<F: JoltField> {
    K: usize,
    rv_claim: F,
    z: F,
    ra: MultilinearPolynomial<F>,
    val: MultilinearPolynomial<F>,
}

impl<F: JoltField> ShoutProverState<F> {
    #[tracing::instrument(skip_all)]
    fn initialize<ProofTranscript: Transcript>(
        lookup_table: Vec<F>,
        read_addresses: &[usize],
        r_cycle: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let K = lookup_table.len();
        let T = read_addresses.len();
        debug_assert_eq!(r_cycle.len(), T.log_2());
        // Used to batch the core PIOP sumcheck and Hamming weight sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();

        let E: Vec<F> = EqPolynomial::evals(r_cycle);

        let span = tracing::span!(tracing::Level::INFO, "compute F");
        let _guard = span.enter();

        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);
        let F: Vec<_> = read_addresses
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, addresses)| {
                let mut result: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut cycle = chunk_index * chunk_size;
                for address in addresses {
                    result[*address] += E[cycle];
                    cycle += 1;
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );
        drop(_guard);
        drop(span);

        let rv_claim: F = F
            .par_iter()
            .zip(lookup_table.par_iter())
            .map(|(&ra, &val)| ra * val)
            .sum();

        let ra = MultilinearPolynomial::from(F.clone());
        let val = MultilinearPolynomial::from(lookup_table);

        let prover_state = Self {
            K,
            z,
            rv_claim,
            ra,
            val,
        };
        (prover_state, E, F)
    }
}

#[derive(Clone)]
struct ShoutSumcheckClaims<F: JoltField> {
    ra_claim: F,
    rv_claim: F,
}

struct ShoutVerifierState<F: JoltField> {
    K: usize,
    z: F,
    val: MultilinearPolynomial<F>,
}

impl<F: JoltField> ShoutVerifierState<F> {
    fn initialize<ProofTranscript: Transcript>(
        lookup_table: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = lookup_table.len();
        let z: F = transcript.challenge_scalar();
        let val = MultilinearPolynomial::from(lookup_table);
        Self { K, z, val }
    }
}

struct ShoutSumcheck<F: JoltField> {
    verifier_state: Option<ShoutVerifierState<F>>,
    prover_state: Option<ShoutProverState<F>>,
    claims: Option<ShoutSumcheckClaims<F>>,
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for ShoutSumcheck<F>
{
    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().K.log_2()
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().K.log_2()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        if self.prover_state.is_some() {
            let ShoutProverState { rv_claim, z, .. } = self.prover_state.as_ref().unwrap();
            // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
            *rv_claim + z
        } else if self.verifier_state.is_some() {
            let ShoutVerifierState { z, .. } = self.verifier_state.as_ref().unwrap();
            let ShoutSumcheckClaims { rv_claim, .. } = self.claims.as_ref().unwrap();
            // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
            *rv_claim + z
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&mut self, _: usize, _previous_claim: F) -> Vec<F> {
        let ShoutProverState { ra, val, z, .. } = self.prover_state.as_ref().unwrap();

        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);

        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                [
                    ra_evals[0] * (*z + val_evals[0]),
                    ra_evals[1] * (*z + val_evals[1]),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let ShoutProverState { ra, val, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.claims.is_none());
        let ShoutProverState { rv_claim, ra, .. } = self.prover_state.as_ref().unwrap();
        self.claims = Some(ShoutSumcheckClaims {
            ra_claim: ra.final_sumcheck_claim(),
            rv_claim: *rv_claim,
        });
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ShoutVerifierState { z, val, .. } = self.verifier_state.as_ref().unwrap();
        let ShoutSumcheckClaims { ra_claim, .. } = self.claims.as_ref().unwrap();

        let r_address: Vec<F> = r.iter().rev().copied().collect();
        *ra_claim * (*z + val.evaluate(&r_address))
    }
}

impl<F: JoltField, ProofTranscript: Transcript> ShoutProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "ShoutProof::prove")]
    pub fn prove(
        lookup_table: Vec<F>,
        read_addresses: Vec<usize>,
        r_cycle: &[F],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let (core_piop_prover_state, E, F) =
            ShoutProverState::initialize(lookup_table, &read_addresses, r_cycle, transcript);
        let booleanity_prover_state =
            BooleanityProverState::initialize(read_addresses, E, F, transcript);

        let mut core_piop_sumcheck = ShoutSumcheck {
            prover_state: Some(core_piop_prover_state),
            verifier_state: None,
            claims: None,
        };

        let mut booleanity_sumcheck = BooleanitySumcheck {
            prover_state: Some(booleanity_prover_state),
            verifier_state: None,
            ra_claim: None,
        };

        let (sumcheck_proof, _r_sumcheck) = BatchedSumcheck::prove(
            vec![&mut core_piop_sumcheck, &mut booleanity_sumcheck],
            transcript,
        );

        let core_piop_claims = core_piop_sumcheck.claims.unwrap();

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Self {
            sumcheck_proof,
            core_piop_claims,
            ra_claim_prime: booleanity_sumcheck.ra_claim.unwrap(),
        }
    }

    pub fn verify(
        &self,
        lookup_table: Vec<F>,
        r_cycle: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let K = lookup_table.len();

        let core_piop_verifier_state = ShoutVerifierState::initialize(lookup_table, transcript);
        let booleanity_verifier_state = BooleanityVerifierState::initialize(r_cycle, K, transcript);

        let core_piop_sumcheck = ShoutSumcheck {
            prover_state: None,
            verifier_state: Some(core_piop_verifier_state),
            claims: Some(self.core_piop_claims.clone()),
        };

        let booleanity_sumcheck = BooleanitySumcheck {
            prover_state: None,
            verifier_state: Some(booleanity_verifier_state),
            ra_claim: Some(self.ra_claim_prime),
        };

        let _r_sumcheck = BatchedSumcheck::verify(
            &self.sumcheck_proof,
            vec![&core_piop_sumcheck, &booleanity_sumcheck],
            transcript,
        )?;

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Ok(())
    }
}

/// Implements the sumcheck prover for the generic core Shout PIOP for d=1.
/// See Figure 7 of https://eprint.iacr.org/2025/105
pub fn prove_generic_core_shout_pip_d_greater_than_one<
    F: JoltField,
    ProofTranscript: Transcript,
>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    d: usize,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    Vec<F>,
    F,
    F,
    F,
    F,
) {
    // This assumes that K and T are powers of 2
    let K = lookup_table.len();
    let T = read_addresses.len();
    let N = (K as f64).powf(1.0 / d as f64).round() as usize;
    // A random field element F^{\log_2 T} for Schwartz-Zippll
    // This is stored in Big Endian
    let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
    // Page 50: eq(44)
    let E_star: Vec<F> = EqPolynomial::evals(&r_cycle);
    // Page 50: eq(47) : what the paper calls v_k
    let C: Vec<_> = (0..K) // This is C[x] = ra(r_cycle, x)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| {
                    if *address == k {
                        // this check will be more complex for d > 1 but let's keep
                        // this for now
                        Some(E_star[cycle])
                    } else {
                        None
                    }
                })
                .sum::<F>()
        })
        .collect();

    let num_rounds = K.log_2() + T.log_2();
    // The vector storing the verifiers sum-check challenges
    let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

    // The sum check answer (for d=1, it's the same as normal one)
    let sumcheck_claim: F = C
        .par_iter()
        .zip(lookup_table.par_iter())
        .map(|(&ra, &val)| ra * val)
        .sum();

    let mut previous_claim = sumcheck_claim;

    // These are the polynomials the prover commits to
    let mut ra = MultilinearPolynomial::from(C);
    let mut val = MultilinearPolynomial::from(lookup_table);

    // Binding the first log_2 K variables
    const DEGREE_ADDR: usize = 2;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..K.log_2() {
        // Page 51: (eq 51)
        let univariate_poly_evals: [F; DEGREE_ADDR] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals = ra.sumcheck_evals(index, DEGREE_ADDR, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(index, DEGREE_ADDR, BindingOrder::LowToHigh);
                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]] // since DEGREE_ADDR=2
            })
            .reduce(
                || [F::zero(); DEGREE_ADDR],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        // Construct coefficients of univariate polynomial from evaluations
        // No Gruen optimisation here for now
        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        // Get challenge that binds the variable
        let r_j = transcript.challenge_scalar::<F>();
        //let r_j = match debug_count {
        //    0 => F::from_u8(2),
        //    1 => F::from_u8(3),
        //    2 => F::from_u8(4),
        //    _ => F::one(), // fallback case if needed
        //};
        r_address.push(r_j);
        previous_claim = univariate_poly.evaluate(&r_j);
        //println!("r_address[{debug_count}] = {r_j}");
        //println!("g_{debug_count}[{r_j}]={previous_claim}");

        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    // tau = r_address (the verifiers challenges which bind all log K variables of memory)
    // This is \widetilde{Val}(\tau) from the paper (eq 52)
    let val_claim = val.final_sumcheck_claim();

    // At this point we should have bound the first log K variables

    // Making E_star into a ML poly
    let mut eq_r_cycle = MultilinearPolynomial::from(E_star);

    // As d > 1, we will have d arrays each of length T
    let eq_taus: Vec<Vec<F>> = compute_eq_taus_parallel(&r_address, d, N.log_2());

    // This is the same E as the one referenced on Page 51 of Shetty/Thaler
    let mut E: Vec<Vec<F>> = vec![vec![F::zero(); T]; d];
    // Filling out E involve any multiplications
    // Just lookups
    // iterate of each table e_j in parallel
    E.par_iter_mut().enumerate().for_each(|(j, e_j)| {
        // for a fixed table e_j iterate through all of time stamps
        e_j.par_iter_mut().enumerate().for_each(|(y, e)| {
            // take the memory cell to be read at time y and extrac j'th digit: addr_j
            // (which is also the index in array e_j[y: addr_j])
            // Here when j=0 we get the MSB and when j=d-1 we get the LSB
            let addr_j = digit_j_of(read_addresses[y], j, d, N);
            // Since eq_taus[0] contains the first log N bits of of read_address
            // we adjust the indexing
            *e = eq_taus[d - j - 1][addr_j];
        });
    });
    // FIX later -- this because extact last digits first so E[j] updates is being mapped to eq_taus[d-j-1]
    E.reverse();
    // The E tables will be dropped once we make ra_taus
    let mut ra_taus: Vec<MultilinearPolynomial<F>> =
        E.into_par_iter().map(MultilinearPolynomial::from).collect();

    let DEGREE_TME: usize = d + 2;
    for _ in 0..T.log_2() {
        let univariate_poly_evals: Vec<F> = (0..ra_taus[0].len() / 2)
            .into_par_iter()
            .map(|index| {
                let eq_r_cycle_evals =
                    eq_r_cycle.sumcheck_evals(index, DEGREE_TME, BindingOrder::LowToHigh);
                // For each of the d ra_taus we should get d sumcheck evals as the evaluation at 1
                // is constructed from the previous claim
                // This happens only d times so there's no need to parallelise
                let ra_evals_per_tau: Vec<Vec<F>> = ra_taus
                    .iter()
                    .map(|ra_tau| ra_tau.sumcheck_evals(index, DEGREE_TME, BindingOrder::LowToHigh))
                    .collect();

                // The parallelisation should be over ra_evals_per_tau which can be as
                // large as ra_taus[0].len()/2 = T/2 initially.
                // It shrinks by half at each round
                // TODO: once the size of ra_taus[0] is small enough
                // we should swtich from par_iter to iter.
                let result: Vec<F> = (0..DEGREE_TME)
                    .map(|i| {
                        let col_product = ra_evals_per_tau
                            .par_iter()
                            .map(|row| row[i])
                            .reduce(|| F::one(), |a, b| a * b);

                        col_product * eq_r_cycle_evals[i]
                    })
                    .collect();
                result
            })
            .reduce(
                || vec![F::zero(); DEGREE_TME],
                |running, new| {
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(a, b)| *a + *b)
                        .collect()
                },
            );

        let d_plus_one_evaluations = construct_final_sumcheck_evals(
            &univariate_poly_evals,
            val_claim,
            previous_claim,
            DEGREE_TME,
        );
        let univariate_poly = UniPoly::from_evals(&d_plus_one_evaluations);

        // Skip the linear term when storing coeffs as we can always re-construct it
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        // Get challenge that binds the variable
        let r_j = transcript.challenge_scalar::<F>();
        //let r_j = match debug_count {
        //    0 => F::from_u8(2),
        //    1 => F::from_u8(3),
        //    2 => F::from_u8(4),
        //    _ => F::one(), // fallback case if needed
        //};

        r_address.push(r_j);

        rayon::join(
            || {
                ra_taus.par_iter_mut().for_each(|ra_tau| {
                    ra_tau.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            },
            || eq_r_cycle.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ras_raddress_rtime_product: F = ra_taus
        .par_iter()
        .map(|ra| ra.final_sumcheck_claim())
        .reduce(|| F::one(), |acc, val| acc * val);

    let eq_r_cycle_at_r_time = eq_r_cycle.final_sumcheck_claim();

    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        sumcheck_claim,
        ras_raddress_rtime_product,
        val_claim,
        eq_r_cycle_at_r_time,
    )
}

/// Constructs the evaluations of the final univariate polynomial for sumcheck in parallel.
///
/// - For `i == 1`: `result[1] = previous_claim - val_claim * univariate_poly_evals[0]`
/// - For other `i`: `result[i] = val_claim * univariate_poly_evals[i]`
///
/// # Arguments
/// - `univariate_poly_evals`: Vector of evaluations of the product polynomial (length `degree`)
/// - `val_claim`: val(r_address) which needs to be nmultiplied to each of the evals
/// - `previous_claim`: Claimed value for this round which is meant to equal result[0] + result[1]
///
/// # Returns
/// A vector `result` of length `degree+1` representing the evaluations of the final univariate polynomial
pub fn construct_final_sumcheck_evals<F: JoltField>(
    univariate_poly_evals: &[F],
    val_claim: F,
    previous_claim: F,
    degree: usize,
) -> Vec<F> {
    let first_term = val_claim
        * univariate_poly_evals
            .first()
            .expect("univariate_poly_evals must be non-empty");

    let result: Vec<_> = (0..=degree)
        .into_par_iter()
        .map(|i| match i {
            1 => previous_claim - first_term,
            _ => {
                let eval_idx = if i > 1 { i - 1 } else { i };
                val_claim * univariate_poly_evals[eval_idx]
            }
        })
        .collect();

    result
}

/// Let tau = r_address, the first log K challenges of the verifier.
/// Computes `d` decomposed evaluation tables of the multilinear equality polynomial `eq(tau, x)`
/// in parallel, given a flattened bitstring `tau ∈ F^{d * log_n}`.
///
/// Each `eq_tau[j]` corresponds to the evaluation of the multilinear equality
/// polynomial over chunk of [j*\log_2 N...(j+1)*\log_2 N]`tau`.
///
/// More precisely:
/// - For each `j ∈ {0, ..., d-1}`, the chunk `r_address[j * log_n .. (j + 1) * log_n]`
///   is reversed and passed into `EqPolynomial::evals()`
///   The reversal happens because EqPolynomial::evals accepts inputs in Big Endian,
///   But r_address is stored in little endian form.
/// - The result is a vector of `d` vectors, where each inner vector contains
///   `2^log_2 N` field elements corresponding to `eq(tau_j, x)` for all `x ∈ {0,1}^log_N`
///
/// # Arguments
///
/// * `r_address` - A slice of field elements of length `d * log_n`,
/// * `d` - The number of address chunks / dimensions (should divide `r_address.len()`).
/// * `log_n` - The number of bits per address chunk (i.e., chunk size).
///
/// # Returns
///
/// A `Vec<Vec<F>>` of length `d`. Each inner vector contains `2^log_n` evaluations
/// of `eq(tau_j, x)` over `x ∈ {0,1}^log_n`, for the j-th address chunk.
///
/// # Panics
///
/// Panics if `r_address.len() != d * log_n`.
///
/// # Parallelism
///
/// The outer loop over `j` is parallelized using `rayon`. Each equality polynomial
/// evaluation is performed independently on a separate chunk of `r_address`.
///
/// # Example
/// ```rust
/// let d = 2;
/// let log_n = 3; // so each chunk is 3 bits = 8 entries
/// let r_address = vec![F::zero(); d * log_n];
///
/// let eq_taus = compute_eq_taus_parallel(&r_address, d, log_n);
/// assert_eq!(eq_taus.len(), d);
/// assert_eq!(eq_taus[0].len(), 1 << log_n); // 8
/// ```
///
/// # Note
///
/// All inputs and evaluations use big-endian ordering for consistency with
/// the rest of the Jolt protocol.
///
/// # See Also
///
/// [`EqPolynomial::evals`] — the underlying function used to evaluate `eq(tau, x)`.
fn compute_eq_taus_parallel<F: JoltField>(
    r_address: &[F], // length must be d * log_N = \log K
    d: usize,
    log_n: usize,
) -> Vec<Vec<F>> {
    assert_eq!(r_address.len(), d * log_n);

    (0..d)
        .into_par_iter()
        .map(|j| {
            let start = j * log_n;
            let end = start + log_n;

            let mut tau_bits = r_address[start..end].to_vec();
            tau_bits.reverse(); // BigEndian

            EqPolynomial::evals(&tau_bits)
        })
        .collect()
}

/// Extracts the `j`-th digit (0-indexed from the most significant digit) of `addr`
/// when written in base `base`, assuming the total number of digits is `d`.
///
/// # Arguments
///
/// - `addr`: The integer address to decompose.
/// - `j`: The digit index (0 = most significant, `d - 1` = least significant).
/// - `d`: The total number of digits in the base-`base` representation.
/// - `base`: The numerical base (e.g., `K^{1/d}`).
///
/// # Returns
///
/// The `j`-th digit of `addr` in base `base`.
///
/// # Panics
///
/// Panics if `base.pow(d as u32)` overflows or if `j >= d`.
///
/// # Examples
///
/// ```rust
/// let addr = 10;
/// let base = 4;
/// let d = 2; // because 10 in base 4 is "2 2"
///
/// assert_eq!(digit_j(addr, 0, d, base), 2); // most significant digit
/// assert_eq!(digit_j(addr, 1, d, base), 2); // least significant digit
/// ```
fn digit_j_of(addr: usize, j: usize, d: usize, base: usize) -> usize {
    // Convert from most-significant-first index (0 = most significant)
    let exp = d - 1 - j;
    (addr / base.pow(exp as u32)) % base
}

/// Implements the sumcheck prover for the generic core Shout PIOP for d=1.
/// See Figure 7 of https://eprint.iacr.org/2025/105
pub fn prove_generic_core_shout_pip<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    _d: u32,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    Vec<F>,
    F,
    F,
    F,
    F,
) {
    // This assumes that K and T are powers of 2
    let K = lookup_table.len();
    let T = read_addresses.len();

    // A random field element F^{\log_2 T} for Schwartz-Zippll
    // This is stored in Big Endian
    let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
    // Page 50: eq(44)
    let E_star: Vec<F> = EqPolynomial::evals(&r_cycle);
    // Page 50: eq(47) : what the paper calls v_k
    let C: Vec<_> = (0..K) // This is C[x] = ra(r_cycle, x)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| {
                    if *address == k {
                        // this check will be more complex for d > 1 but let's keep
                        // this for now
                        Some(E_star[cycle])
                    } else {
                        None
                    }
                })
                .sum::<F>()
        })
        .collect();

    let num_rounds = K.log_2() + T.log_2();
    // The vector storing the verifiers sum-check challenges
    let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

    // The sum check answer (for d=1, it's the same as normal one)
    let sumcheck_claim: F = C
        .par_iter()
        .zip(lookup_table.par_iter())
        .map(|(&ra, &val)| ra * val)
        .sum();

    let mut previous_claim = sumcheck_claim;

    // These are the polynomials the prover commits to
    let mut ra = MultilinearPolynomial::from(C);
    let mut val = MultilinearPolynomial::from(lookup_table);

    // Binding the first log_2 K variables
    const DEGREE: usize = 2;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..K.log_2() {
        // Page 51: (eq 51)
        let univariate_poly_evals: [F; DEGREE] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals = ra.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]] // since DEGREE=2
            })
            .reduce(
                || [F::zero(); DEGREE],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        // Construct coefficients of univariate polynomial from evaluations
        // No Gruen optimisation here
        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        // Get challenge that binds the variable
        let r_j = transcript.challenge_scalar::<F>();
        r_address.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    // tau = r_address (the verifiers challenges which bind all log K variables of memory)
    // This is \widetilde{Val}(\tau) from the paper (eq 52)
    let val_claim = val.final_sumcheck_claim();

    // At this point we should have bound the first log K variables
    // Binding the second log T variables
    let mut eq_r_cycle = MultilinearPolynomial::from(E_star);
    // Endian issue
    let mut r_address_reversed = r_address.clone();
    r_address_reversed.reverse();
    let eq_tau: Vec<F> = EqPolynomial::evals(&r_address_reversed); // This log K and r_cycle was \log T isn't it?
    let mut E = vec![F::zero(); T];
    E.par_iter_mut().enumerate().for_each(|(y, e)| {
        *e = eq_tau[read_addresses[y]];
    });
    let mut ra_tau = MultilinearPolynomial::from(E);

    for _ in 0..T.log_2() {
        let univariate_poly_evals: [F; 2] = (0..ra_tau.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals = ra_tau.sumcheck_evals(index, 2, BindingOrder::LowToHigh);
                let val_evals = eq_r_cycle.sumcheck_evals(index, 2, BindingOrder::LowToHigh);
                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        // Construct coefficients of univariate polynomial from evaluations
        let univariate_poly = UniPoly::from_evals(&[
            val_claim * univariate_poly_evals[0],
            previous_claim - val_claim * univariate_poly_evals[0],
            val_claim * univariate_poly_evals[1],
        ]);
        // Skip the linear term when storing coeffs as we can always re-construct it
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        // Get challenge that binds the variable
        let r_j = transcript.challenge_scalar::<F>();
        r_address.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        rayon::join(
            || ra_tau.bind_parallel(r_j, BindingOrder::LowToHigh),
            || eq_r_cycle.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    // This is wrong : we need to multiply by Vals(Tau)
    let ra_tau_claim = ra_tau.final_sumcheck_claim();
    let eq_r_cycle_at_r_time = eq_r_cycle.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        sumcheck_claim, // These two are currently wrong
        ra_tau_claim,
        val_claim,
        eq_r_cycle_at_r_time,
    )
}

/// Implements the sumcheck prover for the core Shout PIOP when d = 1. See
/// Figure 5 from the Twist+Shout paper.
pub fn prove_core_shout_piop<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, F, F) {
    let K = lookup_table.len();
    let T = read_addresses.len();
    let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
    //let log_T = T.log_2();
    //let r_cycle = vec![F::from_u8(10); log_T];

    // Sumcheck for the core Shout PIOP (Figure 5)
    let num_rounds = K.log_2();
    let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

    let E: Vec<F> = EqPolynomial::evals(&r_cycle);
    let F: Vec<_> = (0..K)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| if *address == k { Some(E[cycle]) } else { None })
                .sum::<F>()
        })
        .collect();

    let sumcheck_claim: F = F
        .par_iter()
        .zip(lookup_table.par_iter())
        .map(|(&ra, &val)| ra * val)
        .sum();
    let mut previous_claim = sumcheck_claim;

    let mut ra = MultilinearPolynomial::from(F);
    let mut val = MultilinearPolynomial::from(lookup_table);

    const DEGREE: usize = 2;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);
        println!("Previous Claim: {previous_claim}");
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = ra.final_sumcheck_claim();
    println!("Final ra_claim: {ra_claim}");
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        sumcheck_claim,
        ra_claim,
    )
}

struct BooleanityProverState<F: JoltField> {
    read_addresses: Vec<usize>,
    K: usize,
    T: usize,
    B: GruenSplitEqPolynomial<F>,
    #[cfg(test)]
    old_B: MultilinearPolynomial<F>,
    F: Vec<F>,
    G: Vec<F>,
    D: MultilinearPolynomial<F>,
    /// Initialized after first log(K) rounds of sumcheck
    H: Option<MultilinearPolynomial<F>>,
}

impl<F: JoltField> BooleanityProverState<F> {
    #[tracing::instrument(skip_all)]
    fn initialize<ProofTranscript: Transcript>(
        read_addresses: Vec<usize>,
        D: Vec<F>,
        G: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = G.len();
        let r: Vec<F> = transcript.challenge_vector(K.log_2());
        let T = read_addresses.len();

        let D = MultilinearPolynomial::from(D);
        let B = GruenSplitEqPolynomial::new(&r); // (53)
        #[cfg(test)]
        let old_B = MultilinearPolynomial::from(EqPolynomial::evals(&r));
        let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
        F[0] = F::one();

        Self {
            read_addresses,
            K,
            T,
            B,
            #[cfg(test)]
            old_B,
            F,
            G,
            D,
            H: None,
        }
    }
}

struct BooleanityVerifierState<F: JoltField> {
    r_address: Vec<F>,
    r_cycle: Vec<F>,
}

impl<F: JoltField> BooleanityVerifierState<F> {
    fn initialize<ProofTranscript: Transcript>(
        r_cycle: &[F],
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let r_cycle: Vec<_> = r_cycle.iter().copied().rev().collect();
        let r_address: Vec<F> = transcript
            .challenge_vector(K.log_2())
            .into_iter()
            .rev()
            .collect();

        Self { r_cycle, r_address }
    }
}

struct BooleanitySumcheck<F: JoltField> {
    verifier_state: Option<BooleanityVerifierState<F>>,
    prover_state: Option<BooleanityProverState<F>>,
    ra_claim: Option<F>,
}

#[cfg(test)]
impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_prover_message_cubic(&self, round: usize) -> Vec<F> {
        const DEGREE: usize = 3;
        let BooleanityProverState {
            K,
            old_B: B,
            F,
            G,
            D,
            H,
            ..
        } = self.prover_state.as_ref().unwrap();

        // EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c: [[F; DEGREE]; 2] = [
            [
                F::one(),        // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
                F::from_i64(-1), // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
                F::from_i64(-2), // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
            ],
            [
                F::zero(),     // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
                F::from_u8(2), // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
                F::from_u8(3), // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
            ],
        ];
        // EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c_squared: [[F; DEGREE]; 2] = [
            [F::one(), F::one(), F::from_u8(4)],
            [F::zero(), F::from_u8(4), F::from_u8(9)],
        ];

        if round < K.log_2() {
            // First log(K) rounds of sumcheck
            let m = round + 1;

            let univariate_poly_evals: [F; DEGREE] = (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);
                    let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                        .par_iter()
                        .enumerate()
                        .map(|(k, &G_k)| {
                            // Since we're binding variables from low to high, k_m is the high bit
                            let k_m = k >> (m - 1);
                            // We then index into F using (k_{m-1}, ..., k_1)
                            let F_k = F[k % (1 << (m - 1))];
                            // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                            let G_times_F = G_k * F_k;
                            // For c \in {0, 2, 3} compute:
                            //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                            //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                            [
                                G_times_F * (eq_km_c_squared[k_m][0] * F_k - eq_km_c[k_m][0]),
                                G_times_F * (eq_km_c_squared[k_m][1] * F_k - eq_km_c[k_m][1]),
                                G_times_F * (eq_km_c_squared[k_m][2] * F_k - eq_km_c[k_m][2]),
                            ]
                        })
                        .reduce(
                            || [F::zero(); DEGREE],
                            |running, new| {
                                [
                                    running[0] + new[0],
                                    running[1] + new[1],
                                    running[2] + new[2],
                                ]
                            },
                        );

                    [
                        B_evals[0] * inner_sum[0],
                        B_evals[1] * inner_sum[1],
                        B_evals[2] * inner_sum[2],
                    ]
                })
                .reduce(
                    || [F::zero(); DEGREE],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                        ]
                    },
                );

            univariate_poly_evals.to_vec()
        } else {
            // Last log(T) rounds of sumcheck

            let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
                .into_par_iter()
                .map(|i| {
                    let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                    let H_evals =
                        H.as_ref()
                            .unwrap()
                            .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                    [
                        D_evals[0] * (H_evals[0] * H_evals[0] - H_evals[0]),
                        D_evals[1] * (H_evals[1] * H_evals[1] - H_evals[1]),
                        D_evals[2] * (H_evals[2] * H_evals[2] - H_evals[2]),
                    ]
                })
                .reduce(
                    || [F::zero(); 3],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                        ]
                    },
                );

            let eq_r_r = B.final_sumcheck_claim();
            univariate_poly_evals = [
                eq_r_r * univariate_poly_evals[0],
                eq_r_r * univariate_poly_evals[1],
                eq_r_r * univariate_poly_evals[2],
            ];

            univariate_poly_evals.to_vec()
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for BooleanitySumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            let BooleanityProverState { K, T, .. } = self.prover_state.as_ref().unwrap();
            K.log_2() + T.log_2()
        } else if self.verifier_state.is_some() {
            let BooleanityVerifierState { r_cycle, r_address } =
                self.verifier_state.as_ref().unwrap();
            r_address.len() + r_cycle.len()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        const DEGREE: usize = 3;
        let BooleanityProverState {
            K, B, F, G, D, H, ..
        } = self.prover_state.as_ref().unwrap();

        let evals = if round < K.log_2() {
            // First log(K) rounds of sumcheck
            let m = round + 1;

            // We use both Dao-Thaler and Gruen's optimizations here. See "Our optimization on top of
            // Gruen's" from Sec. 3 of https://eprint.iacr.org/2024/1210.pdf.
            //
            // We compute the evaluations of the cubic polynomial s(X) = l(X) * q(X) at {0, 2, 3} by
            // first computing the evaluations of the quadratic polynomial q(X) at 0 and infinity.
            // Moreover, we split the evaluations of the eq polynomial into two groups, E_in and E_out.
            // We use the GruenSplitEqPolynomial data structure to do this.
            //
            // Since E_in is bound first, we have two cases to handle: one where E_in is fully bound
            // and one where it is not.
            let quadratic_coeffs: [F; DEGREE - 1] = if B.E_in_current_len() == 1 {
                // Here E_in is fully bound, so we can ignore it and use the evaluations from E_out.
                (0..B.len() / 2)
                    .into_par_iter()
                    .map(|k_prime| {
                        let B_eval = B.E_out_current()[k_prime];
                        let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                            .par_iter()
                            .enumerate()
                            .map(|(k, &G_k)| {
                                // Since we're binding variables from low to high, k_m is the high bit
                                let k_m = k >> (m - 1);
                                // We then index into F using (k_{m-1}, ..., k_1)
                                let F_k = F[k % (1 << (m - 1))];
                                // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                                let G_times_F = G_k * F_k;
                                // For c \in {0, infty} compute:
                                //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                                //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                                //
                                // We want the following values, for k_m \in {0, 1}
                                //   - s(0) = G_times_F * (eq(k_m, 0)^2 * F_k - eq(k_m, 0))
                                //   - s(infty) = G_times_F * eq(k_m, infty)^2 * F_k
                                // But note that
                                //   - eq(0, 0)^2 = eq(0, 0) = 1
                                //   - eq(1, 0)^2 = eq(1, 0) = 0
                                //   - eq(0, infty)^2 = eq(1, infty)^2 = 1
                                // So we can instead compute
                                //   - s(0) = k_m == 0 ? G_times_F * (F_k - 1) : 0
                                //   - s(1) = G_times_F * F_k
                                let eval_0 = if k_m == 0 {
                                    G_times_F * (F_k - F::one())
                                } else {
                                    F::zero()
                                };
                                let eval_infty = G_times_F * F_k;
                                [eval_0, eval_infty]
                            })
                            .reduce(
                                || [F::zero(); DEGREE - 1],
                                |running, new| [running[0] + new[0], running[1] + new[1]],
                            );

                        [B_eval * inner_sum[0], B_eval * inner_sum[1]]
                    })
                    .reduce(
                        || [F::zero(); DEGREE - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    )
            } else {
                // Here E_in has not been fully bound, so the correct evaluation of eq is
                // E_in_eval * E_out_eval. We group the terms with the same value of E_out_eval in
                // order to decrease the total number of multiplications.
                let num_x_in_bits = B.E_in_current_len().log_2();
                let x_bitmask = (1 << num_x_in_bits) - 1;

                (0..B.len() / 2)
                    .collect::<Vec<_>>()
                    // Group values of k_prime where E_out_eval will have the same value
                    .par_chunk_by(|k1, k2| k1 >> num_x_in_bits == k2 >> num_x_in_bits)
                    .map(|chunk| {
                        let x_out = chunk[0] >> num_x_in_bits;
                        let B_E_out_eval = B.E_out_current()[x_out];

                        let chunk_evals = chunk
                            .par_iter()
                            .map(|k_prime| {
                                let x_in = k_prime & x_bitmask;
                                let B_E_in_eval = B.E_in_current()[x_in];

                                let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                                    .par_iter()
                                    .enumerate()
                                    .map(|(k, &G_k)| {
                                        // Since we're binding variables from low to high, k_m is the high bit
                                        let k_m = k >> (m - 1);
                                        // We then index into F using (k_{m-1}, ..., k_1)
                                        let F_k = F[k % (1 << (m - 1))];
                                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                                        let G_times_F = G_k * F_k;
                                        // For c \in {0, infty} compute:
                                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                                        //
                                        // We want the following values, for k_m \in {0, 1}
                                        //   - s(0) = G_times_F * (eq(k_m, 0)^2 * F_k - eq(k_m, 0))
                                        //   - s(infty) = G_times_F * eq(k_m, infty)^2 * F_k
                                        // But note that all of the above values of eq(., .) and
                                        // eq(., .)^2 are either 0 or 1. Namely:
                                        //   - eq(0, 0)^2 = eq(0, 0) = 1
                                        //   - eq(1, 0)^2 = eq(1, 0) = 0
                                        //   - eq(0, infty)^2 = eq(1, infty)^2 = 1
                                        // So we can instead compute
                                        //   - s(0) = k_m == 0 ? G_times_F * (F_k - 1) : 0
                                        //   - s(1) = G_times_F * F_k
                                        let eval_0 = if k_m == 0 {
                                            G_times_F * (F_k - F::one())
                                        } else {
                                            F::zero()
                                        };
                                        let eval_infty = G_times_F * F_k;
                                        [eval_0, eval_infty]
                                    })
                                    .reduce(
                                        || [F::zero(); DEGREE - 1],
                                        |running, new| [running[0] + new[0], running[1] + new[1]],
                                    );

                                [B_E_in_eval * inner_sum[0], B_E_in_eval * inner_sum[1]]
                            })
                            .reduce(
                                || [F::zero(); DEGREE - 1],
                                |running, new| [running[0] + new[0], running[1] + new[1]],
                            );

                        [B_E_out_eval * chunk_evals[0], B_E_out_eval * chunk_evals[1]]
                    })
                    .reduce(
                        || [F::zero(); DEGREE - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    )
            };

            B.sumcheck_evals_from_quadratic_coeffs(
                quadratic_coeffs[0],
                quadratic_coeffs[1],
                previous_claim,
            )
            .to_vec()
        } else {
            // Last log(T) rounds of sumcheck

            let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
                .into_par_iter()
                .map(|i| {
                    let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                    let H_evals =
                        H.as_ref()
                            .unwrap()
                            .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                    [
                        D_evals[0] * (H_evals[0] * H_evals[0] - H_evals[0]),
                        D_evals[1] * (H_evals[1] * H_evals[1] - H_evals[1]),
                        D_evals[2] * (H_evals[2] * H_evals[2] - H_evals[2]),
                    ]
                })
                .reduce(
                    || [F::zero(); 3],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                        ]
                    },
                );

            let eq_r_r = B.current_scalar;
            univariate_poly_evals = [
                eq_r_r * univariate_poly_evals[0],
                eq_r_r * univariate_poly_evals[1],
                eq_r_r * univariate_poly_evals[2],
            ];

            univariate_poly_evals.to_vec()
        };

        #[cfg(test)]
        {
            let test_evals = self.compute_prover_message_cubic(round);
            assert_eq!(evals, test_evals);
        }

        evals
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let BooleanityProverState {
            K,
            B,
            #[cfg(test)]
            old_B,
            F,
            D,
            H,
            read_addresses,
            ..
        } = self.prover_state.as_mut().unwrap();
        if round < K.log_2() {
            // First log(K) rounds of sumcheck
            B.bind(r_j);
            #[cfg(test)]
            old_B.bind_parallel(r_j, BindingOrder::LowToHigh);

            let inner_span = tracing::span!(tracing::Level::INFO, "Update F");
            let _inner_guard = inner_span.enter();

            // Update F for this round (see Equation 55)
            let (F_left, F_right) = F.split_at_mut(1 << round);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });

            if round == K.log_2() - 1 {
                // Transition point; initialize H
                *H = Some(MultilinearPolynomial::from(
                    read_addresses.par_iter().map(|&k| F[k]).collect::<Vec<_>>(),
                ));
            }
        } else {
            // Last log(T) rounds of sumcheck
            rayon::join(
                || D.bind_parallel(r_j, BindingOrder::LowToHigh),
                || {
                    H.as_mut()
                        .unwrap()
                        .bind_parallel(r_j, BindingOrder::LowToHigh)
                },
            );
        }
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.ra_claim.is_none());
        let BooleanityProverState { H, .. } = self.prover_state.as_ref().unwrap();
        let ra_claim = H.as_ref().unwrap().final_sumcheck_claim();
        self.ra_claim = Some(ra_claim);
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let BooleanityVerifierState { r_address, r_cycle } = self.verifier_state.as_ref().unwrap();
        let (r_address_prime, r_cycle_prime) = r.split_at(r_address.len());
        let ra_claim = self.ra_claim.unwrap();

        EqPolynomial::mle(r_address, r_address_prime)
            * EqPolynomial::mle(r_cycle, r_cycle_prime)
            * (ra_claim.square() - ra_claim)
    }
}

/// Implements the sumcheck prover for the Booleanity check in step 3 of
/// Figure 6 in the Twist+Shout paper. The efficient implementation of this
/// sumcheck is described in Section 6.3.
#[tracing::instrument(skip_all, name = "Shout booleanity sumcheck")]
pub fn prove_booleanity<F: JoltField, ProofTranscript: Transcript>(
    read_addresses: Vec<usize>,
    r: &[F],
    D: Vec<F>,
    G: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>, F) {
    const DEGREE: usize = 3;
    let K = r.len().pow2();
    let T = read_addresses.len();
    debug_assert_eq!(D.len(), T);
    debug_assert_eq!(G.len(), K);

    let mut B = MultilinearPolynomial::from(EqPolynomial::evals(r)); // (53)

    // First log(K) rounds of sumcheck

    let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
    F[0] = F::one();

    let num_rounds = K.log_2() + T.log_2();
    let mut r_address_prime: Vec<F> = Vec::with_capacity(K.log_2());
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let mut previous_claim = F::zero();

    // EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
    let eq_km_c: [[F; DEGREE]; 2] = [
        [
            F::one(),        // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
            F::from_i64(-1), // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
            F::from_i64(-2), // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
        ],
        [
            F::zero(),     // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
            F::from_u8(2), // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
            F::from_u8(3), // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
        ],
    ];
    // EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
    let eq_km_c_squared: [[F; DEGREE]; 2] = [
        [F::one(), F::one(), F::from_u8(4)],
        [F::zero(), F::from_u8(4), F::from_u8(9)],
    ];

    // First log(K) rounds of sumcheck
    let span = tracing::span!(
        tracing::Level::INFO,
        "First log(K) rounds of Booleanity sumcheck"
    );
    let _guard = span.enter();

    for round in 0..K.log_2() {
        let m = round + 1;

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 3] = (0..B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);
                let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, &G_k)| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let G_times_F = G_k * F_k;
                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F * (eq_km_c_squared[k_m][0] * F_k - eq_km_c[k_m][0]),
                            G_times_F * (eq_km_c_squared[k_m][1] * F_k - eq_km_c[k_m][1]),
                            G_times_F * (eq_km_c_squared[k_m][2] * F_k - eq_km_c[k_m][2]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );

                [
                    B_evals[0] * inner_sum[0],
                    B_evals[1] * inner_sum[1],
                    B_evals[2] * inner_sum[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
        ]);

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        B.bind_parallel(r_j, BindingOrder::LowToHigh);

        let inner_span = tracing::span!(tracing::Level::INFO, "Update F");
        let _inner_guard = inner_span.enter();

        // Update F for this round (see Equation 55)
        let (F_left, F_right) = F.split_at_mut(1 << round);
        F_left
            .par_iter_mut()
            .zip(F_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(
        tracing::Level::INFO,
        "Last log(T) rounds of Booleanity sumcheck"
    );
    let _guard = span.enter();

    let eq_r_r = B.final_sumcheck_claim();
    let H: Vec<F> = read_addresses.par_iter().map(|&k| F[k]).collect();
    let mut H = MultilinearPolynomial::from(H);
    let mut D = MultilinearPolynomial::from(D);
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(T.log_2());

    // Last log(T) rounds of sumcheck
    for _round in 0..T.log_2() {
        #[cfg(test)]
        {
            let expected: F = eq_r_r
                * (0..H.len())
                    .map(|j| {
                        let D_j = D.get_bound_coeff(j);
                        let H_j = H.get_bound_coeff(j);
                        D_j * (H_j.square() - H_j)
                    })
                    .sum::<F>();
            assert_eq!(
                expected, previous_claim,
                "Sumcheck sanity check failed in round {_round}"
            );
        }

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let H_evals = H.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    D_evals[0] * (H_evals[0] * H_evals[0] - H_evals[0]),
                    D_evals[1] * (H_evals[1] * H_evals[1] - H_evals[1]),
                    D_evals[2] * (H_evals[2] * H_evals[2] - H_evals[2]),
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals = [
            eq_r_r * univariate_poly_evals[0],
            eq_r_r * univariate_poly_evals[1],
            eq_r_r * univariate_poly_evals[2],
        ];

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
        ]);

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || D.bind_parallel(r_j, BindingOrder::LowToHigh),
            || H.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = H.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_prime,
        r_cycle_prime,
        ra_claim,
    )
}

/// Implements the sumcheck prover for the Hamming weight 1 check in step 5 of
/// Figure 6 in the Twist+Shout paper.
pub fn prove_hamming_weight<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    r_cycle_prime: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, F) {
    let K = lookup_table.len();
    let T = read_addresses.len();
    debug_assert_eq!(T.log_2(), r_cycle_prime.len());

    let num_rounds = K.log_2();
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);

    let E: Vec<F> = EqPolynomial::evals(&r_cycle_prime);
    let F: Vec<_> = (0..K)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| if *address == k { Some(E[cycle]) } else { None })
                .sum::<F>()
        })
        .collect();

    let mut ra = MultilinearPolynomial::from(F);
    let mut previous_claim = F::one();

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let univariate_poly_eval: F = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| ra.get_bound_coeff(2 * i))
            .sum();

        let univariate_poly =
            UniPoly::from_evals(&[univariate_poly_eval, previous_claim - univariate_poly_eval]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_double_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        ra.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    let ra_claim = ra.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_double_prime,
        ra_claim,
    )
}

/// Implements the sumcheck prover for the raf-evaluation sumcheck in step 6 of
/// Figure 6 in the Twist+Shout paper.
pub fn prove_raf_evaluation<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    r_cycle: Vec<F>,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, F) {
    let K = lookup_table.len();
    let T = read_addresses.len();
    debug_assert_eq!(T.log_2(), r_cycle.len());

    let E: Vec<F> = EqPolynomial::evals(&r_cycle);
    let F: Vec<_> = (0..K)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| if *address == k { Some(E[cycle]) } else { None })
                .sum::<F>()
        })
        .collect();

    let num_rounds = K.log_2();
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);

    let mut ra = MultilinearPolynomial::from(F);
    let mut int = IdentityPolynomial::new(num_rounds);

    let mut previous_claim = claimed_evaluation;

    const DEGREE: usize = 2;

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let int_evals: Vec<F> = int.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [ra_evals[0] * int_evals[0], ra_evals[1] * int_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_double_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || int.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = ra.final_sumcheck_claim();
    (SumcheckInstanceProof::new(compressed_polys), ra_claim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_ff::{One, Zero}; // often from ark_ff, depending on your setup
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_std::test_rng;
    use rand_core::RngCore;
    #[test]
    fn test_decompose_one_hot_matrix_general() {
        const K: usize = 4;
        const D: usize = 2;
        const N: usize = 2;
        const T: usize = 4; // 2**10
        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..K).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..T).map(|_| rng.next_u32() as usize % K).collect();

        //let lookup_table = [
        //    Fr::from_u8(3),
        //    Fr::from_u8(2),
        //    Fr::from_u8(1),
        //    Fr::from_u8(0),
        //]
        //.to_vec();
        //let read_addresses = vec![2, 3, 2, 3];
        //let K = 4;
        //let d = 2; // So N = 2 since 2^2 = 4
        //let N = 2;

        let ras = decompose_one_hot_matrix::<Fr>(&read_addresses, K, D);
        //===================d=1==========================================================================
        let ra: Vec<Vec<Fr>> = read_addresses
            .par_iter()
            .map(|&addr| {
                (0..K)
                    .into_par_iter()
                    .map(|j| if j == addr { Fr::one() } else { Fr::zero() })
                    .collect()
            })
            .collect();
        let flattened: Vec<Fr> = ra.iter().flat_map(|row| row.iter().cloned()).collect();
        let ra_poly = MultilinearPolynomial::from(flattened);
        //---------------------------------------------------------------------------------------------

        //==============================d > 1====================================
        let flattened_ras: Vec<Vec<Fr>> = (0..D)
            .into_par_iter()
            .map(|d| {
                ras[d]
                    .iter()
                    .flat_map(|row| row.iter().cloned())
                    .collect::<Vec<Fr>>()
            })
            .collect();

        let ra_polys: Vec<MultilinearPolynomial<Fr>> = flattened_ras
            .into_par_iter()
            .map(MultilinearPolynomial::from)
            .collect();
        //-------------------------------------------------

        // CHALLENGES
        let mut r_address: Vec<Fr> = [Fr::from_u8(2), Fr::from_u8(3)].to_vec();
        let r_cycle: Vec<Fr> = [Fr::from_u8(10), Fr::from_u8(10)].to_vec();
        let mut r_time: Vec<Fr> = [Fr::from_u8(4), Fr::from_u8(6)].to_vec();

        // Common Polynomials
        let eq_rcycle = MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle));
        let val = MultilinearPolynomial::from(lookup_table.clone());

        // FULL LOCATION FOR BIG ra
        let mut full_random_location = r_address.clone();
        full_random_location.extend(r_time.clone());
        full_random_location.reverse();
        let ra_evaluated_r_address_r_time = ra_poly.evaluate(&full_random_location);

        // CHUNKED FULL LOCATIONS FOR ras
        let chunk_size = N.log_2();
        let r_address_chunked: Vec<Vec<Fr>> = (0..D)
            .map(|i| {
                let start = i * chunk_size;
                let end = (i + 1) * chunk_size;
                r_address[start..end].to_vec()
            })
            .collect();
        // this is r_address_chunkded || r_time
        let full_chunked_locations: Vec<Vec<Fr>> = (0..D)
            .map(|i| {
                let mut combined = r_address_chunked[i].clone(); // clone the chunk
                combined.extend_from_slice(&r_time); // append r_time
                combined
            })
            .collect();

        // Evaluate each ra[j] j=0..d for chunked location
        let evaluations: Vec<Fr> = (0..D)
            .map(|i| {
                let mut random_location_rev = full_chunked_locations[i].clone();
                random_location_rev.reverse();
                ra_polys[i].evaluate(&random_location_rev) // no semicolon, return this value
            })
            .collect();

        r_time.reverse();
        let _eq_r_cycle_r_time = eq_rcycle.evaluate(&r_time);

        r_address.reverse();
        let _val_raddress = val.evaluate(&r_address);

        let mut acc_val = Fr::one();
        for (j, val) in evaluations.iter().enumerate() {
            for &loc in full_chunked_locations[j].iter() {
                print!("{loc}, ");
            }
            acc_val *= val;
        }
        // Mathematically the big vector == chunked vector
        // so this should really be the same
        assert_eq!(acc_val, ra_evaluated_r_address_r_time);
    }

    #[test]
    fn test_decompose_one_hot_matrix() {
        let lookup_table = [
            Fr::from_u8(3),
            Fr::from_u8(2),
            Fr::from_u8(1),
            Fr::from_u8(0),
        ]
        .to_vec();
        let read_addresses = vec![2, 3, 2, 3];
        let K = 4;
        let d = 2; // So N = 2 since 2^2 = 4
        let N = 2;
        let ras = decompose_one_hot_matrix::<Fr>(&read_addresses, K, d);

        // Expecting 2 matrices (since d = 2)
        assert_eq!(ras.len(), 2);
        // Each matrix should have T = 4 rows
        assert_eq!(ras[0].len(), 4);
        assert_eq!(ras[1].len(), 4);

        // Each row should have N = 2 entries
        assert_eq!(ras[0][0].len(), 2);
        assert_eq!(ras[1][0].len(), 2);

        // Base-4 decomposition:
        // 2  -> MSB [1, 0] LSB
        // 3 ->  MSB [1, 1] LSB

        // FIRST MATRIX FIRST ROW and Second Row
        // ra_0[0: ..]
        // ra_0[1: ..]
        assert_eq!(ras[0][0], vec![Fr::one(), Fr::zero()]);
        assert_eq!(ras[0][1], vec![Fr::zero(), Fr::one()]);

        // Second Matrix
        // ra_1[0: ..]
        // ra_1[1: ..]
        assert_eq!(ras[1][0], vec![Fr::zero(), Fr::one()]);
        assert_eq!(ras[1][1], vec![Fr::zero(), Fr::one()]);

        //===================d=1==========================================================================
        let ra: Vec<Vec<Fr>> = read_addresses
            .par_iter()
            .map(|&addr| {
                (0..K)
                    .into_par_iter()
                    .map(|j| if j == addr { Fr::one() } else { Fr::zero() })
                    .collect()
            })
            .collect();
        let flattened: Vec<Fr> = ra.iter().flat_map(|row| row.iter().cloned()).collect();
        let ra_poly = MultilinearPolynomial::from(flattened);
        //---------------------------------------------------------------------------------------------

        //==============================d > 1====================================
        let flattened_ras: Vec<Vec<Fr>> = (0..d)
            .into_par_iter()
            .map(|d| {
                ras[d]
                    .iter()
                    .flat_map(|row| row.iter().cloned())
                    .collect::<Vec<Fr>>()
            })
            .collect();

        let ra_polys: Vec<MultilinearPolynomial<Fr>> = flattened_ras
            .into_par_iter()
            .map(MultilinearPolynomial::from)
            .collect();
        //-------------------------------------------------

        // CHALLENGES
        let mut r_address: Vec<Fr> = [Fr::from_u8(2), Fr::from_u8(3)].to_vec();
        let r_cycle: Vec<Fr> = [Fr::from_u8(10), Fr::from_u8(10)].to_vec();
        let mut r_time: Vec<Fr> = [Fr::from_u8(4), Fr::from_u8(6)].to_vec();
        for (i, &time) in r_time.iter().enumerate() {
            println!("r_time[{i}] = {time}");
        }
        for (i, address) in r_address.iter().enumerate() {
            println!("r_adddress[{i}] = {address}");
        }

        // Common Polynomials
        let eq_rcycle = MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle));
        let val = MultilinearPolynomial::from(lookup_table.clone());

        // FULL LOCATION FOR BIG ra
        let mut full_random_location = r_address.clone();
        full_random_location.extend(r_time.clone());
        full_random_location.reverse();
        let ra_evaluated_r_address_r_time = ra_poly.evaluate(&full_random_location);

        // CHUNKED FULL LOCATIONS FOR ras
        let chunk_size = N.log_2();
        let r_address_chunked: Vec<Vec<Fr>> = (0..d)
            .map(|i| {
                let start = i * chunk_size;
                let end = (i + 1) * chunk_size;
                r_address[start..end].to_vec()
            })
            .collect();
        // this is r_address_chunkded || r_time
        let full_chunked_locations: Vec<Vec<Fr>> = (0..d)
            .map(|i| {
                let mut combined = r_address_chunked[i].clone(); // clone the chunk
                combined.extend_from_slice(&r_time); // append r_time
                combined
            })
            .collect();

        // Evaluate each ra[j] j=0..d for chunked location
        let evaluations: Vec<Fr> = (0..d)
            .map(|i| {
                let mut random_location_rev = full_chunked_locations[i].clone();
                random_location_rev.reverse();
                ra_polys[i].evaluate(&random_location_rev) // no semicolon, return this value
            })
            .collect();

        r_time.reverse();
        let _eq_r_cycle_r_time = eq_rcycle.evaluate(&r_time);
        //assert_eq!(eq_r_cycle_r_time, Fr::from_u16(67));

        r_address.reverse();
        let _val_raddress = val.evaluate(&r_address);

        let mut acc_val = Fr::one();
        for (j, val) in evaluations.iter().enumerate() {
            for &loc in full_chunked_locations[j].iter() {
                print!("{loc}, ");
            }
            println!();
            acc_val *= val;
            println!("Direct: ra_{j}[above]= {val}");
        }
        println!(
            "Final accumulated value: {acc_val}\nOne-shot value: {ra_evaluated_r_address_r_time}"
        );

        // Mathematically the big vector == chunked vector
        // so this should really be the same
        assert_eq!(acc_val, ra_evaluated_r_address_r_time);
        //assert_eq!(acc_val, Fr::from_u16(33));

        //let mut sumcheck_claim = Fr::zero();
        //for i in 0..8 {
        //    // Construct x as [MSB, mid, LSB] (big-endian)
        //    let x = [
        //        if (i >> 2) & 1 == 1 {
        //            Fr::one()
        //        } else {
        //            Fr::zero()
        //        }, // MSB = x[0]
        //        if (i >> 1) & 1 == 1 {
        //            Fr::one()
        //        } else {
        //            Fr::zero()
        //        }, // mid = x[1]
        //        if i & 1 == 1 { Fr::one() } else { Fr::zero() }, // LSB = x[2]
        //    ];
        //
        //    // x[1..] = LSB and mid (for val)
        //    // x[..1] = MSB only (for eq_rcycle)
        //    let x_tail = &x[1..]; // x[1], x[2] — 2 LSBs
        //    let x_head = &x[..1]; // x[0] — MSB only
        //
        //    let term = ra_poly.evaluate(&x) * val.evaluate(x_tail) * eq_rcycle.evaluate(x_head);
        //
        //    sumcheck_claim += term;
        //}
    }
    #[test]
    fn shout_e2e() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let proof = ShoutProof::prove(
            lookup_table.clone(),
            read_addresses,
            &r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let verification_result = proof.verify(lookup_table, &r_cycle, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn core_shout_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let table_size = lookup_table.len(); // T
        let num_lookups = read_addresses.len(); // K
        let ra: Vec<Vec<Fr>> = read_addresses // T x K one hot encoded matrix
            .par_iter()
            .map(|&addr| {
                (0..table_size)
                    .into_par_iter()
                    .map(|j| if j == addr { Fr::one() } else { Fr::zero() })
                    .collect()
            })
            .collect();
        let flattened: Vec<Fr> = ra
            .par_iter()
            .flat_map_iter(|row| row.iter().cloned())
            .collect();
        // THESE ARE THE THINGS THE PROVER COMMITS TO
        // Technically val is not always committed to as the verifier can compute it using
        // log(K) multiplications where K=|lookuptable|
        let ra_poly = MultilinearPolynomial::from(flattened);
        let val = MultilinearPolynomial::from(lookup_table.clone());

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let (sumcheck_proof, _vfr_challenges, sumcheck_claim, sum_check_oracle_eval) =
            prove_core_shout_piop(lookup_table, read_addresses, &mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);

        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(num_lookups.log_2());
        let verification_result = sumcheck_proof.verify(
            sumcheck_claim,
            table_size.log_2(),
            2,
            &mut verifier_transcript,
        );

        let (final_claim, verifier_challenges) = verification_result.unwrap();
        let mut r_address = verifier_challenges.to_vec();
        r_address.reverse();
        let val_at_r_address = val.evaluate(&r_address);

        // The problem is that when we call EqPoly::eval -- the input is big endian.
        // so r_cycle = [r_{\log T}, ..., r_1] is actually stored in reverse order.
        let mut r_cycle_copy = r_cycle.clone();
        r_cycle_copy.reverse();

        // But the sumcheck challenges is low to high so verifiers challenges is little-endian.
        let mut full_random_location = verifier_challenges.clone();
        full_random_location.extend_from_slice(&r_cycle_copy);

        full_random_location.reverse();
        let ra_evaluated_r_address_r_cycle = ra_poly.evaluate(&full_random_location);

        assert_eq!(ra_evaluated_r_address_r_cycle, sum_check_oracle_eval);
        // Check if the sumcheck oracle check matches
        assert_eq!(
            ra_evaluated_r_address_r_cycle * val_at_r_address,
            final_claim
        );
    }
    fn decompose_one_hot_matrix<F: JoltField>(
        read_addresses: &[usize],
        K: usize,
        d: usize,
    ) -> Vec<Vec<Vec<F>>> {
        let T = read_addresses.len();
        let N = (K as f64).powf(1.0 / d as f64).round() as usize;
        assert_eq!(N.pow(d as u32), K, "K must be a perfect power of N");

        // Step 1: compute base-N digits for each address
        let digits_per_addr: Vec<Vec<usize>> = read_addresses
            .par_iter()
            .map(|&addr| {
                let mut digits = vec![0; d];
                let mut rem = addr;
                for j in (0..d).rev() {
                    digits[j] = rem % N;
                    rem /= N;
                }
                digits
            })
            .collect();

        // Step 2: build d matrices of shape T x N
        let mut result = vec![vec![vec![F::zero(); N]; T]; d];

        for (i, digits) in digits_per_addr.iter().enumerate() {
            for (j, &digit) in digits.iter().enumerate() {
                result[d - j - 1][i][digit] = F::one();
            }
        }

        result
    }

    #[test]
    fn test_core_generic_d_greater_than_one_shout_sumcheck() {
        //------- PROBLEM SETUP----------------------
        const K: usize = 64; // 2**6
        const T: usize = 1 << 10; // 2**10
        const D: usize = 2;
        const N: usize = 2;

        //let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        //let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
        //    .map(|_| rng.next_u32() as usize % TABLE_SIZE)
        //    .collect();
        let seed1: u64 = 42;
        let mut rng1 = StdRng::seed_from_u64(seed1);
        let lookup_table: Vec<Fr> = (0..K).map(|_| Fr::rand(&mut rng1)).collect();
        let read_addresses: Vec<usize> = (0..T).map(|_| (rng1.next_u32() as usize) % K).collect();

        //-------------------------------------------

        //const K: usize = 4; // 2**6
        //const T: usize = 2; // 2**10
        //let read_addresses = vec![2, 3];
        //let lookup_table = [
        //    Fr::from_u8(3),
        //    Fr::from_u8(2),
        //    Fr::from_u8(1),
        //    Fr::from_u8(2),
        //]
        //.to_vec();
        //let D = 2; // So N = 2 since 2^2 = 4
        //let N = 2;
        //
        let ras: Vec<Vec<Vec<Fr>>> = decompose_one_hot_matrix(&read_addresses, K, D);
        let flattened_ras: Vec<Vec<Fr>> = (0..D)
            .into_par_iter()
            .map(|d| {
                ras[d]
                    .iter()
                    .flat_map(|row| row.iter().cloned())
                    .collect::<Vec<Fr>>()
            })
            .collect();

        let ra_polys: Vec<MultilinearPolynomial<Fr>> = flattened_ras
            .into_par_iter()
            .map(MultilinearPolynomial::from)
            .collect();
        let val = MultilinearPolynomial::from(lookup_table.clone());
        //-------------------------------------------------------------------------------
        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let (
            sumcheck_proof,
            verifier_challenges,
            sumcheck_claim,
            ra_address_time_claim,
            val_tau_claim,
            eq_rcycle_rtime_claim,
        ) = prove_generic_core_shout_pip_d_greater_than_one(
            lookup_table,
            read_addresses,
            D,
            &mut prover_transcript,
        );

        for (i, &challenge) in verifier_challenges.iter().enumerate() {
            println!("Round {}, {}", i + 1, challenge);
        }
        println!("SUMCHECK CLAIM: {sumcheck_claim}");
        let product = ra_address_time_claim * val_tau_claim * eq_rcycle_rtime_claim;
        println!(
    "ra_address_time_claim = {ra_address_time_claim}, \nval_tau_claim = {val_tau_claim}, \neq_rcycle_rtime_claim = {eq_rcycle_rtime_claim}, \nproduct = {product}",
);

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);

        // Already in Big ENDIAN
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
        let verification_result = sumcheck_proof.verify(
            sumcheck_claim,
            K.log_2() + T.log_2(),
            D + 2,
            &mut verifier_transcript,
        );
        let (final_claim, _vfr_challenges) = verification_result.unwrap();
        //-------------------------------------------------------------------------

        let (r_address, r_time) = verifier_challenges.split_at(K.log_2());
        let mut r_address_rev = r_address.to_vec();
        r_address_rev.reverse();
        let val_at_r_address = val.evaluate(&r_address_rev);

        // Now i need to take r_address and split it into D chunks
        let chunk_size = N.log_2();
        let r_address_chunked: Vec<Vec<Fr>> = (0..D)
            .map(|i| {
                let start = i * chunk_size;
                let end = (i + 1) * chunk_size;
                r_address[start..end].to_vec()
            })
            .collect();

        // this is r_address_chunkded || r_time
        let full_random_locations: Vec<Vec<Fr>> = (0..D)
            .map(|i| {
                let mut combined = r_address_chunked[i].clone(); // clone the chunk
                combined.extend_from_slice(r_time); // append r_time
                combined
            })
            .collect();

        let evaluations: Vec<Fr> = (0..D)
            .map(|i| {
                let mut random_location_rev = full_random_locations[i].clone();
                random_location_rev.reverse();
                ra_polys[i].evaluate(&random_location_rev) // no semicolon, return this value
            })
            .collect();

        println!("Direct Val@r_address {val_at_r_address}");
        println!("Direct Eq@r_cycle||r_time: {eq_rcycle_rtime_claim}");

        let evaluations_product: Fr = evaluations.iter().fold(Fr::one(), |acc, val| acc * val);
        assert_eq!(evaluations_product, ra_address_time_claim);
        assert_eq!(val_at_r_address, val_tau_claim);
        let mut r_time_rev: Vec<Fr> = r_time.to_vec();
        r_time_rev.reverse();
        let eq_r_cycle_at_r_time =
            MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle)).evaluate(&r_time_rev);
        assert_eq!(eq_r_cycle_at_r_time, eq_rcycle_rtime_claim);
        // SOULD BE ANOTHER CHECK HERE

        // I Should calculate the last one and not use the claim
        let final_oracle_answer = val_at_r_address * evaluations_product * eq_r_cycle_at_r_time;
        assert_eq!(final_oracle_answer, final_claim);
    }

    #[test]
    fn test_core_generic_d_is_one_shout_sumcheck() {
        //------- PROBLEM SETUP----------------------
        const TABLE_SIZE: usize = 64; // 2**6
        const NUM_LOOKUPS: usize = 1 << 10; // 2**10

        //let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        //let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
        //    .map(|_| rng.next_u32() as usize % TABLE_SIZE)
        //    .collect();
        let seed1: u64 = 42;
        let mut rng1 = StdRng::seed_from_u64(seed1);
        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::rand(&mut rng1)).collect();

        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| (rng1.next_u32() as usize) % TABLE_SIZE)
            .collect();

        //-------------------------------------------

        let table_size = lookup_table.len();
        let num_lookups = read_addresses.len();
        let ra: Vec<Vec<Fr>> = read_addresses
            .par_iter()
            .map(|&addr| {
                (0..table_size)
                    .into_par_iter()
                    .map(|j| if j == addr { Fr::one() } else { Fr::zero() })
                    .collect()
            })
            .collect();

        let flattened: Vec<Fr> = ra.iter().flat_map(|row| row.iter().cloned()).collect();

        // What the prover commits to
        let ra_poly = MultilinearPolynomial::from(flattened);
        let val = MultilinearPolynomial::from(lookup_table.clone());

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let (
            sumcheck_proof,
            verifier_challenges,
            sumcheck_claim,
            ra_address_time_claim,
            val_tau_claim,
            eq_rcycle_rtime_claim,
        ) = prove_generic_core_shout_pip(lookup_table, read_addresses, 1, &mut prover_transcript);

        for (i, &challenge) in verifier_challenges.iter().enumerate() {
            println!("Round {}, {}", i + 1, challenge);
        }
        println!("SUMCHECK CLAIM: {sumcheck_claim}");

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);

        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(num_lookups.log_2());
        let verification_result = sumcheck_proof.verify(
            sumcheck_claim,
            table_size.log_2() + num_lookups.log_2(),
            2,
            &mut verifier_transcript,
        );
        let (final_claim, _verifier_challenges) = verification_result.unwrap();
        let (r_address, r_time) = verifier_challenges.split_at(table_size.log_2());

        let mut r_address = r_address.to_vec();
        r_address.reverse();
        let val_at_r_address = val.evaluate(&r_address);

        let mut full_random_location = verifier_challenges.clone();
        full_random_location.reverse();
        let ra_evaluated_r_address_r_time = ra_poly.evaluate(&full_random_location);
        let eq_r_cycle = MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle));
        let mut r_time = r_time.to_vec();
        r_time.reverse();
        let eq_r_cycle_r_time = eq_r_cycle.evaluate(&r_time);

        // These are the 3 product terms evaluated at the final veerifiers
        // challenges
        assert_eq!(ra_evaluated_r_address_r_time, ra_address_time_claim);
        assert_eq!(val_at_r_address, val_tau_claim);
        assert_eq!(eq_r_cycle_r_time, eq_rcycle_rtime_claim);

        // THis is the final opening check
        assert_eq!(
            final_claim,
            ra_evaluated_r_address_r_time * eq_r_cycle_r_time * val_at_r_address
        );
        assert_eq!(1, 0);
    }

    #[test]
    fn booleanity_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r: Vec<Fr> = prover_transcript.challenge_vector(TABLE_SIZE.log_2());
        let r_prime: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let E: Vec<Fr> = EqPolynomial::evals(&r_prime);
        let F: Vec<_> = (0..TABLE_SIZE)
            .into_par_iter()
            .map(|k| {
                read_addresses
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(cycle, address)| if *address == k { Some(E[cycle]) } else { None },
                    )
                    .sum::<Fr>()
            })
            .collect();

        let (sumcheck_proof, _, _, _) =
            prove_booleanity(read_addresses, &r, E, F, &mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _: Vec<Fr> = verifier_transcript.challenge_vector(TABLE_SIZE.log_2());
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());

        let verification_result = sumcheck_proof.verify(
            Fr::zero(),
            TABLE_SIZE.log_2() + NUM_LOOKUPS.log_2(),
            3,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn hamming_weight_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle_prime: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let (sumcheck_proof, _, _) = prove_hamming_weight(
            lookup_table,
            read_addresses,
            r_cycle_prime,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());

        let verification_result =
            sumcheck_proof.verify(Fr::one(), TABLE_SIZE.log_2(), 1, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn raf_evaluation_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();
        let raf = MultilinearPolynomial::from(
            read_addresses.iter().map(|a| *a as u32).collect::<Vec<_>>(),
        );

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let raf_eval = raf.evaluate(&r_cycle);
        let (sumcheck_proof, _) = prove_raf_evaluation(
            lookup_table,
            read_addresses,
            r_cycle,
            raf_eval,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());

        let verification_result =
            sumcheck_proof.verify(raf_eval, TABLE_SIZE.log_2(), 2, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
