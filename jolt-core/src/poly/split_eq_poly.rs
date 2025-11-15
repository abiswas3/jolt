//! Implements the Dao-Thaler + Gruen optimization for EQ polynomial evaluations
//! https://eprint.iacr.org/2024/1210.pdf

use allocative::Allocative;
use rayon::prelude::*;

use super::dense_mlpoly::DensePolynomial;
use super::multilinear_polynomial::BindingOrder;
use crate::{field::JoltField, poly::eq_poly::EqPolynomial, utils::math::Math};

#[derive(Debug, Clone, PartialEq, Allocative)]
/// A struct holding the equality polynomial evaluations for use in sum-check, when incorporating
/// both the Gruen and Dao-Thaler optimizations.
///
/// For the `i = 0..n`-th round of sum-check, we want the following invariants (low to high):
///
/// - `current_index = n - i` (where `n = w.len()`)
/// - `current_scalar = eq(w[(n - i)..],r[..i])`
/// - `E_out_vec.last().unwrap() = [eq(w[..min(i, n/2)], x) for all x in {0, 1}^{n - min(i, n/2)}]`
/// - If `i < n/2`, then `E_in_vec.last().unwrap() = [eq(w[n/2..(n/2 + i + 1)], x) for all x in {0,
///   1}^{n/2 - i - 1}]`; else `E_in_vec` is empty
///
/// Implements both LowToHigh ordering and HighToLow ordering.
pub struct GruenSplitEqPolynomial<F: JoltField> {
    pub(crate) current_index: usize,
    pub(crate) current_scalar: F,
    pub(crate) w: Vec<F::Challenge>,
    pub(crate) E_in_vec: Vec<Vec<F>>,
    pub(crate) E_out_vec: Vec<Vec<F>>,
    pub(crate) binding_order: BindingOrder,
}

impl<F: JoltField> GruenSplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::new_with_scaling")]
    pub fn new_with_scaling(
        w: &[F::Challenge],
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
    ) -> Self {
        match binding_order {
            BindingOrder::LowToHigh => {
                let m = w.len() / 2;
                //   w = [w_out, w_in, w_last]
                //         ↑      ↑      ↑
                //         |      |      |
                //         |      |      last element
                //         |      second half of remaining elements (for E_in)
                //         first half of remaining elements (for E_out)
                let (_, wprime) = w.split_last().unwrap();
                let (w_out, w_in) = wprime.split_at(m);
                let (E_out_vec, E_in_vec) = rayon::join(
                    || EqPolynomial::evals_cached(w_out),
                    || EqPolynomial::evals_cached(w_in),
                );
                Self {
                    current_index: w.len(),
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    binding_order,
                }
            }
            BindingOrder::HighToLow => {
                // For high-to-low binding, we bind from MSB (index 0) to LSB (index n-1).
                // The split should be: w_in = first half, w_out = second half
                // [w_first, w_in, w_out]
                let (_, wprime) = w.split_first().unwrap();
                let m = w.len() / 2;
                let (w_in, w_out) = wprime.split_at(m);
                let (E_in_vec, E_out_vec) = rayon::join(
                    || EqPolynomial::evals_cached_rev(w_in),
                    || EqPolynomial::evals_cached_rev(w_out),
                );

                Self {
                    current_index: 0, // Start from 0 for high-to-low up to w.len() - 1
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    binding_order,
                }
            }
        }
    }

    pub fn new(w: &[F::Challenge], binding_order: BindingOrder) -> Self {
        Self::new_with_scaling(w, binding_order, None)
    }

    pub fn get_num_vars(&self) -> usize {
        self.w.len()
    }

    pub fn len(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => 1 << self.current_index,
            BindingOrder::HighToLow => 1 << (self.w.len() - self.current_index),
        }
    }

    pub fn E_in_current_len(&self) -> usize {
        self.E_in_vec.last().map_or(0, |v| v.len())
    }

    pub fn E_out_current_len(&self) -> usize {
        self.E_out_vec.last().map_or(0, |v| v.len())
    }

    /// Return the last vector from `E1` as a slice
    pub fn E_in_current(&self) -> &[F] {
        self.E_in_vec.last().map_or(&[], |v| v.as_slice())
    }

    /// Return the last vector from `E2` as a slice
    pub fn E_out_current(&self) -> &[F] {
        self.E_out_vec.last().map_or(&[], |v| v.as_slice())
    }

    /// Return the (E_out, E_in) tables corresponding to a streaming window of the
    /// given `window_size`, expressed purely in terms of the cached tables that
    /// were precomputed in `new_with_scaling`.
    ///
    /// The intent is that:
    /// - for `window_size = 1`, this reduces to the existing split-eq behaviour,
    ///   i.e. `E_in_window = E_in_current` and `E_out_window = E_out_current`;
    /// - for larger windows, we "shift" which cached layers we regard as
    ///   `E_in` and `E_out` without recomputing any equality tables. Intuitively
    ///   we pull additional unbound variables into the `E_in` side until the
    ///   window is large enough. This is used only for the streaming grid
    ///   construction in the Spartan outer sumcheck.
    ///
    /// At the moment this is only defined for `BindingOrder::LowToHigh`, which
    /// is the ordering used in the outer Spartan sumcheck. For `HighToLow` it
    /// simply falls back to the current (unshifted) tables.
    ///
    /// This helper returns owned vectors rather than borrowing from the cached
    /// tables so that it can correctly represent the "no bits" case as a
    /// single-entry `[1]` table. This matches the semantics of the equality
    /// polynomial over zero variables (`eq((), ()) = 1`) and avoids having an
    /// empty domain, which would incorrectly zero out streaming grids.
    pub fn E_out_in_for_window(&self, window_size: usize) -> (Vec<F>, Vec<F>) {
        if window_size == 0 {
            return (vec![F::one()], vec![F::one()]);
        }

        match self.binding_order {
            BindingOrder::LowToHigh => {
                // In the LowToHigh implementation we maintain two stacks
                //   - `E_out_vec[j]` is the eq-table for the first `j` bits of `w_out`
                //   - `E_in_vec[j]`  is the eq-table for the first `j` bits of `w_in`
                //
                // `bind` pops from `E_in_vec` first (while there are still "in" bits
                // left), then from `E_out_vec`. Intuitively, going backwards in
                // these stacks exposes more unbound variables on the "in" side.
                //
                // For a window of size 1 we want to preserve the current behaviour:
                // `E_in_window = E_in_current`, `E_out_window = E_out_current`.
                if window_size == 1 {
                    return (self.E_out_current().to_vec(), self.E_in_current().to_vec());
                }

                let extra_bits = window_size - 1;

                // How many bits are currently encoded by the top layer of E_in/E_out?
                // By construction, `E_*_vec.len() = num_bits + 1` and the last entry
                // has length `2^{num_bits}`.
                let mut in_bits = if self.E_in_vec.is_empty() {
                    0
                } else {
                    self.E_in_vec.len() - 1
                };
                let mut out_bits = if self.E_out_vec.is_empty() {
                    0
                } else {
                    self.E_out_vec.len() - 1
                };

                // We conceptually "pull" extra_bits variables from the outer side
                // into the inner side. As long as we still have bits represented
                // in E_out_vec we reduce `out_bits` and increase `in_bits`. Once
                // we run out of E_out bits we stop; at that point any further
                // increase in window size would need information that is not
                // represented in the cached tables.
                let mut remaining = extra_bits;
                while remaining > 0 && out_bits > 0 {
                    out_bits -= 1;
                    in_bits += 1;
                    remaining -= 1;
                }

                // Clamp to available ranges to avoid panics in edge cases where
                // the caller asks for a window that is larger than the number of
                // unbound variables that the cached tables can represent.
                if out_bits > self.E_out_vec.len().saturating_sub(1) {
                    out_bits = self.E_out_vec.len().saturating_sub(1);
                }
                if in_bits > self.E_in_vec.len().saturating_sub(1) {
                    in_bits = self.E_in_vec.len().saturating_sub(1);
                }

                let e_out = if self.E_out_vec.is_empty() {
                    vec![F::one()]
                } else {
                    self.E_out_vec[out_bits].clone()
                };
                let e_in = if self.E_in_vec.is_empty() {
                    vec![F::one()]
                } else {
                    self.E_in_vec[in_bits].clone()
                };

                (e_out, e_in)
            }
            BindingOrder::HighToLow => {
                // Not used in the streaming Spartan prover; fall back to the
                // current (unshifted) tables for now.
                (self.E_out_current().to_vec(), self.E_in_current().to_vec())
            }
        }
    }

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        match self.binding_order {
            BindingOrder::LowToHigh => {
                // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
                // which is the same as `1 - w[i] - r + 2 * w[i] * r`
                let prod_w_r = self.w[self.current_index - 1] * r;
                self.current_scalar *=
                    F::one() - self.w[self.current_index - 1] - r + prod_w_r + prod_w_r;
                // decrement `current_index`
                self.current_index -= 1;
                // pop the last vector from `E_in_vec` or `E_out_vec` (since we don't need it anymore)
                if self.w.len() / 2 < self.current_index {
                    self.E_in_vec.pop();
                } else if 0 < self.current_index {
                    self.E_out_vec.pop();
                }
            }
            BindingOrder::HighToLow => {
                // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
                // which is the same as `1 - w[i] - r + 2 * w[i] * r`
                let prod_w_r = self.w[self.current_index] * r;
                self.current_scalar *=
                    F::one() - self.w[self.current_index] - r + prod_w_r + prod_w_r;

                // increment `current_index` (going from 0 to n-1)
                self.current_index += 1;

                // pop the last vector from `E_in_vec` or `E_out_vec` (since we don't need it anymore)
                // For high-to-low, we bind variables in the first half first (E_in),
                // then variables in the second half (E_out)
                if self.current_index <= self.w.len() / 2 {
                    // We're binding variables from the first half (E_in)
                    self.E_in_vec.pop();
                } else if self.current_index <= self.w.len() {
                    // We're binding variables from the second half (E_out)
                    self.E_out_vec.pop();
                }
            }
        }
    }

    /// Compute the cubic sumcheck evaluations (i.e., the evaluations at {0, 2, 3}) of a
    /// polynomial s(X) = l(X) * q(X), where l(X) is the current (linear) eq polynomial and
    /// q(X) = c + dX + eX^2, given the following:
    /// - c, the constant term of q
    /// - e, the quadratic term of q
    /// - the previous round claim, s(0) + s(1)
    pub fn gruen_evals_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
    ) -> [F; 3] {
        // We want to compute the evaluations of the cubic polynomial s(X) = l(X) * q(X), where
        // l is linear, and q is quadratic, at the points {0, 2, 3}.
        //
        // At this point, we have
        // - the linear polynomial, l(X) = a + bX
        // - the quadratic polynomial, q(X) = c + dX + eX^2
        // - the previous round's claim s(0) + s(1) = a * c + (a + b) * (c + d + e)
        //
        // Both l and q are represented by their evaluations at 0 and infinity. I.e., we have a, b, c,
        // and e, but not d. We compute s by first computing l and q at points 2 and 3.

        // Evaluations of the linear polynomial
        let eq_eval_1 = self.current_scalar
            * match self.binding_order {
                BindingOrder::LowToHigh => self.w[self.current_index - 1],
                BindingOrder::HighToLow => self.w[self.current_index],
            };
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;
        let eq_eval_3 = eq_eval_2 + eq_m;

        // Evaluations of the quadratic polynomial
        let quadratic_eval_0 = q_constant;
        let cubic_eval_0 = eq_eval_0 * quadratic_eval_0;
        let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
        // q(1) = c + d + e
        let quadratic_eval_1 = cubic_eval_1 / eq_eval_1;
        // q(2) = c + 2d + 4e = q(1) + q(1) - q(0) + 2e
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - quadratic_eval_0 + e_times_2;
        // q(3) = c + 3d + 9e = q(2) + q(1) - q(0) + 4e
        let quadratic_eval_3 =
            quadratic_eval_2 + quadratic_eval_1 - quadratic_eval_0 + e_times_2 + e_times_2;

        [
            cubic_eval_0,
            eq_eval_2 * quadratic_eval_2,
            eq_eval_3 * quadratic_eval_3,
        ]
    }

    /// Compute the quadratic sumcheck evaluations (i.e., the evaluations at {0, 2}) of a
    /// polynomial s(X) = l(X) * q(X), where l(X) is the current (linear) Dao-Thaler eq polynomial and
    /// q(X) = c + dx
    /// - c, the constant term of q
    /// - the previous round claim, s(0) + s(1)
    pub fn gruen_evals_deg_2(&self, q_0: F, previous_claim: F) -> [F; 2] {
        // We want to compute the evaluations of the quadratic polynomial s(X) = l(X) * q(X), where
        // l is linear, and q is linear, at the points {0, 2}.
        //
        // At this point, we have:
        // - the linear polynomial, l(X) = a + bX
        // - the linear polynomial, q(X) = c + dX
        // - the previous round's claim s(0) + s(1) = a * c + (a + b) * (c + d)
        //
        // We have q(0) = c, and we need to compute q(1) and q(2).

        // Evaluations of the linear eq polynomial
        let eq_eval_1 = self.current_scalar
            * match self.binding_order {
                BindingOrder::LowToHigh => self.w[self.current_index - 1],
                BindingOrder::HighToLow => self.w[self.current_index],
            };
        let eq_eval_0 = self.current_scalar - eq_eval_1;

        // slope for eq
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;

        // Evaluations of the linear q(x) polynomial
        let linear_eval_0 = q_0;
        let quadratic_eval_0 = eq_eval_0 * linear_eval_0;
        let quadratic_eval_1 = previous_claim - quadratic_eval_0;

        // q(1) = c + d
        let linear_eval_1 = quadratic_eval_1 / eq_eval_1;

        // q(2) = c + 2d = 2*q(1) - q(0)
        let linear_eval_2 = linear_eval_1 + linear_eval_1 - linear_eval_0;

        [quadratic_eval_0, eq_eval_2 * linear_eval_2]
    }

    pub fn merge(&self) -> DensePolynomial<F> {
        let evals = match self.binding_order {
            BindingOrder::LowToHigh => {
                // For low-to-high, current_index tracks how many variables remain unbound
                // We want eq(w[0..current_index], x)
                EqPolynomial::evals_parallel(
                    &self.w[..self.current_index],
                    Some(self.current_scalar),
                )
            }
            BindingOrder::HighToLow => {
                // For high-to-low, current_index tracks how many variables have been bound
                // We want eq(w[current_index..], x)
                EqPolynomial::evals_parallel(
                    &self.w[self.current_index..],
                    Some(self.current_scalar),
                )
            }
        };
        DensePolynomial::new(evals)
    }

    pub fn get_current_scalar(&self) -> F {
        self.current_scalar
    }

    pub fn get_current_w(&self) -> F::Challenge {
        match self.binding_order {
            BindingOrder::LowToHigh => self.w[self.current_index - 1],
            BindingOrder::HighToLow => self.w[self.current_index],
        }
    }

    /// Emulates the behavior of EqPolynomial::evals(&self.w).par_iter().enumerate()
    /// Only works if `self.binding_order` is `BindingOrder::LowToHigh`.
    /// For the high-to-low version, see `par_iter_high_to_low`.
    pub fn par_iter_low_to_high(&self) -> impl ParallelIterator<Item = (usize, F)> + use<'_, F> {
        assert_eq!(self.binding_order, BindingOrder::LowToHigh);
        assert_eq!(
            self.current_index,
            self.w.len(),
            "par_iter_low_to_high only supports unbound polynomials"
        );

        let E_in = self.E_in_current();
        let x_in_bits = E_in.len().log_2();
        let E_out = self.E_out_current();
        let w_current = self.get_current_w();
        E_out.par_iter().enumerate().flat_map(move |(x_out, high)| {
            E_in.par_iter().enumerate().flat_map(move |(x_in, low)| {
                let high_low = *high * low;
                let eval_1 = high_low * w_current;
                let eval_0 = high_low - eval_1;
                let index = (x_out << (x_in_bits + 1)) + (x_in << 1);
                [(index, eval_0), (index + 1, eval_1)]
            })
        })
    }

    /// Emulates the behavior of EqPolynomial::evals(&self.w).par_iter().enumerate()
    /// Only works if `self.binding_order` is `BindingOrder::HighToLow`.
    /// For the low-to-high version, see `par_iter_low_to_high`.
    pub fn par_iter_high_to_low(&self) -> impl ParallelIterator<Item = (usize, F)> + use<'_, F> {
        assert_eq!(self.binding_order, BindingOrder::HighToLow);
        assert_eq!(
            self.current_index, 0,
            "par_iter_high_to_low only supports unbound polynomials"
        );

        let E_in = self.E_in_current();
        let x_in_bits = E_in.len().log_2();
        let E_out = self.E_out_current();
        let x_out_bits = E_out.len().log_2();
        let w_current = self.get_current_w();
        [F::one() - w_current, w_current.into()]
            .into_par_iter()
            .enumerate()
            .flat_map(move |(msb, eq_msb)| {
                E_in.par_iter().enumerate().flat_map(move |(x_in, high)| {
                    E_out.par_iter().enumerate().map(move |(x_out, low)| {
                        let index =
                            (msb << (x_in_bits + x_out_bits)) + (x_in << x_out_bits) + x_out;
                        (index, eq_msb * high * low)
                    })
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::math::Math;
    use ark_bn254::Fr;
    use ark_std::test_rng;

    #[test]
    fn bind_low_high() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut regular_eq = DensePolynomial::<Fr>::new(EqPolynomial::evals(&w));
        let mut split_eq = GruenSplitEqPolynomial::new(&w, BindingOrder::LowToHigh);
        assert_eq!(regular_eq, split_eq.merge());

        for _ in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            regular_eq.bound_poly_var_bot(&r);
            split_eq.bind(r);

            let merged = split_eq.merge();
            assert_eq!(regular_eq.Z[..regular_eq.len()], merged.Z[..merged.len()]);
        }
    }

    #[test]
    fn bind_high_low() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut regular_eq = DensePolynomial::<Fr>::new(EqPolynomial::evals(&w));
        let mut split_eq_high_to_low = GruenSplitEqPolynomial::new(&w, BindingOrder::HighToLow);

        // Verify they start equal
        assert_eq!(regular_eq, split_eq_high_to_low.merge());

        // Bind with same random values, but regular_eq uses top and split uses new high-to-low
        for _ in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            regular_eq.bound_poly_var_top(&r);
            split_eq_high_to_low.bind(r);
            let merged = split_eq_high_to_low.merge();

            assert_eq!(regular_eq.Z[..regular_eq.len()], merged.Z[..merged.len()]);
        }
    }

    #[test]
    fn par_iter_low_to_high() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let split_eq: GruenSplitEqPolynomial<Fr> =
            GruenSplitEqPolynomial::new(&w, BindingOrder::LowToHigh);
        let regular_eq = DensePolynomial::<Fr>::new(EqPolynomial::evals(&w));
        let indices: Vec<_> = split_eq.par_iter_low_to_high().map(|(i, _)| i).collect();
        let coeffs: Vec<_> = split_eq
            .par_iter_low_to_high()
            .map(|(_, coeff)| coeff)
            .collect();

        assert_eq!(indices, (0..indices.len()).collect::<Vec<_>>());
        assert_eq!(regular_eq.Z, coeffs);
    }

    #[test]
    fn par_iter_high_to_low() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let split_eq: GruenSplitEqPolynomial<Fr> =
            GruenSplitEqPolynomial::new(&w, BindingOrder::HighToLow);
        let regular_eq = DensePolynomial::<Fr>::new(EqPolynomial::evals(&w));

        let indices: Vec<_> = split_eq.par_iter_high_to_low().map(|(i, _)| i).collect();
        let coeffs: Vec<_> = split_eq
            .par_iter_high_to_low()
            .map(|(_, coeff)| coeff)
            .collect();

        assert_eq!(indices, (0..indices.len()).collect::<Vec<_>>());
        assert_eq!(regular_eq.Z, coeffs);
    }

    /// For window_size = 1, `E_out_in_for_window` should reduce to the existing
    /// split-eq behaviour (`E_out_current`, `E_in_current`) for all rounds.
    #[test]
    fn window_size_one_matches_current() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut split_eq: GruenSplitEqPolynomial<Fr> =
            GruenSplitEqPolynomial::new(&w, BindingOrder::LowToHigh);

        for _round in 0..NUM_VARS {
            let (e_out_window, e_in_window) = split_eq.E_out_in_for_window(1);
            assert_eq!(e_out_window, split_eq.E_out_current());
            assert_eq!(e_in_window, split_eq.E_in_current());

            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            split_eq.bind(r);
        }
    }

    /// Check basic bit-accounting invariants for `E_out_in_for_window`:
    ///   log2(|E_out|) + log2(|E_in|) + 1 = number of unbound variables
    /// for all window sizes and all rounds (LowToHigh).
    #[test]
    fn window_bit_accounting_invariants() {
        const NUM_VARS: usize = 8;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut split_eq: GruenSplitEqPolynomial<Fr> =
            GruenSplitEqPolynomial::new(&w, BindingOrder::LowToHigh);

        // Walk through all rounds, checking all window sizes that are
        // meaningful at that point (at least one unbound variable).
        for _round in 0..NUM_VARS {
            let num_unbound = split_eq.len().log_2();
            if num_unbound == 0 {
                break;
            }

            for window_size in 1..=num_unbound {
                let (e_out, e_in) = split_eq.E_out_in_for_window(window_size);
                // By construction, an "empty" side is represented as a [1] table.
                debug_assert!(!e_out.is_empty());
                debug_assert!(!e_in.is_empty());

                let bits_out = e_out.len().log_2();
                let bits_in = e_in.len().log_2();

                // One bit is reserved for the current variable in the Gruen
                // cubic (the eq polynomial is linear in that bit).
                assert_eq!(
                    bits_out + bits_in + 1,
                    num_unbound,
                    "bit accounting failed for window_size={} (bits_out={}, bits_in={}, num_unbound={})",
                    window_size,
                    bits_out,
                    bits_in,
                    num_unbound,
                );
            }

            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            split_eq.bind(r);
        }
    }
}
