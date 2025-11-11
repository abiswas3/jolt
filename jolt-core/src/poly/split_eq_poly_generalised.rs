//! Generalised SplitEq from Thaler/Dao
//! //! https://eprint.iacr.org/2024/1210.pdf

use super::multilinear_polynomial::BindingOrder;
use crate::{field::JoltField, poly::eq_poly::EqPolynomial};
use allocative::Allocative;

#[derive(Debug, Clone, PartialEq, Allocative)]
pub struct GruenSplitEqPolynomialGeneral<F: JoltField> {
    pub(crate) current_index: usize,
    pub(crate) window_end_index: usize,
    pub(crate) current_scalar: F,
    pub w: Vec<F::Challenge>,
    pub(crate) E_in_vec: Vec<Vec<F>>,
    pub(crate) E_out_vec: Vec<Vec<F>>,
    pub(crate) E_active: Vec<Vec<F>>,
    pub(crate) challenges: Vec<F::Challenge>,
    pub(crate) window_size: usize,
    pub(crate) binding_order: BindingOrder,
}

impl<F: JoltField> GruenSplitEqPolynomialGeneral<F> {
    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomialGeneral::new_with_scaling")]
    pub fn new_with_scaling(
        w: &[F::Challenge],
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
        window_size: usize,
    ) -> Self {
        match binding_order {
            BindingOrder::LowToHigh => {
                // First, split off the window from the end
                let window_start = w.len() - window_size;
                let (w_body, w_window) = w.split_at(window_start);

                // Split the window into active and current
                let (w_active, w_curr_slice) = w_window.split_at(window_size - 1);
                let _ = w_curr_slice[0]; // The current/last variable

                // Now split w_body into w_out and w_in
                let m = w_body.len() / 2;
                let (w_out, w_in) = w_body.split_at(m);

                // Compute evaluations in parallel
                let (E_out_vec, rest) = rayon::join(
                    || EqPolynomial::evals_cached(w_out),
                    || {
                        rayon::join(
                            || EqPolynomial::evals_cached(w_in),
                            || EqPolynomial::evals_cached(w_active), // Only the active portion
                        )
                    },
                );
                let (E_in_vec, E_active) = rest;

                let challenges: Vec<F::Challenge> = Vec::new();
                Self {
                    current_index: w.len() - 1, // Points to w0 (the current free variable)
                    window_end_index: w.len() - window_size,
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    E_active,
                    challenges,
                    window_size,
                    binding_order,
                }
            }
            BindingOrder::HighToLow => {
                todo!();
            }
        }
    }

    pub fn evaluate_curr_at_zero(&self) -> F {
        F::one() - self.w[self.current_index]
    }

    pub fn evaluate_curr_at_one(&self) -> F {
        self.w[self.current_index].into()
    }

    pub fn num_challenges(&self) -> usize {
        self.challenges.len()
    }
    pub fn get_challenges(&self) -> &[F::Challenge] {
        self.challenges.as_slice()
    }
    pub fn new(w: &[F::Challenge], binding_order: BindingOrder, window_size: usize) -> Self {
        Self::new_with_scaling(w, binding_order, None, window_size)
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

    pub fn E_active_current_len(&self) -> usize {
        self.E_active.last().map_or(0, |v| v.len())
    }

    pub fn E_in_current(&self) -> &[F] {
        self.E_in_vec.last().map_or(&[], |v| v.as_slice())
    }

    pub fn E_out_current(&self) -> &[F] {
        self.E_out_vec.last().map_or(&[], |v| v.as_slice())
    }

    pub fn E_active_current(&self) -> &[F] {
        self.E_active.last().map_or(&[], |v| v.as_slice())
    }

    pub fn get_cuurent_idx(&self) -> usize {
        self.current_index
    }

    pub fn get_window_end(&self) -> usize {
        self.window_end_index
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
                BindingOrder::LowToHigh => self.w[self.current_index],
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
                todo!("Not yet implemented");
            }
        }
    }
}
