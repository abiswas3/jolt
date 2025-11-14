use allocative::Allocative;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};

/// Multiquadratic polynomial represented by its evaluations on the grid
/// {0, 1, ∞}^num_vars in base-3 layout (z_0 least-significant / fastest-varying).
#[derive(Allocative)]
pub struct MultiquadraticPolynomial<F: JoltField> {
    num_vars: usize,
    evals: Vec<F>,
}

impl<F: JoltField> MultiquadraticPolynomial<F> {
    /// Construct a multiquadratic polynomial from its full grid of evaluations.
    /// The caller is responsible for ensuring that `evals` is laid out in base-3
    /// order with z_0 as the least-significant digit.
    pub fn new(num_vars: usize, evals: Vec<F>) -> Self {
        let expected_len = 3usize.pow(num_vars as u32);
        debug_assert!(
            evals.len() == expected_len,
            "MultiquadraticPolynomial: expected {} evals, got {}",
            expected_len,
            evals.len()
        );
        Self { num_vars, evals }
    }

    /// Number of variables in the polynomial.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Underlying evaluations on {0, 1, ∞}^num_vars.
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Given evaluations of a degree-1 multivariate polynomial over {0,1}^dim,
    /// expand them to the corresponding multiquadratic grid over {0,1,∞}^dim.
    ///
    /// The input is a length-2^dim slice `input` containing evaluations on the
    /// Boolean hypercube. The caller must provide two length-3^dim buffers:
    /// - `output` will contain the final {0,1,∞}^dim values on return
    /// - `tmp` is a scratch buffer which this routine may use internally
    ///
    /// Layout is product-order with the last variable as the fastest-varying
    /// coordinate. For each 1D slice (f0, f1) along a new dimension we write
    /// (f(0), f(1), f(∞)) = (f0, f1, f1 - f0), so ∞ stores the slope.
    ///
    /// TODO: special-case dim ∈ {1,2,3} with hand-unrolled code to reduce
    /// loop overhead on small windows.
    #[inline(always)]
    pub fn expand_linear_grid_to_multiquadratic(
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

    /// Bind the first (least-significant) variable z_0 := r, reducing the
    /// dimension from w to w-1 and keeping the base-3 layout invariant.
    ///
    /// For each assignment to (z_1, ..., z_{w-1}), we have three stored values
    ///   f(0, ..), f(1, ..), f(∞, ..)
    /// and interpolate the unique quadratic in z_0 that matches them, then
    /// evaluate it at z_0 = r.
    pub fn bind_first_variable(&mut self, r: F::Challenge) {
        let w = self.num_vars;
        debug_assert!(w > 0);

        let new_size = 3_usize.pow((w - 1) as u32);
        let one = F::one();

        for new_idx in 0..new_size {
            let old_base_idx = new_idx * 3;
            let eval_at_0 = self.evals[old_base_idx]; // z_0 = 0
            let eval_at_1 = self.evals[old_base_idx + 1]; // z_0 = 1
            let eval_at_inf = self.evals[old_base_idx + 2]; // z_0 = ∞

            self.evals[new_idx] =
                eval_at_0 * (one - r) + eval_at_1 * r + eval_at_inf * r * (r - one);
        }

        self.num_vars -= 1;
        self.evals.truncate(new_size);
    }

    /// Project t'(z_0, z_1, ..., z_{w-1}) to a univariate in z_0 by summing
    /// against `E_active` over the remaining coordinates.
    ///
    /// The `E_active` table is interpreted identically to the existing outer
    /// Spartan streaming implementation: each index encodes, in binary, which
    /// of z_1..z_{w-1} take the "active" value (mapped to base-3 offset 1).
    /// `first_coord_val` is the z_0 coordinate in {0, 1, 2}, where 2 encodes ∞.
    pub fn project_to_first_variable(&self, E_active: &[F], first_coord_val: usize) -> F {
        let w = self.num_vars;
        debug_assert!(w >= 1);

        let offset = first_coord_val; // z_0 lives at the units place in base-3

        E_active
            .iter()
            .enumerate()
            .map(|(eq_active_idx, eq_active_val)| {
                let mut index = offset;
                let mut temp = eq_active_idx;
                let mut power = 3; // start at 3^1 for z_1

                for _ in 0..(w - 1) {
                    if temp & 1 == 1 {
                        index += power;
                    }
                    power *= 3;
                    temp >>= 1;
                }

                self.evals[index] * *eq_active_val
            })
            .sum()
    }
}

impl<F: JoltField> PolynomialBinding<F> for MultiquadraticPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_vars == 0 || self.evals.len() == 1
    }

    #[tracing::instrument(skip_all, name = "MultiquadraticPolynomial::bind")]
    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bind_first_variable(r),
            BindingOrder::HighToLow => {
                // Not currently needed by the outer Spartan streaming code.
                unimplemented!(
                    "HighToLow binding order is not implemented for MultiquadraticPolynomial"
                )
            }
        }
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Window sizes are small; fall back to the sequential implementation.
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert!(self.is_bound());
        debug_assert_eq!(self.evals.len(), 1);
        self.evals[0]
    }
}
