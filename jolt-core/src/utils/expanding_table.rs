use allocative::Allocative;
use rayon::prelude::*;
use std::ops::Index;

use crate::field::JoltField;
use crate::utils::thread::unsafe_allocate_zero_vec;

/// Controls how an `ExpandingTable` interprets indices as bit-strings.
///
/// - `LeastSignificantBit`: the most recently bound challenge corresponds to
///   the least-significant bit of the index (this matches the original
///   `r_grid` semantics used in the outer Spartan sumcheck).
/// - `MostSignificantBit`: the most recently bound challenge corresponds to
///   the most-significant bit of the index; this view is implemented by
///   bit-reversing the index before looking into the underlying storage.
#[derive(Clone, Copy, Debug, Default, Allocative)]
pub enum ExpansionOrder {
    #[default]
    LeastSignificantBit,
    MostSignificantBit,
}

/// Table containing the evaluations `EQ(x_1, ..., x_j, r_1, ..., r_j)`,
/// built up incrementally as we receive random challenges `r_j` over the
/// course of sumcheck.
#[derive(Clone, Debug, Allocative)]
pub struct ExpandingTable<F: JoltField> {
    len: usize,
    values: Vec<F>,
    scratch_space: Vec<F>,
    order: ExpansionOrder,
}

impl<F: JoltField> Default for ExpandingTable<F> {
    fn default() -> Self {
        Self {
            len: 0,
            values: Vec::new(),
            scratch_space: Vec::new(),
            order: ExpansionOrder::LeastSignificantBit,
        }
    }
}

impl<F: JoltField> ExpandingTable<F> {
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the indexing order used by this table.
    pub fn order(&self) -> ExpansionOrder {
        self.order
    }

    /// Initializes an `ExpandingTable` with the given `capacity`, using
    /// least-significant-bit expansion order (backwards compatible default).
    #[tracing::instrument(skip_all, name = "ExpandingTable::new")]
    pub fn new(capacity: usize) -> Self {
        Self::new_with_order(capacity, ExpansionOrder::LeastSignificantBit)
    }

    /// Initializes an `ExpandingTable` with the given `capacity` and explicit
    /// bit order for index interpretation.
    #[tracing::instrument(skip_all, name = "ExpandingTable::new_with_order")]
    pub fn new_with_order(capacity: usize, order: ExpansionOrder) -> Self {
        let (values, scratch_space) = rayon::join(
            || unsafe_allocate_zero_vec(capacity),
            || unsafe_allocate_zero_vec(capacity),
        );
        Self {
            len: 0,
            values,
            scratch_space,
            order,
        }
    }

    /// Resets this table to be length 1, containing only the given `value`.
    pub fn reset(&mut self, value: F) {
        self.values[0] = value;
        self.len = 1;
    }

    pub fn clone_values(&self) -> Vec<F> {
        self.values[..self.len].to_vec()
    }

    /// Updates this table (expanding it by a factor of 2) to incorporate
    /// the new random challenge `r_j`.
    #[tracing::instrument(skip_all, name = "ExpandingTable::update")]
    pub fn update(&mut self, r_j: F::Challenge) {
        self.values[..self.len]
            .par_iter()
            .zip(self.scratch_space.par_chunks_mut(2))
            .for_each(|(&v_i, dest)| {
                let eval_1 = r_j * v_i;
                dest[0] = v_i - eval_1;
                dest[1] = eval_1;
            });
        std::mem::swap(&mut self.values, &mut self.scratch_space);
        self.len *= 2;
    }
}

impl<F: JoltField> Index<usize> for ExpandingTable<F> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        assert!(index < self.len, "index: {}, len: {}", index, self.len);
        let physical_index = match self.order {
            ExpansionOrder::LeastSignificantBit => index,
            ExpansionOrder::MostSignificantBit => {
                // Reverse the lowest `bits` bits of `index` so that the most
                // recently-bound challenge is treated as the MSB.
                if self.len <= 1 {
                    0
                } else {
                    let bits = self.len.trailing_zeros() as usize;
                    debug_assert_eq!(1usize << bits, self.len);
                    let mut i = index;
                    let mut rev = 0usize;
                    for _ in 0..bits {
                        rev = (rev << 1) | (i & 1);
                        i >>= 1;
                    }
                    rev
                }
            }
        };
        &self.values[physical_index]
    }
}

impl<'data, F: JoltField> IntoParallelIterator for &'data ExpandingTable<F> {
    type Item = &'data F;
    type Iter = rayon::slice::Iter<'data, F>;

    fn into_par_iter(self) -> Self::Iter {
        self.values[..self.len].into_par_iter()
    }
}
