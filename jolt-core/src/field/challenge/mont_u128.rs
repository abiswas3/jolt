use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
//use ark_std::{One, Zero};
use std::fmt::{Debug, Display};
use std::hash::Hash;
//use std::iter::{Product, Sum};
use std::ops::*;

/// Bespoke implementation of Challenge type that is a subset of the JoltField
/// with the property that the 2 least significant digits are 0'd out, and it needs
/// 125 bits to represent.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, Hash, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct MontU128Challenge<F: JoltField> {
    value: F,
}

// Critical: Challenge * F -> F
impl<F: JoltField> Mul<F> for MontU128Challenge<F> {
    type Output = F;

    fn mul(self, rhs: F) -> F {
        self.value * rhs
    }
}
