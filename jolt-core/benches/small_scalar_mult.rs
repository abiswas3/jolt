#![allow(clippy::uninlined_format_args)]
use ark_bn254::Fr;
use ark_ff::Zero;
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::field::JoltField;
use jolt_core::poly::compact_polynomial::SmallScalar;
use jolt_core::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn mult_with_addr(mut a: Fr, mut b: u8) -> Fr {
    let mut result = Fr::zero();

    while b > 0 {
        if b & 1 == 1 {
            result += a;
        }
        a = a + a; // equivalent to a + a
        b >>= 1;
    }

    result
}
fn mult_with_field_mul(a: Fr, b: u8) -> Fr {
    b.field_mul(a)
}

fn small_scalar_mults(c: &mut Criterion) {
    let mut group = c.benchmark_group("Small scalar multiplication");
    let mut rng = ChaCha20Rng::seed_from_u64(123_u64);
    let a = Fr::random(&mut rng);
    let b = Fr::random(&mut rng);
    //let c = rand::random::<u8>();
    let c = 2_u8;
    let z = a * b;
    // First kind of multiplication
    //c.field_mul(z);
    let id_opt = "full mul".to_string();
    group.bench_function(&id_opt, |b| b.iter(|| z * c.to_field::<Fr>()));

    let id_opt = "field_mul".to_string();
    group.bench_function(&id_opt, |b| b.iter(|| mult_with_field_mul(z, c)));

    let id_opt = "mul via addr".to_string();
    group.bench_function(&id_opt, |b| b.iter(|| mult_with_addr(z, c)));

    group.finish();
}

criterion_group!(benches, small_scalar_mults);
criterion_main!(benches);
