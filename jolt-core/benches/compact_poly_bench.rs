#![allow(clippy::uninlined_format_args)]
use ark_bn254::Fr;
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::field::JoltField;
use jolt_core::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use jolt_core::utils::math::Math;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
fn setup_u8_inputs(n: usize, c: f64) -> (MultilinearPolynomial<Fr>, Vec<Fr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(n as u64);
    let coeffs: Vec<u8> = (0..n)
        .map(|_| {
            if rand::random::<f64>() < c {
                0_u8
            } else {
                rand::random::<u8>()
            }
        })
        .collect();

    let poly = MultilinearPolynomial::U8Scalars(
        jolt_core::poly::compact_polynomial::CompactPolynomial::from_coeffs(coeffs),
    );

    let eval_point = (0..n.log_2()).map(|_| Fr::random(&mut rng)).collect();

    (poly, eval_point)
}
fn bench_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_polynomial_evaluation_u8_with_sparsity");
    //group.measurement_time(std::time::Duration::from_secs(20));

    for &prob_of_seeing_zero in &[0.2, 0.6, 0.8] {
        for &exp in &[14, 16, 18] {
            let num_vars = 1 << exp; // 2^exp
            let (poly, eval_point) = setup_u8_inputs(num_vars, prob_of_seeing_zero);

            let id_dot = format!("u8-dot-product-{}-p{}", exp, prob_of_seeing_zero);
            group.bench_function(&id_dot, |b| {
                b.iter(|| poly.evaluate_dot_product(&eval_point))
            });

            let id_opt = format!("u8-inside-out-{}-p{}", exp, prob_of_seeing_zero);
            group.bench_function(&id_opt, |b| b.iter(|| poly.evaluate(&eval_point)));

            let id_sparse = format!("u8-sparse-eq-{}-p{}", exp, prob_of_seeing_zero);
            group.bench_function(&id_sparse, |b| {
                b.iter(|| poly.evaluate_sparse_dot_product(&eval_point))
            });
        }
    }

    group.finish();
}
criterion_group!(benches, bench_u8);
criterion_main!(benches);
