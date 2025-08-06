#![allow(clippy::uninlined_format_args)]
use ark_bn254::Fr;
use ark_ff::Zero;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::field::JoltField;
use jolt_core::{
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::math::Math,
};

fn sparse_inputs(n: u64, c: f64) -> (MultilinearPolynomial<Fr>, Vec<Fr>) {
    assert!(n.is_power_of_two(), "n must be a power of 2");
    let mut rng = StdRng::seed_from_u64(123);
    // Compute number of zeros
    // Each position independently: zero with prob c, random otherwise
    let values: Vec<Fr> = (0..n)
        .map(|_| {
            if rng.gen::<f64>() < c {
                Fr::zero()
            } else {
                Fr::random(&mut rng)
            }
        })
        .collect();

    let poly = MultilinearPolynomial::from(values);

    // Random evaluation point remains unchanged
    let eval_point = (0..(n as usize).log_2())
        .map(|_| Fr::random(&mut rng))
        .collect::<Vec<_>>();

    (poly, eval_point)
}

fn bench_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_polynomial_evaluation_large_scalars_with_sparsity");
    // group.measurement_time(std::time::Duration::from_secs(60));

    for exp in [12, 14, 16, 18, 20, 22] {
        for sparsity in [0.2, 0.6, 0.85] {
            let num_vars = 1 << exp; // 2^exp
            let (poly, eval_point) = sparse_inputs(num_vars as u64, sparsity);

            let id_dot = format!("dot-product-{}-{}", exp, sparsity);
            group.bench_function(&id_dot, |b| {
                b.iter(|| poly.evaluate_dot_product(eval_point.as_slice()))
            });

            let id_opt = format!("inside-out-{}-{}", exp, sparsity);
            group.bench_function(&id_opt, |b| b.iter(|| poly.evaluate(eval_point.as_slice())));

            let id_split = format!("split_eq-{}-{}", exp, sparsity);
            group.bench_function(&id_split, |b| {
                b.iter(|| poly.evaluate_sparse_dot_product(eval_point.as_slice()))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
