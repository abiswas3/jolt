#![allow(clippy::uninlined_format_args)]
use ark_bn254::Fr;
use ark_ff::Zero;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::field::JoltField;
use jolt_core::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use jolt_core::utils::math::Math;

fn setup_sparse_batch_inputs(
    n: usize,
    batch_size: usize,
    sparsity: f64,
) -> (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>) {
    assert!(n.is_power_of_two(), "n must be a power of 2");
    let mut rng = StdRng::seed_from_u64(123);
    let mut polys: Vec<MultilinearPolynomial<Fr>> = Vec::new();
    for _ in 0..batch_size {
        // Compute number of zeros
        // Each position independently: zero with prob c, random otherwise
        let values: Vec<Fr> = (0..n)
            .map(|_| {
                if rng.gen::<f64>() < sparsity {
                    Fr::zero()
                } else {
                    Fr::random(&mut rng)
                }
            })
            .collect();

        let poly = MultilinearPolynomial::from(values);
        polys.push(poly);
    }
    // Random evaluation point remains unchanged
    let eval_point = (0..n.log_2())
        .map(|_| Fr::random(&mut rng))
        .collect::<Vec<_>>();

    (polys, eval_point)
}

fn bench_batch_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched_polynomial_evaluation_large_scalar");
    group.measurement_time(std::time::Duration::from_secs(20));

    let batch_size = 10;
    let sparsity = 0.8;
    for n in [16, 18] {
        let (polys, eval_loc) = setup_sparse_batch_inputs(1 << n, batch_size, sparsity);
        assert_eq!(eval_loc.len(), n);
        let poly_refs: Vec<&MultilinearPolynomial<Fr>> = polys.iter().collect();
        let id_dot = format!("dot-product-{}", n);
        group.bench_function(&id_dot, |b| {
            b.iter(|| MultilinearPolynomial::batch_evaluate(&poly_refs, &eval_loc))
        });

        // This is still dot producting
        let id_opt = format!("split_eq-{}", n);
        group.bench_function(&id_opt, |b| {
            b.iter(|| {
                MultilinearPolynomial::batch_evaluate_optimised_for_sparse_polynomials(
                    &poly_refs, &eval_loc,
                )
            })
        });

        let id_opt = format!("inside_out-{}", n);
        group.bench_function(&id_opt, |b| {
            b.iter(|| MultilinearPolynomial::batch_evaluate_inside_out(&poly_refs, &eval_loc))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_batch_evaluate);
criterion_main!(benches);
