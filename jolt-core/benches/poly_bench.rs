#![allow(clippy::uninlined_format_args)]
use ark_bn254::Fr;
use ark_ff::{One, Zero};
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::field::JoltField;
use jolt_core::{
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::math::Math,
};
fn setup_batch_inputs(n: usize, batch_size: usize) -> (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>) {
    let mut rng = StdRng::seed_from_u64(123);
    let eval_loc: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<Fr>>();

    // This is a dense polynomial
    let polys: Vec<MultilinearPolynomial<Fr>> = (0..batch_size)
        .map(|_| {
            MultilinearPolynomial::from(
                (0..n.pow2())
                    .map(|_| Fr::random(&mut rng))
                    .collect::<Vec<Fr>>(),
            )
        })
        .collect();

    (polys, eval_loc)
}

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
    let eval_point = (0..(n as usize).ilog2())
        .map(|_| Fr::random(&mut rng))
        .collect::<Vec<_>>();

    (poly, eval_point)
}

fn _bench_batch_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("evals");
    group.measurement_time(std::time::Duration::from_secs(20));

    let batch_size = 10;
    for n in [12, 14] {
        let (polys, eval_loc) = setup_batch_inputs(n, batch_size);

        assert_eq!(eval_loc.len(), n);
        let poly_refs: Vec<&MultilinearPolynomial<Fr>> = polys.iter().collect();
        let id_dot = format!("dot-product-{}", n);
        group.bench_function(&id_dot, |b| {
            b.iter(|| MultilinearPolynomial::batch_evaluate(&poly_refs, &eval_loc))
        });

        // This is still dot producting
        let id_opt = format!("inside-out-{}", n);
        group.bench_function(&id_opt, |b| {
            b.iter(|| {
                MultilinearPolynomial::batch_evaluate_optimised_for_sparse_polynomials(
                    &poly_refs, &eval_loc,
                )
            })
        });

        let id_opt = format!("split_eq-{}", n);
        group.bench_function(&id_opt, |b| {
            b.iter(|| {
                MultilinearPolynomial::batch_evaluate_optimised_for_sparse_polynomials(
                    &poly_refs, &eval_loc,
                )
            })
        });
    }
}

fn bench_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("evals");
    //group.measurement_time(std::time::Duration::from_secs(60));

    for exp in [12, 14, 16, 18, 20, 22] {
        let num_vars = 1 << exp; // 2^exp
        let (poly, eval_point) = sparse_inputs(num_vars as u64, 0.6);
        //let (poly, eval_point) = setup_inputs(num_vars as u64);
        let id_dot = format!("dot-product-{}", exp);
        group.bench_function(&id_dot, |b| {
            b.iter(|| poly.evaluate_dot_product(eval_point.as_slice()))
        });

        let id_opt = format!("inside-out-{}", exp);
        group.bench_function(&id_opt, |b| b.iter(|| poly.evaluate(eval_point.as_slice())));

        let id_opt = format!("sparse_eq-{}", exp);
        group.bench_function(&id_opt, |b| {
            b.iter(|| poly.evaluate_sparse_dot_product(eval_point.as_slice()))
        });
    }

    group.finish();
}
criterion_group!(benches, bench_all);
criterion_main!(benches);
