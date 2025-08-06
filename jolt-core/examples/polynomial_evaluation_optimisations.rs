#![allow(clippy::uninlined_format_args)]
//use ark_bn254::Fr;
use ark_ff::Zero;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use jolt_core::field::tracked_ark::TrackedFr as Fr;
use jolt_core::field::JoltField;
use jolt_core::utils::counters::{get_mult_count, reset_mult_count};
use jolt_core::{
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::math::Math,
};

use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;
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

fn _bench_batch_evaluate() {
    let batch_size = 10;
    for n in [12, 14] {
        let (polys, eval_loc) = setup_batch_inputs(n, batch_size);
        assert_eq!(eval_loc.len(), n);
        let poly_refs: Vec<&MultilinearPolynomial<Fr>> = polys.iter().collect();
        let _ = MultilinearPolynomial::batch_evaluate(&poly_refs, &eval_loc);
        // This is still dot producting
        let _ = MultilinearPolynomial::batch_evaluate_optimised_for_sparse_polynomials(
            &poly_refs, &eval_loc,
        );

        let _ = MultilinearPolynomial::batch_evaluate_optimised_for_sparse_polynomials(
            &poly_refs, &eval_loc,
        );
    }
}

fn main() {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("results.csv")
        .expect("Unable to open file");

    // Write CSV header
    writeln!(file, "exp,num_vars,c,algorithm,time_ms,mults,theory,trial").unwrap();

    let num_trials = 10;

    for exp in [16, 18, 20, 22, 24] {
        let num_vars = 1 << exp;

        for c in [0.20, 0.35, 0.50, 0.66, 0.75] {
            for trial in 0..num_trials {
                let (poly, eval_point) = sparse_inputs(num_vars as u64, c);

                let sparsity: u64 = (0..poly.len())
                    .map(|i| if poly.get_coeff(i).is_zero() { 1 } else { 0 })
                    .sum();
                let non_zero = poly.len() as u64 - sparsity;

                // --- Algorithm 1: Dot Product ---
                reset_mult_count();
                let start = Instant::now();
                let dot_prod = poly.evaluate_dot_product(&eval_point);
                let time_ms = start.elapsed().as_millis();
                let mults = get_mult_count();
                let theory = 2 * num_vars;
                writeln!(
                    file,
                    "{},{},{},DotProduct,{}, {}, {}, {}",
                    exp, num_vars, c, time_ms, mults, theory, trial
                )
                .unwrap();

                // --- Algorithm 2: Inside-Out Product ---
                reset_mult_count();
                let start = Instant::now();
                let inside_out_prod = poly.evaluate(&eval_point);
                let time_ms = start.elapsed().as_millis();
                let mults = get_mult_count();
                let theory = non_zero * non_zero - non_zero;
                writeln!(
                    file,
                    "{},{},{},InsideOut,{}, {}, {}, {}",
                    exp, num_vars, c, time_ms, mults, theory, trial
                )
                .unwrap();

                // --- Algorithm 3: Sparse Dot Product ---
                reset_mult_count();
                let start = Instant::now();
                let sparse_prod = poly.evaluate_sparse_dot_product(&eval_point);
                let time_ms = start.elapsed().as_millis();
                let mults = get_mult_count();
                let theory = 2 * ((1 << (exp / 2)) - 1) + non_zero;
                writeln!(
                    file,
                    "{},{},{},SparseDot,{}, {}, {}, {}",
                    exp, num_vars, c, time_ms, mults, theory, trial
                )
                .unwrap();

                assert_eq!(dot_prod, inside_out_prod);
                assert_eq!(dot_prod, sparse_prod);
            }
        }
    }
}
//fn main() {
//    let mut file = OpenOptions::new()
//        .create(true)
//        .write(true)
//        .truncate(true)
//        .open("results.csv")
//        .expect("Unable to open file");
//
//    // Write CSV header
//    writeln!(file, "exp,num_vars,c,algorithm,time_ms,mults,theory").unwrap();
//
//    for exp in [12, 14, 16, 18, 20, 22, 24] {
//        let num_vars = 1 << exp;
//
//        for c in [0.05, 0.25, 0.50, 0.75, 1.0] {
//            let (poly, eval_point) = sparse_inputs(num_vars as u64, c);
//
//            let sparsity: u64 = (0..poly.len())
//                .map(|i| if poly.get_coeff(i).is_zero() { 1 } else { 0 })
//                .sum();
//            let non_zero = poly.len() as u64 - sparsity;
//
//            // --- Algorithm 1: Dot Product ---
//            reset_mult_count();
//            let start = Instant::now();
//            let dot_prod = poly.evaluate_dot_product(&eval_point);
//            let time_ms = start.elapsed().as_millis();
//            let mults = get_mult_count();
//            let theory = 2 * num_vars;
//            writeln!(
//                file,
//                "{},{},{},DotProduct,{}, {}, {}",
//                exp, num_vars, c, time_ms, mults, theory
//            )
//            .unwrap();
//
//            // --- Algorithm 2: Inside-Out Product ---
//            reset_mult_count();
//            let start = Instant::now();
//            let inside_out_prod = poly.evaluate(&eval_point);
//            let time_ms = start.elapsed().as_millis();
//            let mults = get_mult_count();
//            let theory = non_zero * non_zero - non_zero;
//            writeln!(
//                file,
//                "{},{},{},InsideOut,{}, {}, {}",
//                exp, num_vars, c, time_ms, mults, theory
//            )
//            .unwrap();
//
//            // --- Algorithm 3: Sparse Dot Product ---
//            reset_mult_count();
//            let start = Instant::now();
//            let sparse_prod = poly.evaluate_sparse_dot_product(&eval_point);
//            let time_ms = start.elapsed().as_millis();
//            let mults = get_mult_count();
//            let theory = 2 * ((1 << (exp / 2)) - 1) + non_zero;
//            writeln!(
//                file,
//                "{},{},{},SparseDot,{}, {}, {}",
//                exp, num_vars, c, time_ms, mults, theory
//            )
//            .unwrap();
//
//            assert_eq!(dot_prod, inside_out_prod);
//            assert_eq!(dot_prod, sparse_prod);
//        }
//    }
//}
