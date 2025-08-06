//use ark_bn254::Fr;
use ark_ff::UniformRand;
//use ark_ff::{One, Zero};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use jolt_core::field::tracked_ark::TrackedFr as Fr;
use jolt_core::field::JoltField;
//use jolt_core::poly::multilinear_polynomial::{PolynomialBinding, PolynomialEvaluation};
use jolt_core::poly::{
    eq_poly::EqPolynomial,
    //multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
};
//use jolt_core::subprotocols::shout::{
//    prove_generic_core_shout_piop_d_is_one_w_gruen, prove_generic_core_shout_pip,
//};
//use jolt_core::utils::counters::{get_mult_count, reset_mult_count};
use jolt_core::utils::math::Math;
//use jolt_core::utils::math::Math;
//use jolt_core::utils::transcript::KeccakTranscript;
//use jolt_core::utils::transcript::Transcript;
use rand_core::RngCore;
use rayon::prelude::*;
use std::time::Instant;

fn construct_vector_c_in_shout_faster<F: JoltField>(
    table_size: usize,
    read_addresses: &Vec<usize>,
    e_star: &Vec<F>,
) -> Vec<F> {
    let c = read_addresses
        .par_iter()
        .zip(e_star.par_iter())
        .fold(
            || vec![F::zero(); table_size],
            |mut acc, (&address, &val)| {
                if address < table_size {
                    acc[address] += val;
                }
                acc
            },
        )
        .reduce(
            || vec![F::zero(); table_size],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b) {
                    *ai += bi;
                }
                a
            },
        );

    c
}

fn construct_vector_c_in_shout<F: JoltField>(
    table_size: usize,
    read_addresses: &Vec<usize>,
    e_star: &Vec<F>,
) -> Vec<F> {
    let c: Vec<_> = (0..table_size) // This is C[x] = ra(r_cycle, x)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| {
                    if *address == k {
                        // this check will be more complex for d > 1 but let's keep
                        // this for now
                        Some(e_star[cycle])
                    } else {
                        None
                    }
                })
                .sum::<F>()
        })
        .collect();
    c
}

fn main() {
    const K: usize = 64;
    const T: usize = 1 << 20;
    const D: usize = 2;

    let n = (K as f64).powf(1.0 / D as f64).round() as usize;
    assert_eq!(n.pow(D as u32), K, "K must be a perfect power of N");

    let seed1: u64 = 42;
    let mut rng1 = StdRng::seed_from_u64(seed1);
    let _lookup_table: Vec<Fr> = (0..K).map(|_| Fr::rand(&mut rng1)).collect();
    let read_addresses: Vec<usize> = (0..T).map(|_| (rng1.next_u32() as usize) % K).collect();

    let r_cycle = (0..T.log_2())
        .map(|_| Fr::rand(&mut rng1))
        .collect::<Vec<Fr>>();
    let e_star = EqPolynomial::evals(&r_cycle);

    let start = Instant::now();
    let c1 = construct_vector_c_in_shout(K, &read_addresses, &e_star);
    let duration = start.elapsed();
    println!("Time to construct C: {}", duration.as_millis());

    let start = Instant::now();
    let c2 = construct_vector_c_in_shout_faster(K, &read_addresses, &e_star);
    let duration = start.elapsed();
    println!("Time to construct C: {}", duration.as_millis());
    assert!(c1.iter().zip(c2).all(|(x, y)| *x == *y), "Arrays diffe",);
    //const K: usize = 64;
    //const T: usize = 1 << 6;
    //let log_k = K.log_2();
    //let a: Vec<TrackedFr> = (0..log_k).map(|_| TrackedFr::from_u8(2)).collect();
    //let _ = EqPolynomial::evals(&a);
    //println!("{}", get_mult_count());
    //reset_mult_count();
    //
    //let mut rng = StdRng::seed_from_u64(122);
    //let b = (0..K)
    //    .map(|_| TrackedFr::rand(&mut rng))
    //    .collect::<Vec<TrackedFr>>();
    //
    //let c = (0..K)
    //    .map(|_| TrackedFr::rand(&mut rng))
    //    .collect::<Vec<TrackedFr>>();
    //
    //reset_mult_count();
    //let mut ra = MultilinearPolynomial::from(b);
    //
    //let mut val = MultilinearPolynomial::from(c);
    //let degree = 2;
    //for round in 0..K.log_2() {
    //    let _univariate_poly_evals: Vec<TrackedFr> = (0..ra.len() / 2)
    //        .into_par_iter()
    //        .map(|index| {
    //            let ra_evals = ra.sumcheck_evals(index, degree, BindingOrder::LowToHigh);
    //            let val_evals = val.sumcheck_evals(index, degree, BindingOrder::LowToHigh);
    //            vec![ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]] // since DEGREE=2
    //        })
    //        .reduce(
    //            || vec![TrackedFr::zero(); degree],
    //            |running, new| [running[0] + new[0], running[1] + new[1]].to_vec(),
    //        );
    //    //println!("{},", univariate_poly_evals[0]);
    //    println!("Before ra {}", get_mult_count());
    //    ra.bind_parallel(TrackedFr::from_u32(102), BindingOrder::LowToHigh);
    //    println!("Before Bal {}", get_mult_count());
    //    val.bind_parallel(TrackedFr::from_u32(102), BindingOrder::LowToHigh);
    //    println!("{}", get_mult_count());
    //    println!("Round {round} over \n");
    //    reset_mult_count();
    //}
}
