use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::subprotocols::shout::{
    prove_generic_core_shout_piop_d_is_one_w_gruen, prove_generic_core_shout_pip,
    prove_generic_core_shout_pip_d_greater_than_one,
    prove_generic_core_shout_pip_d_greater_than_one_with_gruen,
};
use jolt_core::utils::transcript::KeccakTranscript;
use jolt_core::utils::transcript::Transcript;
use rand_core::RngCore;
use std::time::Duration;

pub fn bench_prove(c: &mut Criterion) {
    // ------- PROBLEM SETUP ----------------------
    const K: usize = 64;
    const T: usize = 1 << 18;
    const D: usize = 2;

    let n = (K as f64).powf(1.0 / D as f64).round() as usize;
    assert_eq!(n.pow(D as u32), K, "K must be a perfect power of N");

    let seed1: u64 = 42;
    let mut rng1 = StdRng::seed_from_u64(seed1);
    let lookup_table: Vec<Fr> = (0..K).map(|_| Fr::rand(&mut rng1)).collect();
    let read_addresses: Vec<usize> = (0..T).map(|_| (rng1.next_u32() as usize) % K).collect();

    // ------- CUSTOM CRITERION SETTINGS ---------
    let mut group = c.benchmark_group("prove_benchmarks");
    group.sample_size(20); // fewer samples
    group.warm_up_time(Duration::from_secs(30)); // longer warmup
    group.measurement_time(Duration::from_secs(60)); // run longer

    // ------- BENCHMARKS -----------------------
    //group.bench_function("prove_generic_core_shout_pip_d_greater_than_one", |b| {
    //    let lookup_table_1 = lookup_table.clone();
    //    let read_addresses_1 = read_addresses.clone();
    //    b.iter(|| {
    //        let mut transcript = KeccakTranscript::new(b"bench");
    //        prove_generic_core_shout_pip_d_greater_than_one(
    //            lookup_table_1.clone(),
    //            read_addresses_1.clone(),
    //            D,
    //            &mut transcript,
    //        );
    //    });
    //});
    //
    //group.bench_function(
    //    "prove_generic_core_shout_pip_d_greater_than_one_with_gruen",
    //    |b| {
    //        let lookup_table_1 = lookup_table.clone();
    //        let read_addresses_1 = read_addresses.clone();
    //        b.iter(|| {
    //            let mut transcript = KeccakTranscript::new(b"bench");
    //            prove_generic_core_shout_pip_d_greater_than_one_with_gruen(
    //                lookup_table_1.clone(),
    //                read_addresses_1.clone(),
    //                D,
    //                &mut transcript,
    //            );
    //        });
    //    },
    //);
    //
    group.bench_function("prove_generic_core_shout_pip_d_is_one", |b| {
        let lookup_table_1 = lookup_table.clone();
        let read_addresses_1 = read_addresses.clone();
        b.iter(|| {
            let mut transcript = KeccakTranscript::new(b"bench");
            prove_generic_core_shout_pip(
                lookup_table_1.clone(),
                read_addresses_1.clone(),
                &mut transcript,
            );
        });
    });

    group.bench_function(
        "prove_generic_core_shout_pip_d_greater_than_one_with_gruen",
        |b| {
            let lookup_table_1 = lookup_table.clone();
            let read_addresses_1 = read_addresses.clone();
            b.iter(|| {
                let mut transcript = KeccakTranscript::new(b"bench");
                prove_generic_core_shout_piop_d_is_one_w_gruen(
                    lookup_table_1.clone(),
                    read_addresses_1.clone(),
                    &mut transcript,
                );
            });
        },
    );

    group.finish();
}
criterion_group!(benches, bench_prove);
criterion_main!(benches);
