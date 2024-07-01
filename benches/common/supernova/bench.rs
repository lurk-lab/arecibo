// Code is considered dead unless used in all benchmark targets
#![allow(dead_code)]

use crate::common::supernova::{
  num_cons, NonUniformBench, SnarkType, E1, E2, NUM_CONS_VERIFIER_CIRCUIT_PRIMARY, NUM_SAMPLES,
};
use crate::common::{noise_threshold_env, BenchParams};
use arecibo::provider::{PallasEngine, VestaEngine};
use arecibo::{
  supernova::NonUniformCircuit,
  supernova::{snark::CompressedSNARK, PublicParams, RecursiveSNARK},
  traits::{
    snark::RelaxedR1CSSNARKTrait,
    snark::{default_ck_hint, BatchedRelaxedR1CSSNARKTrait},
    Engine,
  },
};
use criterion::{measurement::WallTime, *};

/// Benchmarks the SNARK at a provided number of constraints
///
/// Parameters
/// - `num_augmented_circuits`: the number of augmented circuits in this configuration
/// - `group`: the criterion benchmark group
/// - `num_cons`: the number of constraints in the step circuit
pub fn bench_snark_internal_with_arity<
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  num_augmented_circuits: usize,
  num_cons: usize,
  snark_type: SnarkType,
) {
  let bench: NonUniformBench<E1> = match snark_type {
    SnarkType::Recursive => NonUniformBench::<E1>::new(2, num_cons),
    SnarkType::Compressed => NonUniformBench::<E1>::new(num_augmented_circuits, num_cons),
  };
  let pp = match snark_type {
    SnarkType::Recursive => PublicParams::setup(&bench, &*default_ck_hint(), &*default_ck_hint()),
    SnarkType::Compressed => PublicParams::setup(&bench, &*S1::ck_floor(), &*S2::ck_floor()),
  };

  // TODO: Can we use the same number of warmup steps for recursive and compressed?
  let num_warmup_steps = match snark_type {
    SnarkType::Recursive => 10,
    SnarkType::Compressed => 3,
  };
  let z0_primary = vec![<E1 as Engine>::Scalar::from(2u64)];
  let z0_secondary = vec![<E2 as Engine>::Scalar::from(2u64)];
  let mut recursive_snark_option: Option<RecursiveSNARK<E1>> = None;
  let mut selected_augmented_circuit = 0;

  for _ in 0..num_warmup_steps {
    let mut recursive_snark = recursive_snark_option.unwrap_or_else(|| {
      RecursiveSNARK::new(
        &pp,
        &bench,
        &bench.primary_circuit(0),
        &bench.secondary_circuit(),
        &z0_primary,
        &z0_secondary,
      )
      .unwrap()
    });

    if selected_augmented_circuit == 0 || selected_augmented_circuit == 1 {
      recursive_snark
        .prove_step(
          &pp,
          &bench.primary_circuit(selected_augmented_circuit),
          &bench.secondary_circuit(),
        )
        .expect("Prove step failed");

      recursive_snark
        .verify(&pp, &z0_primary, &z0_secondary)
        .expect("Verify failed");
    } else {
      unimplemented!()
    }

    selected_augmented_circuit = (selected_augmented_circuit + 1) % num_augmented_circuits;
    recursive_snark_option = Some(recursive_snark)
  }

  assert!(recursive_snark_option.is_some());
  let recursive_snark = recursive_snark_option.unwrap();

  let bench_params = BenchParams {
    step_size: num_cons,
    date: env!("VERGEN_GIT_COMMIT_DATE"),
    sha: env!("VERGEN_GIT_SHA"),
  };

  match snark_type {
    SnarkType::Compressed => {
      let (prover_key, verifier_key) = CompressedSNARK::<_, S1, S2>::setup(&pp).unwrap();
      // Benchmark the prove time
      group.bench_function(bench_params.bench_id("Prove"), |b| {
        b.iter(|| {
          CompressedSNARK::<_, S1, S2>::prove(
            black_box(&pp),
            black_box(&prover_key),
            black_box(&recursive_snark),
          )
          .unwrap();
        })
      });

      let compressed_snark =
        CompressedSNARK::<_, S1, S2>::prove(&pp, &prover_key, &recursive_snark).unwrap();
      // Benchmark the verification time
      group.bench_function(bench_params.bench_id("Verify"), |b| {
        b.iter(|| {
          black_box(&compressed_snark)
            .verify(
              black_box(&pp),
              black_box(&verifier_key),
              black_box(&z0_primary),
              black_box(&z0_secondary),
            )
            .unwrap();
        })
      });
    }
    SnarkType::Recursive => {
      // Benchmark the prove time
      group.bench_function(bench_params.bench_id("Prove"), |b| {
        b.iter(|| {
          black_box(&mut recursive_snark.clone())
            .prove_step(
              black_box(&pp),
              &bench.primary_circuit(0),
              &bench.secondary_circuit(),
            )
            .unwrap();
        })
      });

      // Benchmark the verification time
      group.bench_function(bench_params.bench_id("Verify"), |b| {
        b.iter(|| {
          black_box(&mut recursive_snark.clone())
            .verify(
              black_box(&pp),
              black_box(&[<PallasEngine as Engine>::Scalar::from(2u64)]),
              black_box(&[<VestaEngine as Engine>::Scalar::from(2u64)]),
            )
            .unwrap();
        })
      });
    }
  }
}

pub fn run_bench<S1: BatchedRelaxedR1CSSNARKTrait<E1>, S2: RelaxedR1CSSNARKTrait<E2>>(
  c: &mut Criterion,
  group_name: &str,
  arity: usize,
  snark_type: SnarkType,
) {
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in num_cons().iter() {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit
      .checked_sub(NUM_CONS_VERIFIER_CIRCUIT_PRIMARY)
      .expect("Negative `num_cons`, constraint numbers out of date!");

    let mut group = c.benchmark_group(group_name);
    group.sample_size(NUM_SAMPLES);
    group.noise_threshold(noise_threshold_env().unwrap_or(0.3));

    bench_snark_internal_with_arity::<S1, S2>(&mut group, arity, num_cons, snark_type);

    group.finish();
  }
}
