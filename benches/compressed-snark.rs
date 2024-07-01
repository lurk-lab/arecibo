#![allow(non_snake_case)]
use arecibo::{
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    snark::RelaxedR1CSSNARKTrait,
    Engine,
  },
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use criterion::{measurement::WallTime, *};
use ff::PrimeField;
use halo2curves::bn256::Bn256;
use std::time::Duration;

mod common;
use common::{noise_threshold_env, BenchParams};

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type EE1 = arecibo::provider::pcs::hyperkzg::EvaluationEngine<Bn256, E1>;
type EE2 = arecibo::provider::pcs::ipa_pc::EvaluationEngine<E2>;
// SNARKs without computation commitmnets
type S1 = arecibo::spartan::snark::RelaxedR1CSSNARK<E1, EE1>;
type S2 = arecibo::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;
// SNARKs with computation commitmnets
type SS1 = arecibo::spartan::ppsnark::RelaxedR1CSSNARK<E1, EE1>;
type SS2 = arecibo::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // the computation commitment is not used for the trivial circuit

// To run these benchmarks, first download `criterion` with `cargo install cargo-criterion`.
// Then `cargo criterion --bench compressed-snark`. The results are located in `target/criterion/data/<name-of-benchmark>`.
// For flamegraphs, run `cargo criterion --bench compressed-snark --features flamegraph -- --profile-time <secs>`.
// The results are located in `target/criterion/profile/<name-of-benchmark>`.
cfg_if::cfg_if! {
  if #[cfg(feature = "flamegraph")] {
    criterion_group! {
      name = compressed_snark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
      targets = bench_compressed_snark, bench_compressed_snark_with_computational_commitments, bench_compressed_batched_snark, bench_compressed_batched_snark_with_computational_commitments
    }
  } else {
    criterion_group! {
      name = compressed_snark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000));
      targets = bench_compressed_snark, bench_compressed_snark_with_computational_commitments, bench_compressed_batched_snark, bench_compressed_batched_snark_with_computational_commitments
    }
  }
}

criterion_main!(compressed_snark);

// This should match the value for the primary in test_recursive_circuit_pasta
const NUM_CONS_VERIFIER_CIRCUIT_PRIMARY: usize = 9825;
const NUM_SAMPLES: usize = 10;

/// Benchmarks the compressed SNARK at a provided number of constraints
///
/// Parameters
/// - `group``: the criterion benchmark group
/// - `num_cons`: the number of constraints in the step circuit
fn bench_compressed_snark_internal<S1: RelaxedR1CSSNARKTrait<E1>, S2: RelaxedR1CSSNARKTrait<E2>>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  num_cons: usize,
) {
  let c_primary = NonTrivialCircuit::new(num_cons);
  let c_secondary = TrivialCircuit::default();

  // Produce public parameters
  let pp = PublicParams::<E1>::setup(&c_primary, &c_secondary, &*S1::ck_floor(), &*S2::ck_floor())
    .unwrap();

  // Produce prover and verifier keys for CompressedSNARK
  let (pk, vk) = CompressedSNARK::<_, S1, S2>::setup(&pp).unwrap();

  // produce a recursive SNARK
  let num_steps = 3;
  let mut recursive_snark: RecursiveSNARK<E1> = RecursiveSNARK::new(
    &pp,
    &c_primary,
    &c_secondary,
    &[<E1 as Engine>::Scalar::from(2u64)],
    &[<E2 as Engine>::Scalar::from(2u64)],
  )
  .unwrap();

  for i in 0..num_steps {
    recursive_snark
      .prove_step(&pp, &c_primary, &c_secondary)
      .unwrap();

    // verify the recursive snark at each step of recursion
    recursive_snark
      .verify(
        &pp,
        i + 1,
        &[<E1 as Engine>::Scalar::from(2u64)],
        &[<E2 as Engine>::Scalar::from(2u64)],
      )
      .unwrap();
  }

  let bench_params = BenchParams {
    step_size: num_cons,
    date: env!("VERGEN_GIT_COMMIT_DATE"),
    sha: env!("VERGEN_GIT_SHA"),
  };

  // Bench time to produce a compressed SNARK
  group.bench_function(bench_params.bench_id("Prove"), |b| {
    b.iter(|| {
      CompressedSNARK::<_, S1, S2>::prove(
        black_box(&pp),
        black_box(&pk),
        black_box(&recursive_snark),
      )
      .unwrap();
    })
  });
  let compressed_snark = CompressedSNARK::<_, S1, S2>::prove(&pp, &pk, &recursive_snark).unwrap();

  // Benchmark the verification time
  group.bench_function(bench_params.bench_id("Verify"), |b| {
    b.iter(|| {
      black_box(&compressed_snark)
        .verify(
          black_box(&vk),
          black_box(num_steps),
          black_box(&[<E1 as Engine>::Scalar::from(2u64)]),
          black_box(&[<E2 as Engine>::Scalar::from(2u64)]),
        )
        .unwrap();
    })
  });
}

fn bench_compressed_snark(c: &mut Criterion) {
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in [
    NUM_CONS_VERIFIER_CIRCUIT_PRIMARY,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
  ]
  .iter()
  {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit - NUM_CONS_VERIFIER_CIRCUIT_PRIMARY;

    let mut group = c.benchmark_group("CompressedSNARK");
    group.sample_size(NUM_SAMPLES);
    group.noise_threshold(noise_threshold_env().unwrap_or(0.05));

    bench_compressed_snark_internal::<S1, S2>(&mut group, num_cons);

    group.finish();
  }
}

fn bench_compressed_snark_with_computational_commitments(c: &mut Criterion) {
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in [
    NUM_CONS_VERIFIER_CIRCUIT_PRIMARY,
    16384,
    32768,
    65536,
    131072,
    262144,
  ]
  .iter()
  {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit - NUM_CONS_VERIFIER_CIRCUIT_PRIMARY;

    let mut group = c.benchmark_group("CompressedSNARK-Commitments");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(NUM_SAMPLES);
    group.noise_threshold(noise_threshold_env().unwrap_or(0.05));

    bench_compressed_snark_internal::<SS1, SS2>(&mut group, num_cons);

    group.finish();
  }
}

// SNARKs without computation commitmnets
type BS1 = arecibo::spartan::batched::BatchedRelaxedR1CSSNARK<E1, EE1>;
type BS2 = arecibo::spartan::batched::BatchedRelaxedR1CSSNARK<E2, EE2>;
// SNARKs with computation commitmnets
type BSS1 = arecibo::spartan::batched_ppsnark::BatchedRelaxedR1CSSNARK<E1, EE1>;
type BSS2 = arecibo::spartan::batched_ppsnark::BatchedRelaxedR1CSSNARK<E2, EE2>;

fn bench_compressed_batched_snark(c: &mut Criterion) {
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in [
    NUM_CONS_VERIFIER_CIRCUIT_PRIMARY,
    16384,
    32768,
    65536,
    131072,
    262144,
  ]
  .iter()
  {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit - NUM_CONS_VERIFIER_CIRCUIT_PRIMARY;

    let mut group = c.benchmark_group("BatchedCompressedSNARK");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(NUM_SAMPLES);
    group.noise_threshold(noise_threshold_env().unwrap_or(0.05));

    bench_compressed_snark_internal::<BS1, BS2>(&mut group, num_cons);

    group.finish();
  }
}

fn bench_compressed_batched_snark_with_computational_commitments(c: &mut Criterion) {
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in [
    NUM_CONS_VERIFIER_CIRCUIT_PRIMARY,
    16384,
    32768,
    65536,
    131072,
    262144,
  ]
  .iter()
  {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit - NUM_CONS_VERIFIER_CIRCUIT_PRIMARY;

    let mut group = c.benchmark_group("BatchedCompressedSNARK-Commitments");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(NUM_SAMPLES);
    group.noise_threshold(noise_threshold_env().unwrap_or(0.05));

    bench_compressed_snark_internal::<BSS1, BSS2>(&mut group, num_cons);

    group.finish();
  }
}

#[derive(Clone, Debug, Default)]
struct NonTrivialCircuit<F: PrimeField> {
  num_cons: usize,
  _p: PhantomData<F>,
}

impl<F: PrimeField> NonTrivialCircuit<F> {
  pub fn new(num_cons: usize) -> Self {
    Self {
      num_cons,
      _p: PhantomData,
    }
  }
}
impl<F: PrimeField> StepCircuit<F> for NonTrivialCircuit<F> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    // Consider a an equation: `x^2 = y`, where `x` and `y` are respectively the input and output.
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..self.num_cons {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok(vec![y])
  }
}
