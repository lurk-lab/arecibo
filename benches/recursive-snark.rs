#![allow(non_snake_case)]
use arecibo::{
  provider::{PallasEngine, VestaEngine},
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    snark::default_ck_hint,
    Engine,
  },
  PublicParams, RecursiveSNARK, StepCounterType,
};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use criterion::*;
use ff::PrimeField;
use std::time::Duration;

mod common;
use common::{noise_threshold_env, BenchParams};

type E1 = PallasEngine;
type E2 = VestaEngine;

// To run these benchmarks, first download `criterion` with `cargo install cargo-criterion`.
// Then `cargo criterion --bench recursive-snark`. The results are located in `target/criterion/data/<name-of-benchmark>`.
// For flamegraphs, run `cargo criterion --bench recursive-snark --features flamegraph -- --profile-time <secs>`.
// The results are located in `target/criterion/profile/<name-of-benchmark>`.
cfg_if::cfg_if! {
  if #[cfg(feature = "flamegraph")] {
    criterion_group! {
      name = recursive_snark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
      targets = bench_recursive_snark
    }
  } else {
    criterion_group! {
      name = recursive_snark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000));
      targets = bench_recursive_snark
    }
  }
}

criterion_main!(recursive_snark);

// This should match the value for the primary in test_recursive_circuit_pasta
const NUM_CONS_VERIFIER_CIRCUIT_PRIMARY: usize = 9825;
const NUM_SAMPLES: usize = 10;

fn bench_recursive_snark(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("RecursiveSNARK");
    group.sample_size(NUM_SAMPLES);
    group.noise_threshold(noise_threshold_env().unwrap_or(0.05));

    let c_primary = NonTrivialCircuit::new(num_cons);
    let c_secondary = TrivialCircuit::default();

    // Produce public parameters
    let pp = PublicParams::<E1>::setup(
      &c_primary,
      &c_secondary,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    // Bench time to produce a recursive SNARK;
    // we execute a certain number of warm-up steps since executing
    // the first step is cheaper than other steps owing to the presence of
    // a lot of zeros in the satisfying assignment
    let num_warmup_steps = 10;
    let mut recursive_snark: RecursiveSNARK<E1> = RecursiveSNARK::new(
      &pp,
      &c_primary,
      &c_secondary,
      &[<E1 as Engine>::Scalar::from(2u64)],
      &[<E2 as Engine>::Scalar::from(2u64)],
    )
    .unwrap();

    for i in 0..num_warmup_steps {
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

    group.bench_function(bench_params.bench_id("Prove"), |b| {
      b.iter(|| {
        // produce a recursive SNARK for a step of the recursion
        black_box(&mut recursive_snark.clone())
          .prove_step(
            black_box(&pp),
            black_box(&c_primary),
            black_box(&c_secondary),
          )
          .unwrap();
      })
    });

    // Benchmark the verification time
    group.bench_function(bench_params.bench_id("Verify"), |b| {
      b.iter(|| {
        black_box(&recursive_snark)
          .verify(
            black_box(&pp),
            black_box(num_warmup_steps),
            black_box(&[<E1 as Engine>::Scalar::from(2u64)]),
            black_box(&[<E2 as Engine>::Scalar::from(2u64)]),
          )
          .unwrap();
      });
    });
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
   
  fn get_counter_type(&self) -> StepCounterType {
    StepCounterType::Incremental
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
