// Code is considered dead unless used in all benchmark targets
#![allow(dead_code)]

pub mod bench;
pub mod targets;

use anyhow::anyhow;
use arecibo::{
  supernova::{NonUniformCircuit, StepCircuit, TrivialTestCircuit},
  traits::{CurveCycleEquipped, Dual, Engine},
};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::PrimeField;
use halo2curves::bn256::Bn256;

pub type E1 = arecibo::provider::Bn256EngineKZG;
pub type E2 = arecibo::provider::GrumpkinEngine;
pub type EE1 = arecibo::provider::pcs::hyperkzg::EvaluationEngine<Bn256, E1>;
pub type EE2 = arecibo::provider::pcs::ipa_pc::EvaluationEngine<E2>;
// SNARKs without computation commitments
pub type S1 = arecibo::spartan::batched::BatchedRelaxedR1CSSNARK<E1, EE1>;
pub type S2 = arecibo::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;
// SNARKs with computation commitments
pub type SS1 = arecibo::spartan::batched_ppsnark::BatchedRelaxedR1CSSNARK<E1, EE1>;
pub type SS2 = arecibo::spartan::ppsnark::RelaxedR1CSSNARK<E2, EE2>;

// This should match the value in test_supernova_recursive_circuit_pasta
// Note `NUM_CONS_VERIFIER_CIRCUIT_PRIMARY` is different for Nova and Supernova
// TODO: This should also be a table matching the num_augmented_circuits in the below
pub const NUM_CONS_VERIFIER_CIRCUIT_PRIMARY: usize = 9844;
pub const NUM_SAMPLES: usize = 10;

#[derive(Copy, Clone)]
pub enum SnarkType {
  Recursive,
  Compressed,
}

// TODO: Move this up a level to `common/mod.rs`, then integrate with non-Supernova benches
pub fn num_cons() -> Vec<usize> {
  num_cons_env().unwrap_or_else(|_| {
    vec![
      NUM_CONS_VERIFIER_CIRCUIT_PRIMARY,
      16384,
      32768,
      65536,
      131072,
      262144,
      524288,
      1048576,
    ]
  })
}

fn num_cons_env() -> anyhow::Result<Vec<usize>> {
  std::env::var("BENCH_NUM_CONS")
    .map_err(|e| anyhow!("BENCH_NUM_CONS env var not set: {e}"))
    .and_then(|rc| {
      let vec: anyhow::Result<Vec<usize>> = rc
        .split(',')
        .map(|rc| {
          rc.parse::<usize>()
            .map_err(|e| anyhow!("Failed to parse constraint number: {e}"))
        })
        .collect();
      vec
    })
}

pub struct NonUniformBench<E1>
where
  E1: CurveCycleEquipped,
{
  num_circuits: usize,
  num_cons: usize,
  _p: PhantomData<E1>,
}

impl<E1> NonUniformBench<E1>
where
  E1: CurveCycleEquipped,
{
  fn new(num_circuits: usize, num_cons: usize) -> Self {
    Self {
      num_circuits,
      num_cons,
      _p: PhantomData,
    }
  }
}

impl<E1> NonUniformCircuit<E1> for NonUniformBench<E1>
where
  E1: CurveCycleEquipped,
{
  type C1 = NonTrivialTestCircuit<E1::Scalar>;
  type C2 = TrivialTestCircuit<<Dual<E1> as Engine>::Scalar>;

  fn num_circuits(&self) -> usize {
    self.num_circuits
  }

  fn primary_circuit(&self, circuit_index: usize) -> Self::C1 {
    assert!(
      circuit_index < self.num_circuits,
      "Circuit index out of bounds: asked for {circuit_index}, but there are only {} circuits.",
      self.num_circuits
    );

    NonTrivialTestCircuit::new(self.num_cons)
  }

  fn secondary_circuit(&self) -> Self::C2 {
    Default::default()
  }
}

#[derive(Clone, Debug, Default)]
pub struct NonTrivialTestCircuit<F: PrimeField> {
  num_cons: usize,
  _p: PhantomData<F>,
}

impl<F> NonTrivialTestCircuit<F>
where
  F: PrimeField,
{
  pub fn new(num_cons: usize) -> Self {
    Self {
      num_cons,
      _p: Default::default(),
    }
  }
}
impl<F> StepCircuit<F> for NonTrivialTestCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    // Consider a an equation: `x^{2 * num_cons} = y`, where `x` and `y` are respectively the input and output.
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..self.num_cons {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok((pc.cloned(), vec![y]))
  }
}
