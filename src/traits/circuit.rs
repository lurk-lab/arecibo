//! This module defines traits that a step function must implement
use crate::StepCounterType;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::PrimeField;

/// A helper trait for a step of the incremental computation (i.e., circuit for F)
pub trait StepCircuit<F: PrimeField>: Send + Sync + Clone {
  /// Return the number of inputs or outputs of each step
  /// (this method is called only at circuit synthesis time)
  /// `synthesize` and `output` methods are expected to take as
  /// input a vector of size equal to arity and output a vector of size equal to arity
  fn arity(&self) -> usize;

  /// Returns the type of the counter to be used with this circuit
  fn get_counter_type(&self) -> StepCounterType;

  /// Sythesize the circuit for a computation step and return variable
  /// that corresponds to the output of the step `z_{i+1}`
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>;
}

/// A trivial step circuit that simply returns the input
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrivialCircuit<F> {
  _p: PhantomData<F>,
  counter_type: StepCounterType,
}

impl<F> TrivialCircuit<F>
where
  F: PrimeField,
{
  /// Creates a new trivial test circuit with a particular step counter type
  pub fn new(counter_type: StepCounterType) -> TrivialCircuit<F> {
    Self {
      _p: PhantomData,
      counter_type,
    }
  }
}

impl<F> Default for TrivialCircuit<F>
where
  F: PrimeField,
{
  /// Creates a new trivial test circuit with step counter type Incremental
  fn default() -> TrivialCircuit<F> {
    Self {
      _p: PhantomData,
      counter_type: StepCounterType::Incremental,
    }
  }
}

impl<F> StepCircuit<F> for TrivialCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn get_counter_type(&self) -> StepCounterType {
    self.counter_type
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    Ok(z.to_vec())
  }
}
