//! This module defines a collection of traits that define the behavior of a `zkSNARK` for `RelaxedR1CS`
use crate::{
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::Engine,
  CommitmentKey,
};

use crate::gadgets::lookup::Lookup;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Public parameter creation takes a size hint. This size hint carries the particular requirements of
/// the final compressing SNARK the user expected to use with these public parameters, and the below
/// is a sensible default, which is to not require any more bases then the usual (maximum of the number of
/// variables and constraints of the involved R1CS circuit).
pub fn default_ck_hint<E: Engine>() -> Box<dyn for<'a> Fn(&'a R1CSShape<E>) -> usize> {
  // The default is to not put an additional floor on the size of the commitment key
  Box::new(|_shape: &R1CSShape<E>| 0)
}

/// A trait that defines the behavior of a `zkSNARK`
pub trait RelaxedR1CSSNARKTrait<E: Engine>:
  Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// A type that represents the prover's key
  type ProverKey: Send + Sync;

  /// A type that represents the verifier's key
  type VerifierKey: Send + Sync + Serialize;

  /// This associated function (not a method) provides a hint that offers
  /// a minimum sizing cue for the commitment key used by this SNARK
  /// implementation. The commitment key passed in setup should then
  /// be at least as large as this hint.
  fn ck_floor() -> Box<dyn for<'a> Fn(&'a R1CSShape<E>) -> usize> {
    // The default is to not put an additional floor on the size of the commitment key
    default_ck_hint()
  }

  /// Produces the keys for the prover and the verifier
  fn setup(
    ck: Arc<CommitmentKey<E>>,
    S: &R1CSShape<E>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError>;

  /// Produces a new SNARK for a relaxed R1CS
  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: &R1CSShape<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
  ) -> Result<Self, NovaError>;

  /// Verifies a SNARK for a relaxed R1CS
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<E>) -> Result<(), NovaError>;
}

/// A trait that defines the behavior of a `lookup SNARK`.
pub trait RelaxedR1CSLookupSNARKTrait<E: Engine>:
  Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// A type that represents the prover's key
  type ProverKey: Send + Sync;

  /// A type that represents the verifier's key
  type VerifierKey: Send + Sync + DigestHelperTrait<E>;

  /// This associated function (not a method) provides a hint that offers
  /// a minimum sizing cue for the commitment key used by this SNARK
  /// implementation. The commitment key passed in setup should then
  /// be at least as large as this hint.
  fn ck_floor() -> Box<dyn for<'a> Fn(&'a R1CSShape<E>) -> usize> {
    default_ck_hint()
  }

  /// Produces the keys for the prover and the verifier
  fn setup(
    ck: Arc<CommitmentKey<E>>,
    S: &R1CSShape<E>,
    initial_table: &Lookup<E::Scalar>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError>
  where
    <E as Engine>::Scalar: Ord;

  
  /// Produces a new SNARK for a relaxed R1CS
  ///
  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: &R1CSShape<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
    challenges: (E::Scalar, E::Scalar),
    RW_acc: E::Scalar,
    initial_table: Lookup<E::Scalar>,
    final_table: Lookup<E::Scalar>,
  ) -> Result<Self, NovaError>;

  /// Verifies a SNARK for a relaxed R1CS
  fn verify<E2: Engine>(
    &self,
    vk: &Self::VerifierKey,
    U: &RelaxedR1CSInstance<E>,
    lookup_intermediate_gamma: E::Scalar,
    RW_acc: E::Scalar,
    challenges: (E::Scalar, E::Scalar),
  ) -> Result<(), NovaError>
  where
    E: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E as Engine>::Scalar>;

}

/// A trait that defines the behavior of a `zkSNARK` to prove knowledge of satisfying witness to batches of relaxed R1CS instances.
pub trait BatchedRelaxedR1CSSNARKTrait<E: Engine>:
  Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// A type that represents the prover's key
  type ProverKey: Send + Sync;

  /// A type that represents the verifier's key
  type VerifierKey: Send + Sync + DigestHelperTrait<E>;

  /// This associated function (not a method) provides a hint that offers
  /// a minimum sizing cue for the commitment key used by this SNARK
  /// implementation. The commitment key passed in setup should then
  /// be at least as large as this hint.
  fn ck_floor() -> Box<dyn for<'a> Fn(&'a R1CSShape<E>) -> usize> {
    default_ck_hint()
  }

  /// Produces the keys for the prover and the verifier
  ///
  /// **Note:** This method should be cheap and should not copy most of the
  /// commitment key. Look at `CommitmentEngineTrait::setup` for generating SRS data.
  fn setup(
    ck: Arc<CommitmentKey<E>>,
    S: Vec<&R1CSShape<E>>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError>;

  /// Produces a new SNARK for a batch of relaxed R1CS
  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: Vec<&R1CSShape<E>>,
    U: &[RelaxedR1CSInstance<E>],
    W: &[RelaxedR1CSWitness<E>],
  ) -> Result<Self, NovaError>;

  /// Verifies a SNARK for a batch of relaxed R1CS
  fn verify(&self, vk: &Self::VerifierKey, U: &[RelaxedR1CSInstance<E>]) -> Result<(), NovaError>;
}

/// A helper trait that defines the behavior of a verifier key of `zkSNARK`
pub trait DigestHelperTrait<E: Engine> {
  /// Returns the digest of the verifier's key
  fn digest(&self) -> E::Scalar;
}
