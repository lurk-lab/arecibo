//! This module defines a collection of traits that define the behavior of a polynomial evaluation engine
//! A vector of size N is treated as a multilinear polynomial in \log{N} variables,
//! and a commitment provided by the commitment engine is treated as a multilinear polynomial commitment
use std::sync::Arc;

use crate::{
  errors::NovaError,
  traits::{commitment::CommitmentEngineTrait, Engine},
  CommitmentKey, CommitmentTrait,
};
use serde::{Deserialize, Serialize};

/// A trait that returns commitment of an evaluation argument
pub trait GetEvalCommitmentsTrait<E: Engine> {
  /// Returns the commitment at index
  fn get_eval_commitment(
    &self,
    index: usize,
  ) -> <<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as CommitmentTrait<E>>::CompressedCommitment;
}

/// A trait that ties different pieces of the commitment evaluation together
pub trait EvaluationEngineTrait<E: Engine + serde::Serialize>: Clone + Send + Sync {
  /// A type that holds the prover key
  type ProverKey: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that holds the verifier key
  type VerifierKey: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that holds the evaluation argument
  type EvaluationArgument: Clone
  + Send
  + Sync
  + Serialize
  + for<'de> Deserialize<'de>
  + GetEvalCommitmentsTrait<E>;

  /// A method to perform any additional setup needed to produce proofs of evaluations
  ///
  /// **Note:** This method should be cheap and should not copy most of the
  /// commitment key. Look at `CommitmentEngineTrait::setup` for generating SRS data.
  fn setup(
    ck: Arc<<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey>,
  ) -> (Self::ProverKey, Self::VerifierKey);

  // /// A method to prove the evaluation of a multilinear polynomial
  // fn prove(
  //   ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  //   pk: &Self::ProverKey,
  //   transcript: &mut E::TE,
  //   comm: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
  //   poly: &[E::Scalar],
  //   point: &[E::Scalar],
  //   eval: &E::Scalar,
  // ) -> Result<Self::EvaluationArgument, NovaError>;

  // /// A method to verify the purported evaluation of a multilinear polynomials
  // fn verify(
  //   vk: &Self::VerifierKey,
  //   transcript: &mut E::TE,
  //   comm: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
  //   point: &[E::Scalar],
  //   eval: &E::Scalar,
  //   arg: &Self::EvaluationArgument,
  // ) -> Result<(), NovaError>;


  /// A method to prove evaluations of a batch of polynomials
  #[allow(clippy::too_many_arguments)]
  fn prove_batch(
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
    pk: &Self::ProverKey,
    transcript: &mut E::TE,
    comm: &[<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment],
    polys: &[Vec<E::Scalar>],
    blinds_polys: &[E::Scalar],
    points: &[Vec<E::Scalar>],
    evals: &[E::Scalar],
    blinds_evals: &[E::Scalar],
    comm_evals: &[<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as CommitmentTrait<E>>::CompressedCommitment],
  ) -> Result<Self::EvaluationArgument, NovaError>;

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify_batch(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &[<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment],
    points: &[Vec<E::Scalar>],
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError>;

  /// Get single generator from pk
  fn get_scalar_gen_pk(pk: Self::ProverKey) -> CommitmentKey<E>;

  /// Get single generator from vk
  fn get_scalar_gen_vk(vk: Self::VerifierKey) -> CommitmentKey<E>;

  /// Get vector of generators from vk
  fn get_vector_gen_vk(vk: Self::VerifierKey) -> Arc<CommitmentKey<E>>;
}
