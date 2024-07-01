//! This module defines errors returned by the library.
use core::fmt::Debug;
use thiserror::Error;

/// Errors returned by Nova
#[derive(Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum NovaError {
  /// returned if the supplied row or col in (row,col,val) tuple is out of range
  #[error("InvalidIndex")]
  InvalidIndex,
  /// returned if the step circuit calls inputize or alloc_io in its synthesize method
  /// instead of passing output with the return value
  #[error("InvalidStepCircuitIO")]
  InvalidStepCircuitIO,
  /// returned if the supplied input is not of the right length
  #[error("InvalidInputLength")]
  InvalidInputLength,
  /// returned if the supplied witness is not of the right length
  #[error("InvalidWitnessLength")]
  InvalidWitnessLength,
  /// returned if the supplied witness is not a satisfying witness to a given shape and instance
  #[error("UnSat")]
  UnSat,
  /// returned if the supplied witness is not a satisfying witness to a given shape and instance, with error constraint index
  #[error("UnSatIndex")]
  UnSatIndex(usize),
  /// returned if the counter types for the primary and secondary circuit are not the same
  #[error("MismatchedCounterType")]
  MismatchedCounterType,
  /// returned when the supplied compressed commitment cannot be decompressed
  #[error("DecompressionError")]
  DecompressionError,
  /// returned if proof verification fails
  #[error("ProofVerifyError")]
  ProofVerifyError,
  /// returned if the provided commitment key is not of sufficient length
  #[error("InvalidCommitmentKeyLength")]
  InvalidCommitmentKeyLength,
  /// returned if the provided number of steps is zero
  #[error("InvalidNumSteps")]
  InvalidNumSteps,
  /// returned if there is an error in the proof/verification of a PCS
  #[error("PCSError")]
  PCSError(#[from] PCSError),
  /// returned when an invalid knowledge proof is returned
  #[error("InvalidZkKnowledgeProof")]
  InvalidZkKnowledgeProof,
  /// returned when an invalid equality proof is returned
  #[error("InvalidZkEqualityProof")]
  InvalidZkEqualityProof,
  /// returned when an invalid product proof is returned
  #[error("InvalidZkProductProof")]
  InvalidZkProductProof,
  /// returned when an invalid dot product proof (schnorr) is provided
  #[error("InvalidZkDotProductProof")]
  InvalidZkDotProductProof,
  /// returned when an invalid sum-check proof is provided
  #[error("InvalidSumcheckProof")]
  InvalidSumcheckProof,
  /// InvalidIPA
  #[error("InvalidIPA")]
  InvalidIPA,
  /// returned when the initial input to an incremental computation differs from a previously declared arity
  #[error("InvalidInitialInputLength")]
  InvalidInitialInputLength,
  /// returned when the step execution produces an output whose length differs from a previously declared arity
  #[error("InvalidStepOutputLength")]
  InvalidStepOutputLength,
  /// returned when the transcript engine encounters an overflow of the round number
  #[error("InternalTranscriptError")]
  InternalTranscriptError,
  /// returned when the multiset check fails
  #[error("InvalidMultisetProof")]
  InvalidMultisetProof,
  /// returned when the product proof check fails
  #[error("InvalidProductProof")]
  InvalidProductProof,
  /// returned when the consistency with public IO and assignment used fails
  #[error("IncorrectWitness")]
  IncorrectWitness,
  /// return when error during synthesis
  #[error("SynthesisError: {0}")]
  SynthesisError(String),
  /// returned when there is an error creating a digest
  #[error("DigestError")]
  DigestError,
  /// returned when the prover cannot prove the provided statement due to completeness error
  #[error("InternalError")]
  InternalError,
}

/// Errors specific to the Polynomial commitment scheme
#[derive(Debug, Eq, PartialEq, Error)]
pub enum PCSError {
  /// returned when an invalid PCS evaluation argument is provided
  #[error("InvalidPCS")]
  InvalidPCS,
  /// returned when there is a Zeromorph error
  #[error("ZMError")]
  ZMError,
  /// returned when a length check fails in a PCS
  #[error("LengthError")]
  LengthError,
}

impl From<bellpepper_core::SynthesisError> for NovaError {
  fn from(err: bellpepper_core::SynthesisError) -> Self {
    Self::SynthesisError(err.to_string())
  }
}
