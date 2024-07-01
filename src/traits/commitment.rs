//! This module defines a collection of traits that define the behavior of a commitment engine
//! We require the commitment engine to provide a commitment to vectors with a single group element
use crate::provider::traits::DlogGroup;
use crate::{
  errors::NovaError,
  traits::{AbsorbInROTrait, Engine, TranscriptReprTrait},
};
use abomonation::Abomonation;
use group::prime::PrimeCurve;
use core::{
  fmt::Debug,
  ops::{Add, Mul, MulAssign, Sub},
};
use serde::{Deserialize, Serialize};

/// A helper trait for types implementing scalar multiplication.
pub trait ScalarMul<Rhs, Output = Self>: Mul<Rhs, Output = Output> + MulAssign<Rhs> {}

impl<T, Rhs, Output> ScalarMul<Rhs, Output> for T where T: Mul<Rhs, Output = Output> + MulAssign<Rhs>
{}

/// This trait defines the behavior of the commitment
pub trait CommitmentTrait<E: Engine>:
  Clone
  + Copy
  + Debug
  + Default
  + PartialEq
  + Eq
  + Send
  + Sync
  + TranscriptReprTrait<E::GE>
  + Serialize
  + for<'de> Deserialize<'de>
  + Abomonation
  + AbsorbInROTrait<E>
  + Add<Self, Output = Self>
  + Sub<Self, Output = Self>
  + ScalarMul<E::Scalar>
{
  /// Holds the type of the compressed commitment
  type CompressedCommitment: Clone
    + Debug
    + PartialEq
    + Eq
    + Send
    + Sync
    + TranscriptReprTrait<E::GE>
    + Serialize
    + for<'de> Deserialize<'de>;

  /// Compresses self into a compressed commitment
  fn compress(&self) -> Self::CompressedCommitment;

  /// Returns the coordinate representation of the commitment
  fn to_coordinates(&self) -> (E::Base, E::Base, bool);

  /// Decompresses a compressed commitment into a commitment
  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, NovaError>;

  /// Reinterpret as generator
  fn reinterpret_as_generator(&self) -> <<E as Engine>::GE as PrimeCurve>::Affine
  where
    E::GE: DlogGroup;
}

/// A trait that helps determine the length of a structure.
/// Note this does not impose any memory representation constraints on the structure.
pub trait Len {
  /// Returns the length of the structure.
  fn length(&self) -> usize;
}

/// A trait that ties different pieces of the commitment generation together
pub trait CommitmentEngineTrait<E: Engine>: Clone + Send + Sync {
  /// Holds the type of the commitment key
  /// The key should quantify its length in terms of group generators.
  type CommitmentKey: Len
    + Clone
    + PartialEq
    + Debug
    + Send
    + Sync
    + Serialize
    + for<'de> Deserialize<'de>
    + Abomonation;

  /// Holds the type of the commitment
  type Commitment: CommitmentTrait<E>;

  /// Samples a new commitment key of a specified size (power of 2)
  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey;

  /// Samples a new commitment key of a specified size
  fn setup_exact(label: &'static [u8], n: usize) -> Self::CommitmentKey;

  /// Samples a new commitment key (power of 2) but reuses the blinding generator of ck
  fn setup_with_blinding(
    label: &'static [u8],
    n: usize,
    h: &<<E as Engine>::GE as PrimeCurve>::Affine,
  ) -> Self::CommitmentKey
  where
    E::GE: DlogGroup;

  /// Samples a new commitment key of specific size but reuses the blinding generator of ck
  fn setup_exact_with_blinding(
    label: &'static [u8],
    n: usize,
    h: &<<E as Engine>::GE as PrimeCurve>::Affine,
  ) -> Self::CommitmentKey
  where
    E::GE: DlogGroup;

  /// Commits to the provided vector using the provided generators
  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar], r: &E::Scalar) -> Self::Commitment;

  /// Returns the generators of the commitment
  fn get_gens(
    ck: &Self::CommitmentKey,
  ) -> Vec<<<E as Engine>::GE as PrimeCurve>::Affine>
  where
    E::GE: DlogGroup;

  /// Returns the blinding generator of the commitment
  fn get_blinding_gen(
    ck: &Self::CommitmentKey,
  ) -> <<E as Engine>::GE as PrimeCurve>::Affine
  where
    E::GE: DlogGroup;

  /// Converts a commitment into generators (with no blinding generator)
  fn from_preprocessed(
    com: Vec<<<E as Engine>::GE as PrimeCurve>::Affine>,
  ) -> Self::CommitmentKey
  where
    E::GE: DlogGroup;
}
