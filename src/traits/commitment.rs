//! This module defines a collection of traits that define the behavior of a commitment engine
//! We require the commitment engine to provide a commitment to vectors with a single group element
use crate::{
  errors::NovaError,
  traits::{AbsorbInROTrait, Group, TranscriptReprTrait},
};
use abomonation::Abomonation;
use core::{
  fmt::Debug,
  ops::{Add, AddAssign},
};
use serde::{Deserialize, Serialize};
use std::{
  cmp::min,
  io::{self, Write},
};

use super::ScalarMul;

/// Defines basic operations on commitments
pub trait CommitmentOps<Rhs = Self, Output = Self>:
  Add<Rhs, Output = Output> + AddAssign<Rhs>
{
}

impl<T, Rhs, Output> CommitmentOps<Rhs, Output> for T where
  T: Add<Rhs, Output = Output> + AddAssign<Rhs>
{
}

/// A helper trait for references with a commitment operation
pub trait CommitmentOpsOwned<Rhs = Self, Output = Self>:
  for<'r> CommitmentOps<&'r Rhs, Output>
{
}
impl<T, Rhs, Output> CommitmentOpsOwned<Rhs, Output> for T where
  T: for<'r> CommitmentOps<&'r Rhs, Output>
{
}

/// This trait defines the behavior of the commitment
pub trait CommitmentTrait<G: Group>:
  Clone
  + Copy
  + Debug
  + Default
  + PartialEq
  + Eq
  + Send
  + Sync
  + TranscriptReprTrait<G>
  + Serialize
  + for<'de> Deserialize<'de>
  + Abomonation
  + AbsorbInROTrait<G>
  + CommitmentOps
  + CommitmentOpsOwned
  + ScalarMul<G::Scalar>
{
  /// Holds the type of the compressed commitment
  type CompressedCommitment: Clone
    + Debug
    + PartialEq
    + Eq
    + Send
    + Sync
    + TranscriptReprTrait<G>
    + Serialize
    + for<'de> Deserialize<'de>;

  /// Compresses self into a compressed commitment
  fn compress(&self) -> Self::CompressedCommitment;

  /// Returns the coordinate representation of the commitment
  fn to_coordinates(&self) -> (G::Base, G::Base, bool);

  /// Decompresses a compressed commitment into a commitment
  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, NovaError>;
}

/// A trait that defines the behavior of commitment keys
pub trait CommitmentKeyTrait {
  /// Returns the length of the commitment key.
  fn length(&self) -> usize;

  /// Checks if self is a prefix of the given key
  fn is_prefix_of(&self, other: &Self) -> bool;

  /// Encode into a writer
  fn encode<W: Write>(&self, write: &mut W) -> io::Result<()>;

  /// Decode from a reader
  fn decode(bytes: &mut [u8], n: usize) -> io::Result<Self>
  where
    Self: Sized;
}

/// A trait that ties different pieces of the commitment generation together
pub trait CommitmentEngineTrait<G: Group>: Clone + Send + Sync {
  /// Holds the type of the commitment key
  /// The key should quantify its length in terms of group generators.
  type CommitmentKey: CommitmentKeyTrait
    + Clone
    + PartialEq
    + Debug
    + Send
    + Sync
    + Serialize
    + for<'de> Deserialize<'de>
    + Abomonation;

  /// Holds the type of the commitment
  type Commitment: CommitmentTrait<G>;

  /// Samples a new commitment key of a specified size
  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey;

  /// Internal implementation of [CommitmentEngineTrait::extend]
  ///
  /// **Do not call this directly**. This does not check the consistency of the given label.
  fn extend_aux(ck: &mut Self::CommitmentKey, label: &'static [u8], n: usize);

  /// Extends a commitment key by a given size.
  ///
  /// This is an optimization to avoid regenerating an entire key again when
  /// we just need to extend it by a few elements. To verify that the same label
  /// was used between the given one and the original one used to generate the key,
  /// we check [crate::constants::CK_CHECKING_THRESHOLD] elements and assert equality.
  fn extend(ck: &mut Self::CommitmentKey, label: &'static [u8], n: usize) -> Result<(), NovaError>
  where
    Self: Sized,
  {
    let threshold = min(ck.length(), crate::constants::CK_CHECKING_THRESHOLD);
    let test_ck = Self::setup(label, threshold);

    if !test_ck.is_prefix_of(ck) {
      return Err(NovaError::InvalidCommitmentKeyLabel);
    }

    Self::extend_aux(ck, label, n);

    Ok(())
  }

  /// Commits to the provided vector using the provided generators
  fn commit(ck: &Self::CommitmentKey, v: &[G::Scalar]) -> Self::Commitment;
}
