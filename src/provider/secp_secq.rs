//! This module implements the Nova traits for `secp::Point`, `secp::Scalar`, `secq::Point`, `secq::Scalar`.
use crate::{
  impl_traits,
  provider::{traits::DlogGroup, util::msm::cpu_best_msm},
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::{FromUniformBytes, PrimeField};
use group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup};
use group::GroupEncoding;
use num_bigint::BigInt;
use num_traits::Num;
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;
/// Re-exports that give access to the standard aliases used in the code base, for secp
pub mod secp256k1 {
  pub use halo2curves::secp256k1::{
    Fp as Base, Fq as Scalar, Secp256k1 as Point, Secp256k1Affine as Affine,
    Secp256k1Compressed as Compressed,
  };
}

/// Re-exports that give access to the standard aliases used in the code base, for secq
pub mod secq256k1 {
  pub use halo2curves::secq256k1::{
    Fp as Base, Fq as Scalar, Secq256k1 as Point, Secq256k1Affine as Affine,
    Secq256k1Compressed as Compressed,
  };
}

impl_traits!(
  secp256k1,
  "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
  "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
);

impl_traits!(
  secq256k1,
  "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
  "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"
);
