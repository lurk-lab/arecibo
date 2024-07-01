//! This module implements the Nova traits for `bn256::Point`, `bn256::Scalar`, `grumpkin::Point`, `grumpkin::Scalar`.
use crate::{
  impl_traits,
  provider::{traits::DlogGroup, util::msm::cpu_best_msm},
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::{FromUniformBytes, PrimeField};
use group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup, GroupEncoding};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use grumpkin_msm::{bn256 as bn256_msm, grumpkin as grumpkin_msm};
// Remove this when https://github.com/zcash/pasta_curves/issues/41 resolves
use halo2curves::{bn256::G2Affine, CurveAffine, CurveExt};
use num_bigint::BigInt;
use num_traits::Num;
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

// Thus compile-time assertions checks important assumptions in the memory representation
// of group data that supports the use of Abomonation.
static_assertions::assert_eq_size!(G2Affine, [u64; 16]);

/// Re-exports that give access to the standard aliases used in the code base, for bn256
pub mod bn256 {
  pub use halo2curves::bn256::{
    Fq as Base, Fr as Scalar, G1Affine as Affine, G1Compressed as Compressed, G1 as Point,
  };
}

/// Re-exports that give access to the standard aliases used in the code base, for grumpkin
pub mod grumpkin {
  pub use halo2curves::grumpkin::{
    Fq as Base, Fr as Scalar, G1Affine as Affine, G1Compressed as Compressed, G1 as Point,
  };
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl_traits!(
  bn256,
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
  bn256_msm
);
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl_traits!(
  bn256,
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
);

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl_traits!(
  grumpkin,
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
  grumpkin_msm
);
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl_traits!(
  grumpkin,
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
);

#[cfg(test)]
mod tests {
  use ff::Field;
  use rand::thread_rng;

  use crate::provider::{
    bn256_grumpkin::{bn256, grumpkin},
    traits::DlogGroup,
    util::msm::cpu_best_msm,
  };

  #[test]
  fn test_bn256_msm_correctness() {
    let npoints = 1usize << 16;
    let points = bn256::Point::from_label(b"test", npoints);

    let mut rng = thread_rng();
    let scalars = (0..npoints)
      .map(|_| bn256::Scalar::random(&mut rng))
      .collect::<Vec<_>>();

    let cpu_msm = cpu_best_msm(&points, &scalars);
    let gpu_msm = bn256::Point::vartime_multiscalar_mul(&scalars, &points);

    assert_eq!(cpu_msm, gpu_msm);
  }

  #[test]
  fn test_grumpkin_msm_correctness() {
    let npoints = 1usize << 16;
    let points = grumpkin::Point::from_label(b"test", npoints);

    let mut rng = thread_rng();
    let scalars = (0..npoints)
      .map(|_| grumpkin::Scalar::random(&mut rng))
      .collect::<Vec<_>>();

    let cpu_msm = cpu_best_msm(&points, &scalars);
    let gpu_msm = grumpkin::Point::vartime_multiscalar_mul(&scalars, &points);

    assert_eq!(cpu_msm, gpu_msm);
  }
}
