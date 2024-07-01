//! This module implements the Nova traits for `pallas::Point`, `pallas::Scalar`, `vesta::Point`, `vesta::Scalar`.
use crate::{
  provider::{traits::DlogGroup, util::msm::cpu_best_msm},
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use derive_more::{From, Into};
use digest::{ExtendableOutput, Update};
use ff::{FromUniformBytes, PrimeField};
use group::{prime::PrimeCurveAffine, Curve};
use num_bigint::BigInt;
use num_traits::Num;
use pasta_curves::{
  self,
  arithmetic::{CurveAffine, CurveExt},
  group::{Group as AnotherGroup, GroupEncoding},
  pallas, vesta,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::Shake256;
use std::io::Read;

/// A wrapper for compressed group elements of pallas
#[derive(Clone, Copy, Debug, Eq, From, Into, PartialEq, Serialize, Deserialize)]
pub struct PallasCompressedElementWrapper([u8; 32]);

impl PallasCompressedElementWrapper {
  /// Wraps repr into the wrapper
  pub const fn new(repr: [u8; 32]) -> Self {
    Self(repr)
  }
}

/// A wrapper for compressed group elements of vesta
#[derive(Clone, Copy, Debug, Eq, From, Into, PartialEq, Serialize, Deserialize)]
pub struct VestaCompressedElementWrapper([u8; 32]);

impl VestaCompressedElementWrapper {
  /// Wraps repr into the wrapper
  pub const fn new(repr: [u8; 32]) -> Self {
    Self(repr)
  }
}

macro_rules! impl_traits {
  (
    $name:ident,
    $name_compressed:ident,
    $order_str:literal,
    $base_str:literal
  ) => {
    // These compile-time assertions check important assumptions in the memory representation
    // of group data that supports the use of Abomonation.
    static_assertions::assert_eq_size!($name::Affine, [u64; 8]);
    static_assertions::assert_eq_size!($name::Point, [u64; 12]);

    impl Group for $name::Point {
      type Base = $name::Base;
      type Scalar = $name::Scalar;

      fn group_params() -> (Self::Base, Self::Base, BigInt, BigInt) {
        let A = $name::Point::a();
        let B = $name::Point::b();
        let order = BigInt::from_str_radix($order_str, 16).unwrap();
        let base = BigInt::from_str_radix($base_str, 16).unwrap();

        (A, B, order, base)
      }
    }

    impl DlogGroup for $name::Point {
      type ScalarExt = $name::Scalar;
      type AffineExt = $name::Affine;
      type Compressed = $name_compressed;

      #[tracing::instrument(
        skip_all,
        level = "trace",
        name = "<_ as Group>::vartime_multiscalar_mul"
      )]
      fn vartime_multiscalar_mul(scalars: &[Self::ScalarExt], bases: &[Self::Affine]) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if scalars.len() >= 128 {
          grumpkin_msm::pasta::$name(bases, scalars)
        } else {
          cpu_best_msm(bases, scalars)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        cpu_best_msm(bases, scalars)
      }

      fn preprocessed(&self) -> Self::Affine {
        self.to_affine()
      }

      fn group(p: &Self::Affine) -> Self {
        $name::Point::from(*p)
      }

      fn compress(&self) -> Self::Compressed {
        $name_compressed::new(self.to_bytes())
      }

      fn from_label(label: &[u8], n: usize) -> Vec<Self::Affine> {
        let mut shake = Shake256::default();
        shake.update(label);
        let mut reader = shake.finalize_xof();
        let mut uniform_bytes_vec = Vec::new();
        for _ in 0..n {
          let mut uniform_bytes = [0u8; 32];
          reader.read_exact(&mut uniform_bytes).unwrap();
          uniform_bytes_vec.push(uniform_bytes);
        }
        let ck_proj: Vec<$name::Point> = (0..n)
          .into_par_iter()
          .map(|i| {
            let hash = $name::Point::hash_to_curve("from_uniform_bytes");
            hash(&uniform_bytes_vec[i])
          })
          .collect();

        let num_threads = rayon::current_num_threads();
        if ck_proj.len() > num_threads {
          let chunk = (ck_proj.len() as f64 / num_threads as f64).ceil() as usize;
          (0..num_threads)
            .into_par_iter()
            .flat_map(|i| {
              let start = i * chunk;
              let end = if i == num_threads - 1 {
                ck_proj.len()
              } else {
                core::cmp::min((i + 1) * chunk, ck_proj.len())
              };
              if end > start {
                let mut ck = vec![$name::Affine::identity(); end - start];
                <Self as Curve>::batch_normalize(&ck_proj[start..end], &mut ck);
                ck
              } else {
                vec![]
              }
            })
            .collect()
        } else {
          let mut ck = vec![$name::Affine::identity(); n];
          <Self as Curve>::batch_normalize(&ck_proj, &mut ck);
          ck
        }
      }

      fn zero() -> Self {
        $name::Point::identity()
      }

      fn gen() -> Self {
        $name::Point::generator()
      }

      fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
        let coordinates = self.to_affine().coordinates();
        if coordinates.is_some().unwrap_u8() == 1 {
          (*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
        } else {
          (Self::Base::zero(), Self::Base::zero(), true)
        }
      }
    }

    impl PrimeFieldExt for $name::Scalar {
      fn from_uniform(bytes: &[u8]) -> Self {
        let bytes_arr: [u8; 64] = bytes.try_into().unwrap();
        $name::Scalar::from_uniform_bytes(&bytes_arr)
      }
    }

    impl<G: DlogGroup> TranscriptReprTrait<G> for $name_compressed {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.0.to_vec()
      }
    }

    impl<G: Group> TranscriptReprTrait<G> for $name::Scalar {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.to_repr().to_vec()
      }
    }

    impl<G: DlogGroup> TranscriptReprTrait<G> for $name::Affine {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        let (x, y, is_infinity_byte) = {
          let coordinates = self.coordinates();
          if coordinates.is_some().unwrap_u8() == 1 {
            (
              *coordinates.unwrap().x(),
              *coordinates.unwrap().y(),
              u8::from(false),
            )
          } else {
            ($name::Base::zero(), $name::Base::zero(), u8::from(true))
          }
        };

        x.to_repr()
          .into_iter()
          .chain(y.to_repr().into_iter())
          .chain(std::iter::once(is_infinity_byte))
          .collect()
      }
    }
  };
}

impl_traits!(
  pallas,
  PallasCompressedElementWrapper,
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001"
);

impl_traits!(
  vesta,
  VestaCompressedElementWrapper,
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001"
);

#[cfg(test)]
mod tests {
  use ff::Field;
  use pasta_curves::{pallas, vesta};
  use rand::thread_rng;

  use crate::provider::{traits::DlogGroup, util::msm::cpu_best_msm};

  #[test]
  fn test_pallas_msm_correctness() {
    let npoints = 1usize << 16;
    let points = pallas::Point::from_label(b"test", npoints);

    let mut rng = thread_rng();
    let scalars = (0..npoints)
      .map(|_| pallas::Scalar::random(&mut rng))
      .collect::<Vec<_>>();

    let cpu_msm = cpu_best_msm(&points, &scalars);
    let gpu_msm = pallas::Point::vartime_multiscalar_mul(&scalars, &points);

    assert_eq!(cpu_msm, gpu_msm);
  }

  #[test]
  fn test_vesta_msm_correctness() {
    let npoints = 1usize << 16;
    let points = vesta::Point::from_label(b"test", npoints);

    let mut rng = thread_rng();
    let scalars = (0..npoints)
      .map(|_| vesta::Scalar::random(&mut rng))
      .collect::<Vec<_>>();

    let cpu_msm = cpu_best_msm(&points, &scalars);
    let gpu_msm = vesta::Point::vartime_multiscalar_mul(&scalars, &points);

    assert_eq!(cpu_msm, gpu_msm);
  }
}
