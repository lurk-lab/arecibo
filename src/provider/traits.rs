use crate::traits::{Group, TranscriptReprTrait};
use abomonation::Abomonation;
use group::prime::PrimeCurveAffine;
use group::{prime::PrimeCurve, GroupEncoding};
use pasta_curves::pallas::Affine;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::Mul;

/// A trait that defines extensions to the Group trait
pub trait DlogGroup:
  Group<Scalar = <Self as DlogGroup>::ScalarExt>
  + Serialize
  + for<'de> Deserialize<'de>
  + PrimeCurve<Scalar = <Self as DlogGroup>::ScalarExt, Affine = <Self as DlogGroup>::AffineExt>
{
  type ScalarExt: Clone;
  type AffineExt: Clone
    + Debug
    + Eq
    + Serialize
    + for<'de> Deserialize<'de>
    + Sync
    + Send
    // technical bounds, should disappear when associated_type_bounds stabilizes
    + Mul<Self::ScalarExt, Output = Self>
    + PrimeCurveAffine<Curve = Self, Scalar = Self::ScalarExt>;
  type Compressed: Clone
    + Debug
    + Eq
    + From<<Self as GroupEncoding>::Repr>
    + Into<<Self as GroupEncoding>::Repr>
    + Serialize
    + for<'de> Deserialize<'de>
    + Sync
    + Send
    + TranscriptReprTrait<Self>;

  /// A method to compute a multiexponentation
  fn vartime_multiscalar_mul(scalars: &[Self::ScalarExt], bases: &[Self::AffineExt]) -> Self;

  /// Produce a vector of group elements using a static label
  fn from_label(label: &[u8], n: usize) -> Vec<Self::Affine>;

  /// Compresses the group element
  fn compress(&self) -> Self::Compressed;

  /// Produces a preprocessed element
  fn preprocessed(&self) -> Self::Affine;

  /// Returns a group element from a preprocessed group element
  fn group(p: &Self::Affine) -> Self;

  /// Returns an element that is the additive identity of the group
  fn zero() -> Self;

  /// Returns the generator of the group
  fn gen() -> Self;

  /// Returns the affine coordinates (x, y, infinity) for the point
  fn to_coordinates(&self) -> (<Self as Group>::Base, <Self as Group>::Base, bool);
}

/// This implementation behaves in ways specific to the halo2curves suite of curves in:
// - to_coordinates,
// - vartime_multiscalar_mul, where it does not call into accelerated implementations.
// A specific reimplementation exists for the pasta curves in their own module.
#[macro_export]
macro_rules! impl_traits {
  (
    $name:ident,
    $order_str:literal,
    $base_str:literal
  ) => {
    $crate::impl_traits!($name, $order_str, $base_str, cpu_best_msm);
  };
  (
    $name:ident,
    $order_str:literal,
    $base_str:literal,
    $large_msm_method: ident
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
      // note: for halo2curves implementations, $name::Compressed == <$name::Point as GroupEncoding>::Repr
      // so the blanket impl<T> From<T> for T and impl<T> Into<T> apply.
      type Compressed = $name::Compressed;

      fn vartime_multiscalar_mul(scalars: &[Self::ScalarExt], bases: &[Self::AffineExt]) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if scalars.len() >= 128 {
          $large_msm_method(bases, scalars)
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
        self.to_bytes()
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
        let gens_proj: Vec<$name::Point> = (0..n)
          .into_par_iter()
          .map(|i| {
            let hash = $name::Point::hash_to_curve("from_uniform_bytes");
            hash(&uniform_bytes_vec[i])
          })
          .collect();

        let num_threads = rayon::current_num_threads();
        if gens_proj.len() > num_threads {
          let chunk = (gens_proj.len() as f64 / num_threads as f64).ceil() as usize;
          (0..num_threads)
            .into_par_iter()
            .flat_map(|i| {
              let start = i * chunk;
              let end = if i == num_threads - 1 {
                gens_proj.len()
              } else {
                core::cmp::min((i + 1) * chunk, gens_proj.len())
              };
              if end > start {
                let mut gens = vec![$name::Affine::identity(); end - start];
                <Self as Curve>::batch_normalize(&gens_proj[start..end], &mut gens);
                gens
              } else {
                vec![]
              }
            })
            .collect()
        } else {
          let mut gens = vec![$name::Affine::identity(); n];
          <Self as Curve>::batch_normalize(&gens_proj, &mut gens);
          gens
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
        if coordinates.is_some().unwrap_u8() == 1 && ($name::Point::identity() != *self) {
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

    impl<G: DlogGroup> TranscriptReprTrait<G> for $name::Compressed {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.as_ref().to_vec()
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
          if coordinates.is_some().unwrap_u8() == 1 && ($name::Affine::identity() != *self) {
            let c = coordinates.unwrap();
            (*c.x(), *c.y(), u8::from(false))
          } else {
            ($name::Base::zero(), $name::Base::zero(), u8::from(false))
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
