//! This module implements `RelaxedR1CSSNARKTrait` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! We provide two implementations, one in snark.rs (which does not use any preprocessing)
//! and another in ppsnark.rs (which uses preprocessing to keep the verifier's state small if the PCS provides a succinct verifier)
//! We also provide direct.rs that allows proving a step circuit directly with either of the two SNARKs.
//!
//! In polynomial.rs we also provide foundational types and functions for manipulating multilinear polynomials.

pub mod batched;
pub mod batched_ppsnark;
#[macro_use]
mod macros;
pub(crate) mod math;
pub mod polys;
pub mod ppsnark;
pub mod lookup_ppsnark;
pub mod snark;
mod sumcheck;

use crate::{
  r1cs::{R1CSShape, SparseMatrix},
  traits::Engine,
  Commitment,
};
use ff::Field;
use itertools::Itertools as _;
use rayon::{iter::IntoParallelRefIterator, prelude::*};
use rayon_scan::ScanParallelIterator as _;
use ref_cast::RefCast;

// Creates a vector of the first `n` powers of `s`.
/// Creates a vector of the first `n` powers of `s`.
pub fn powers<F: Field>(s: &F, n: usize) -> Vec<F> {
  assert!(n >= 1);
  let mut v = vec![*s; n];
  v[0] = F::ONE;
  v.into_par_iter().scan(|a, b| *a * *b, F::ONE).collect()
}

/// A type that holds a witness to a polynomial evaluation instance
#[repr(transparent)]
#[derive(Debug, RefCast)]
struct PolyEvalWitness<E: Engine> {
  p: Vec<E::Scalar>, // polynomial
}

impl<E: Engine> PolyEvalWitness<E> {
  /// Given [Pᵢ] and s, compute P = ∑ᵢ sⁱ⋅Pᵢ
  ///
  /// # Details
  ///
  /// We allow the input polynomials to have different sizes, and interpret smaller ones as
  /// being padded with 0 to the maximum size of all polynomials.
  fn batch_diff_size(W: &[&Self], s: E::Scalar) -> Self {
    let powers = powers(&s, W.len());

    let size_max = W.iter().map(|w| w.p.len()).max().unwrap();
    let p_vec = W.par_iter().map(|w| &w.p);
    // Scale the input polynomials by the power of s
    let p = zip_with!((p_vec, powers.par_iter()), |v, weight| {
      // compute the weighted sum for each vector
      v.iter()
        .map(|&x| {
          if *weight != E::Scalar::ONE {
            x * *weight
          } else {
            x
          }
        })
        .collect::<Vec<_>>()
    })
    .reduce(
      || vec![E::Scalar::ZERO; size_max],
      |left, right| {
        // Sum into the largest polynomial
        let (mut big, small) = if left.len() > right.len() {
          (left, right)
        } else {
          (right, left)
        };

        #[allow(clippy::disallowed_methods)]
        big
          .par_iter_mut()
          .zip(small.par_iter())
          .for_each(|(b, s)| *b += s);

        big
      },
    );

    Self { p }
  }

  /// Given a set of polynomials \[Pᵢ\] and a scalar `s`, this method computes the weighted sum
  /// of the polynomials, where each polynomial Pᵢ is scaled by sⁱ.
  ///
  /// # Panics
  ///
  /// This method panics if the polynomials in `p_vec` are not all of the same length.
  fn batch(p_vec: &[&Vec<E::Scalar>], s: &E::Scalar) -> Self {
    p_vec
      .iter()
      .skip(1)
      .for_each(|p| assert_eq!(p.len(), p_vec[0].len()));
    let instances = p_vec.iter().map(|p| Self::ref_cast(p)).collect::<Vec<_>>();
    Self::batch_diff_size(&instances, *s)
  }
}

/// A type that holds a polynomial evaluation instance
#[derive(Debug)]
struct PolyEvalInstance<E: Engine> {
  c: Commitment<E>,  // commitment to the polynomial
  x: Vec<E::Scalar>, // evaluation point
  e: E::Scalar,      // claimed evaluation
}

impl<E: Engine> PolyEvalInstance<E> {
  fn batch_diff_size(
    c_vec: &[Commitment<E>],
    e_vec: &[E::Scalar],
    num_vars: &[usize],
    x: Vec<E::Scalar>,
    s: E::Scalar,
  ) -> Self {
    let num_instances = num_vars.len();
    assert_eq!(c_vec.len(), num_instances);
    assert_eq!(e_vec.len(), num_instances);

    let num_vars_max = x.len();
    let powers: Vec<E::Scalar> = powers(&s, num_instances);
    // Rescale evaluations by the first Lagrange polynomial,
    // so that we can check its evaluation against x
    let evals_scaled = zip_with!(iter, (e_vec, num_vars), |eval, num_rounds| {
      // x_lo = [ x[0]   , ..., x[n-nᵢ-1] ]
      // x_hi = [ x[n-nᵢ], ..., x[n]      ]
      let (r_lo, _r_hi) = x.split_at(num_vars_max - num_rounds);
      // Compute L₀(x_lo)
      let lagrange_eval = r_lo
        .iter()
        .map(|r| E::Scalar::ONE - r)
        .product::<E::Scalar>();

      // vᵢ = L₀(x_lo)⋅Pᵢ(x_hi)
      lagrange_eval * eval
    });

    // C = ∑ᵢ γⁱ⋅Cᵢ
    let comm_joint = zip_with!(iter, (c_vec, powers), |c, g_i| *c * *g_i)
      .fold(Commitment::<E>::default(), |acc, item| acc + item);

    // v = ∑ᵢ γⁱ⋅vᵢ
    let eval_joint = zip_with!((evals_scaled, powers.iter()), |e, g_i| e * g_i).sum();

    Self {
      c: comm_joint,
      x,
      e: eval_joint,
    }
  }

  fn batch(c_vec: &[Commitment<E>], x: Vec<E::Scalar>, e_vec: &[E::Scalar], s: &E::Scalar) -> Self {
    let sizes = vec![x.len(); e_vec.len()];
    Self::batch_diff_size(c_vec, e_vec, &sizes, x, *s)
  }
}

/// Binds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials
fn compute_eval_table_sparse<E: Engine>(
  S: &R1CSShape<E>,
  rx: &[E::Scalar],
) -> (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>) {
  assert_eq!(rx.len(), S.num_cons);

  let inner = |M: &SparseMatrix<E::Scalar>, M_evals: &mut Vec<E::Scalar>| {
    for (row_idx, row) in M.iter_rows().enumerate() {
      for (val, col_idx) in M.get_row(row) {
        // TODO(@winston-h-zhang): Parallelize? Will need more complicated locking
        M_evals[*col_idx] += rx[row_idx] * val;
      }
    }
  };

  let (A_evals, (B_evals, C_evals)) = rayon::join(
    || {
      let mut A_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars];
      inner(&S.A, &mut A_evals);
      A_evals
    },
    || {
      rayon::join(
        || {
          let mut B_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars];
          inner(&S.B, &mut B_evals);
          B_evals
        },
        || {
          let mut C_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars];
          inner(&S.C, &mut C_evals);
          C_evals
        },
      )
    },
  );

  (A_evals, B_evals, C_evals)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
  use super::*;
  use crate::provider::PallasEngine;
  use crate::r1cs::util::{FWrap, GWrap};
  use pasta_curves::pallas::Point as PallasPoint;
  use pasta_curves::Fq as Scalar;
  use proptest::collection::vec;
  use proptest::prelude::*;

  impl<E: Engine> PolyEvalWitness<E> {
    fn alt_batch(p_vec: &[&Vec<E::Scalar>], s: &E::Scalar) -> Self {
      p_vec
        .iter()
        .skip(1)
        .for_each(|p| assert_eq!(p.len(), p_vec[0].len()));

      let powers_of_s = powers(s, p_vec.len());

      let p = zip_with!(par_iter, (p_vec, powers_of_s), |v, weight| {
        // compute the weighted sum for each vector
        v.iter().map(|&x| x * *weight).collect::<Vec<E::Scalar>>()
      })
      .reduce(
        || vec![E::Scalar::ZERO; p_vec[0].len()],
        |acc, v| {
          // perform vector addition to combine the weighted vectors
          acc.into_iter().zip_eq(v).map(|(x, y)| x + y).collect()
        },
      );

      Self { p }
    }
  }

  impl<E: Engine> PolyEvalInstance<E> {
    fn alt_batch(
      c_vec: &[Commitment<E>],
      x: Vec<E::Scalar>,
      e_vec: &[E::Scalar],
      s: &E::Scalar,
    ) -> Self {
      let num_instances = c_vec.len();
      assert_eq!(e_vec.len(), num_instances);

      let powers_of_s = powers(s, num_instances);
      // Weighted sum of evaluations
      let e = zip_with!(par_iter, (e_vec, powers_of_s), |e, p| *e * p).sum();
      // Weighted sum of commitments
      let c = zip_with!(par_iter, (c_vec, powers_of_s), |c, p| *c * *p)
        .reduce(Commitment::<E>::default, |acc, item| acc + item);

      Self { c, x, e }
    }
  }

  proptest! {
      #[test]
      fn test_pe_witness_batch_diff_size_batch(
        s in any::<FWrap<Scalar>>(),
        vecs in (50usize..100).prop_flat_map(|size| vec(
            vec(any::<FWrap<Scalar>>().prop_map(|f| f.0), size..=size), // even-sized vec
            1..5))
      )
      {
        // when the vectors are the same size, batch_diff_size and batch agree
        let res = PolyEvalWitness::<PallasEngine>::alt_batch(&vecs.iter().by_ref().collect::<Vec<_>>(), &s.0);
        let witnesses = vecs.iter().map(PolyEvalWitness::ref_cast).collect::<Vec<_>>();
        let res2 = PolyEvalWitness::<PallasEngine>::batch_diff_size(&witnesses, s.0);

        prop_assert_eq!(res.p, res2.p);
      }

      #[test]
      fn test_pe_witness_batch_diff_size_pad_batch(
        s in any::<FWrap<Scalar>>(),
        vecs in (50usize..100).prop_flat_map(|size| vec(
            vec(any::<FWrap<Scalar>>().prop_map(|f| f.0), size-10..=size), // even-sized vec
            1..10))
      )
      {
        let size = vecs.iter().map(|v| v.len()).max().unwrap_or(0);
        // when the vectors are not the same size, batch agrees with the padded version of the input
        let padded_vecs = vecs.iter().cloned().map(|mut v| {v.resize(size, Scalar::ZERO); v}).collect::<Vec<_>>();
        let res = PolyEvalWitness::<PallasEngine>::alt_batch(&padded_vecs.iter().by_ref().collect::<Vec<_>>(), &s.0);
        let witnesses = vecs.iter().map(PolyEvalWitness::ref_cast).collect::<Vec<_>>();
        let res2 = PolyEvalWitness::<PallasEngine>::batch_diff_size(&witnesses, s.0);

        prop_assert_eq!(res.p, res2.p);
      }

      #[test]
      fn test_pe_instance_batch_diff_size_batch(
        s in any::<FWrap<Scalar>>(),
        vecs_tuple in (50usize..100).prop_flat_map(|size|
            (vec(any::<GWrap<PallasPoint>>().prop_map(|f| f.0), size..=size),
             vec(any::<FWrap<Scalar>>().prop_map(|f| f.0), size..=size),
             vec(any::<FWrap<Scalar>>().prop_map(|f| f.0), size..=size)
            ), // even-sized vecs
          )
      )
      {
        let (c_vec, e_vec, x_vec) = vecs_tuple;
        let c_vecs = c_vec.into_iter().map(|c| Commitment::<PallasEngine>{ comm: c }).collect::<Vec<_>>();
        // when poly evals are all for the max # of variables, batch_diff_size and batch agree
        let res = PolyEvalInstance::<PallasEngine>::alt_batch(
          &c_vecs,
          x_vec.clone(),
          &e_vec,
          &s.0);

        let sizes = vec![x_vec.len(); x_vec.len()];
        let res2 = PolyEvalInstance::<PallasEngine>::batch_diff_size(&c_vecs, &e_vec, &sizes, x_vec.clone(), s.0);

        prop_assert_eq!(res.c, res2.c);
        prop_assert_eq!(res.x, res2.x);
        prop_assert_eq!(res.e, res2.e);
      }
  }
}
