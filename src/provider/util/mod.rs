//! Utilities for provider module.
pub(in crate::provider) mod fb_msm;
pub mod msm {
  use halo2curves::msm::best_multiexp;
  use halo2curves::CurveAffine;

  // this argument swap is useful until Rust gets named arguments
  // and saves significant complexity in macro code
  pub fn cpu_best_msm<C: CurveAffine>(bases: &[C], scalars: &[C::Scalar]) -> C::Curve {
    best_multiexp(scalars, bases)
  }
}

pub mod field {
  use crate::errors::NovaError;
  use ff::{BatchInverter, Field};

  #[inline]
  pub fn batch_invert<F: Field>(mut v: Vec<F>) -> Result<Vec<F>, NovaError> {
    // we only allocate the scratch space if every element of v is nonzero
    let mut scratch_space = v
      .iter()
      .map(|x| {
        if !x.is_zero_vartime() {
          Ok(*x)
        } else {
          Err(NovaError::InternalError)
        }
      })
      .collect::<Result<Vec<_>, _>>()?;
    let _ = BatchInverter::invert_with_external_scratch(&mut v, &mut scratch_space[..]);
    Ok(v)
  }
}

pub mod iterators {
  use ff::Field;
  use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
  use rayon_scan::ScanParallelIterator;
  use std::iter::DoubleEndedIterator;
  use std::{
    borrow::Borrow,
    ops::{AddAssign, MulAssign},
  };

  pub trait DoubleEndedIteratorExt: DoubleEndedIterator {
    /// This function employs Horner's scheme and core traits to create a combination of an iterator input with the powers
    /// of a provided coefficient.
    fn rlc<T, F>(&mut self, coefficient: &F) -> T
    where
      T: Clone + for<'a> MulAssign<&'a F> + for<'r> AddAssign<&'r T>,
      Self::Item: Borrow<T>,
    {
      let mut iter = self.rev();
      let Some(fst) = iter.next() else {
        panic!("input iterator should not be empty")
      };

      iter.fold(fst.borrow().clone(), |mut acc, item| {
        acc *= coefficient;
        acc += item.borrow();
        acc
      })
    }
  }

  impl<I: DoubleEndedIterator> DoubleEndedIteratorExt for I {}

  pub trait IndexedParallelIteratorExt: IndexedParallelIterator {
    /// This function core traits to create a combination of an iterator input with the powers
    /// of a provided coefficient.
    fn rlc<T, F>(self, coefficient: &F) -> T
    where
      F: Field,
      Self::Item: Borrow<T>,
      T: Clone + for<'a> MulAssign<&'a F> + for<'r> AddAssign<&'r T> + Send + Sync,
    {
      debug_assert!(self.len() > 0);
      // generate an iterator of powers of the right length
      let v = {
        let mut v = vec![*coefficient; self.len()];
        v[0] = F::ONE;
        v
      };
      // the collect is due to Scan being unindexed
      let powers: Vec<_> = v.into_par_iter().scan(|a, b| *a * *b, F::ONE).collect();

      self
        .zip_eq(powers.into_par_iter())
        .map(|(pt, val)| {
          let mut pt = pt.borrow().clone();
          pt *= &val;
          pt
        })
        .reduce_with(|mut a, b| {
          a += &b;
          a
        })
        .unwrap()
    }
  }

  impl<I: IndexedParallelIterator> IndexedParallelIteratorExt for I {}
}

#[cfg(test)]
pub mod test_utils {
  //! Contains utilities for testing and benchmarking.
  use crate::spartan::polys::multilinear::MultilinearPolynomial;
  use crate::traits::{
    commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait, Engine,
  };
  use ff::Field;
  use rand::rngs::StdRng;
  use rand_core::{CryptoRng, OsRng, RngCore};
use serde::Serialize;
  use std::sync::Arc;

  /// Returns a random polynomial, a point and calculate its evaluation.
  pub(crate) fn random_poly_with_eval<E: Engine, R: RngCore + CryptoRng>(
    num_vars: usize,
    mut rng: &mut R,
  ) -> (
    MultilinearPolynomial<<E as Engine>::Scalar>,
    Vec<<E as Engine>::Scalar>,
    <E as Engine>::Scalar,
  ) {
    // Generate random polynomial and point.
    let poly = MultilinearPolynomial::random(num_vars, &mut rng);
    let point = (0..num_vars)
      .map(|_| <E as Engine>::Scalar::random(&mut rng))
      .collect::<Vec<_>>();

    // Calculation evaluation of point over polynomial.
    let eval = poly.evaluate(&point);

    (poly, point, eval)
  }

  // /// Methods used to test the prove and verify flow of [`MultilinearPolynomial`] Commitment Schemes
  // /// (PCS).
  // ///
  // /// Generates a random polynomial and point from a seed to test a proving/verifying flow of one
  // /// of our [`EvaluationEngine`].
  // pub(crate) fn prove_verify_from_num_vars<E: Engine + Serialize, EE: EvaluationEngineTrait<E>>(
  //   num_vars: usize,
  // ) {
  //   use rand_core::SeedableRng;

    // let mut rng = StdRng::seed_from_u64(num_vars as u64);

  //   let (poly, point, eval) = random_poly_with_eval::<E, StdRng>(num_vars, &mut rng);

  //   // Mock commitment key.
  //   let ck = E::CE::setup(b"test", 1 << num_vars);
  //   let ck = Arc::new(ck);

  //   let r_cross = E::Scalar::random(&mut OsRng);
  //   // Commits to the provided vector using the provided generators.
  //   let commitment = E::CE::commit(&ck, poly.evaluations(), &r_cross);

  //   prove_verify_with::<E, EE>(ck, &commitment, &poly, &point, &eval, true)
  // }

  // fn prove_verify_with<E: Engine + Serialize, EE: EvaluationEngineTrait<E>>(
  //   ck: Arc<<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey>,
  //   commitment: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
  //   poly: &MultilinearPolynomial<<E as Engine>::Scalar>,
  //   point: &[<E as Engine>::Scalar],
  //   eval: &<E as Engine>::Scalar,
  //   evaluate_bad_proof: bool,
  // ) {
  //   use crate::traits::TranscriptEngineTrait;
  //   use std::ops::Add;

  //   // Generate Prover and verifier key for given commitment key.
  //   let ock = ck.clone();
  //   let (prover_key, verifier_key) = EE::setup(ck);

  //   // Generate proof.
  //   let mut prover_transcript = E::TE::new(b"TestEval");
  //   let proof = EE::prove_batch(
  //     &*ock,
  //     &prover_key,
  //     &mut prover_transcript,
  //     commitment,
  //     poly.evaluations(),
  //     point,
  //     eval,
  //   )
  //   .unwrap();
  //   let pcp = prover_transcript.squeeze(b"c").unwrap();

  //   // Verify proof.
  //   let mut verifier_transcript = E::TE::new(b"TestEval");
  //   EE::verify_batch(
  //     &verifier_key,
  //     &mut verifier_transcript,
  //     commitment,
  //     point,
  //     eval,
  //     &proof,
  //   )
  //   .unwrap();
  //   let pcv = verifier_transcript.squeeze(b"c").unwrap();

  //   // Check if the prover transcript and verifier transcript are kept in the same state.
  //   assert_eq!(pcp, pcv);

  //   if evaluate_bad_proof {
  //     // Generate another point to verify proof. Also produce eval.
  //     let altered_verifier_point = point
  //       .iter()
  //       .map(|s| s.add(<E as Engine>::Scalar::ONE))
  //       .collect::<Vec<_>>();
  //     let altered_verifier_eval =
  //       MultilinearPolynomial::evaluate_with(poly.evaluations(), &altered_verifier_point);

  //     // Verify proof, should fail.
  //     let mut verifier_transcript = E::TE::new(b"TestEval");
  //     assert!(EE::verify(
  //       &verifier_key,
  //       &mut verifier_transcript,
  //       commitment,
  //       &altered_verifier_point,
  //       &altered_verifier_eval,
  //       &proof,
  //     )
  //     .is_err());
  //   }
  // }
}
