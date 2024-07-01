//! This module implements Nova's evaluation engine using `HyperKZG`, a KZG-based polynomial commitment for multilinear polynomials
//! HyperKZG is based on the transformation from univariate PCS to multilinear PCS in the Gemini paper (section 2.4.2 in `<https://eprint.iacr.org/2022/420.pdf>`).
//! However, there are some key differences:
//! (1) HyperKZG works with multilinear polynomials represented in evaluation form (rather than in coefficient form in Gemini's transformation).
//! This means that Spartan's polynomial IOP can use commit to its polynomials as-is without incurring any interpolations or FFTs.
//! (2) HyperKZG is specialized to use KZG as the univariate commitment scheme, so it includes several optimizations (both during the transformation of multilinear-to-univariate claims
//! and within the KZG commitment scheme implementation itself).
//! (3) HyperKZG also includes optimisation based on so called Shplonk/HaloInfinite technique (`<https://hackmd.io/@adrian-aztec/BJxoyeCqj#Phase-2-Gemini>`).
//! Compared to pure HyperKZG, this optimisation in theory improves prover (at cost of using 1 fixed KZG opening) and verifier (at cost of eliminating MSM)
//!
#![allow(non_snake_case)]
use crate::provider::pcs::kzg10_utilities::UVKZGPCS;
use crate::{
  errors::NovaError,
  provider::{
    pcs::kzg10_utilities::{KZGCommitmentEngine, KZGProverKey, KZGVerifierKey, UniversalKZGParam},
    pcs::pedersen::Commitment,
    traits::DlogGroup,
    util::iterators::IndexedParallelIteratorExt as _,
  },
  spartan::{math::Math, polys::univariate::UniPoly},
  traits::{
    commitment::{CommitmentEngineTrait, Len},
    evaluation::EvaluationEngineTrait,
    Engine as NovaEngine, Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
};
use core::marker::PhantomData;
use ff::{Field, PrimeFieldBits};
use group::{prime::PrimeCurveAffine as _, Curve, Group as _};
use itertools::Itertools as _;
use pairing::{Engine, MillerLoopResult, MultiMillerLoop};
use rayon::iter::{
  IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::prelude::*;
use ref_cast::RefCast as _;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::sync::Arc;

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::G1Affine: Serialize, E::Fr: Serialize",
  deserialize = "E::G1Affine: Deserialize<'de>, E::Fr: Deserialize<'de>"
))]
pub struct EvaluationArgument<E: Engine> {
  comms: Vec<E::G1Affine>,
  evals: Vec<Vec<E::Fr>>,
  R_x: Vec<E::Fr>,
  C_Q: E::G1Affine,
  C_H: E::G1Affine,
}

/// Provides an implementation of a polynomial evaluation engine using KZG
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<E, NE> {
  _p: PhantomData<(E, NE)>,
}

// This impl block defines helper functions that are not a part of
// EvaluationEngineTrait, but that we will use to implement the trait methods.
impl<E, NE> EvaluationEngine<E, NE>
where
  E: Engine,
  NE: NovaEngine<GE = E::G1, Scalar = E::Fr, CE = KZGCommitmentEngine<E>>,
  E::G1: DlogGroup<ScalarExt = E::Fr, AffineExt = E::G1Affine>,
  // the following bounds repeat existing, satisfied bounds on associated types of the above
  // but are required since the equality constraints we use in the above do not transitively carry bounds
  // we should be able to remove most of those constraints when rust supports associated_type_bounds
  E::Fr: Serialize + DeserializeOwned,
  E::G1Affine: Serialize + DeserializeOwned,
  E::G1Affine: TranscriptReprTrait<E::G1>, // TODO: this bound on DlogGroup is really unusable!
  E::G2Affine: Serialize + DeserializeOwned,
  E::Fr: PrimeFieldBits + TranscriptReprTrait<E::G1>,
  <E::G1 as Group>::Base: TranscriptReprTrait<E::G1>,
{
  fn compute_challenge(
    com: &[E::G1Affine],
    transcript: &mut impl TranscriptEngineTrait<NE>,
  ) -> E::Fr {
    transcript.absorb(b"c", &com);
    transcript.squeeze(b"c").unwrap()
  }

  // Compute challenge q = Hash(vk, C0, ..., C_{k-1}, u0, ...., u_{t-1},
  // (f_i(u_j))_{i=0..k-1,j=0..t-1})
  // It is assumed that both 'C' and 'u' are already absorbed by the transcript
  fn get_batch_challenge(
    v: &[Vec<E::Fr>],
    transcript: &mut impl TranscriptEngineTrait<NE>,
  ) -> E::Fr {
    transcript.absorb(
      b"v",
      &v.iter()
        .flatten()
        .cloned()
        .collect::<Vec<E::Fr>>()
        .as_slice(),
    );

    transcript.squeeze(b"r").unwrap()
  }

  fn compute_a(c_q: &E::G1Affine, transcript: &mut impl TranscriptEngineTrait<NE>) -> E::Fr {
    transcript.absorb(b"C_Q", c_q);
    transcript.squeeze(b"a").unwrap()
  }

  fn compute_pi_polynomials(hat_P: &[E::Fr], point: &[E::Fr]) -> Vec<Vec<E::Fr>> {
    let mut polys: Vec<Vec<E::Fr>> = Vec::new();
    polys.push(hat_P.to_vec());

    for i in 0..point.len() - 1 {
      let Pi_len = polys[i].len() / 2;
      let mut Pi = vec![E::Fr::ZERO; Pi_len];

      (0..Pi_len)
        .into_par_iter()
        .map(|j| {
          point[point.len() - i - 1] * (polys[i][2 * j + 1] - polys[i][2 * j]) + polys[i][2 * j]
        })
        .collect_into_vec(&mut Pi);

      polys.push(Pi);
    }

    assert_eq!(polys.len(), hat_P.len().log_2());

    polys
  }

  fn compute_commitments(
    ck: &UniversalKZGParam<E>,
    _C: &Commitment<NE>,
    polys: &[Vec<E::Fr>],
  ) -> Vec<E::G1Affine> {
    let comms: Vec<NE::GE> = (1..polys.len())
      .into_par_iter()
      .map(|i| <NE::CE as CommitmentEngineTrait<NE>>::commit(ck, &polys[i]).comm)
      .collect();

    let mut comms_affine: Vec<E::G1Affine> = vec![E::G1Affine::identity(); comms.len()];
    NE::GE::batch_normalize(&comms, &mut comms_affine);
    comms_affine
  }

  fn compute_evals(polys: &[Vec<E::Fr>], u: &[E::Fr]) -> Vec<Vec<E::Fr>> {
    let mut v = vec![vec!(E::Fr::ZERO; polys.len()); u.len()];
    v.par_iter_mut().enumerate().for_each(|(i, v_i)| {
      // for each point u
      v_i.par_iter_mut().zip_eq(polys).for_each(|(v_ij, f)| {
        // for each poly f (except the last one - since it is constant)
        *v_ij = UniPoly::ref_cast(f).evaluate(&u[i]);
      });
    });
    v
  }

  fn compute_k_polynomial(
    batched_Pi: &UniPoly<E::Fr>,
    Q_x: &UniPoly<E::Fr>,
    D: &UniPoly<E::Fr>,
    R_x: &UniPoly<E::Fr>,
    a: E::Fr,
  ) -> UniPoly<E::Fr> {
    let mut tmp = Q_x.clone();
    tmp *= &D.evaluate(&a);
    tmp[0] += &R_x.evaluate(&a);
    let mut K_x = batched_Pi.clone();
    K_x -= &tmp;
    K_x
  }
}

impl<E, NE> EvaluationEngineTrait<NE> for EvaluationEngine<E, NE>
where
  E: MultiMillerLoop,
  NE: NovaEngine<GE = E::G1, Scalar = E::Fr, CE = KZGCommitmentEngine<E>>,
  E::Fr: Serialize + DeserializeOwned,
  E::G1Affine: Serialize + DeserializeOwned,
  E::G2Affine: Serialize + DeserializeOwned,
  E::G1: DlogGroup<ScalarExt = E::Fr, AffineExt = E::G1Affine>,
  <E::G1 as Group>::Base: TranscriptReprTrait<E::G1>, // Note: due to the move of the bound TranscriptReprTrait<G> on G::Base from Group to Engine
  E::Fr: PrimeFieldBits, // TODO due to use of gen_srs_for_testing, make optional
  E::Fr: TranscriptReprTrait<E::G1>,
  E::G1Affine: TranscriptReprTrait<E::G1>,
{
  type EvaluationArgument = EvaluationArgument<E>;
  type ProverKey = KZGProverKey<E>;
  type VerifierKey = KZGVerifierKey<E>;

  fn setup(ck: Arc<UniversalKZGParam<E>>) -> (Self::ProverKey, Self::VerifierKey) {
    let len = ck.length() - 1;
    UniversalKZGParam::trim(ck, len)
  }

  fn prove(
    ck: &UniversalKZGParam<E>,
    pk: &Self::ProverKey,
    transcript: &mut <NE as NovaEngine>::TE,
    _C: &Commitment<NE>,
    hat_P: &[E::Fr],
    point: &[E::Fr],
    _eval: &E::Fr,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let x: Vec<E::Fr> = point.to_vec();
    let ell = x.len();
    let n = hat_P.len();
    assert_eq!(n, 1 << ell); // Below we assume that n is a power of two

    // Phase 1  -- create commitments com_1, ..., com_\ell
    // We do not compute final Pi (and its commitment as well since it is already committed according to EvaluationEngineTrait API) as it is constant and equals to 'eval'
    // also known to verifier, so can be derived on its side as well
    let polys = Self::compute_pi_polynomials(hat_P, point);
    let comms = Self::compute_commitments(ck, _C, &polys);

    // Phase 2
    let r = Self::compute_challenge(&comms, transcript);
    let u = vec![r, -r, r * r];
    let evals = Self::compute_evals(&polys, &u);

    // Phase 3
    // Compute B(x) = f_0(x) + q * f_1(x) + ... + q^(k-1) * f_{k-1}(x)
    let q = Self::get_batch_challenge(&evals, transcript);
    let batched_Pi: UniPoly<E::Fr> = polys.into_par_iter().map(UniPoly::new).rlc(&q);

    // Q(x), R(x) = P(x) / D(x), where D(x) = (x - r) * (x + r) * (x - r^2) = 1 * x^3 - r^2 * x^2 - r^2 * x + r^4
    let D = UniPoly::new(vec![u[2] * u[2], -u[2], -u[2], E::Fr::from(1)]);
    let (Q_x, R_x) = batched_Pi.divide_with_q_and_r(&D).unwrap();

    let C_Q = <NE::CE as CommitmentEngineTrait<NE>>::commit(ck, &Q_x.coeffs)
      .comm
      .to_affine();

    let a = Self::compute_a(&C_Q, transcript);

    // K(x) = P(x) - Q(x) * D(a) - R(a), note that R(a) should be subtracted from a free term of polynomial
    let K_x = Self::compute_k_polynomial(&batched_Pi, &Q_x, &D, &R_x, a);

    let C_H = UVKZGPCS::<E>::open(pk, &K_x, &a).unwrap();

    Ok(EvaluationArgument::<E> {
      comms,
      evals,
      R_x: R_x.coeffs,
      C_Q,
      C_H: C_H.opening,
    })
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut <NE as NovaEngine>::TE,
    C: &Commitment<NE>,
    point: &[E::Fr],
    P_of_x: &E::Fr,
    pi: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let r = Self::compute_challenge(&pi.comms, transcript);
    let u = [r, -r, r * r];

    if pi.evals.len() != u.len() {
      return Err(NovaError::ProofVerifyError);
    }
    if pi.R_x.len() != u.len() {
      return Err(NovaError::ProofVerifyError);
    }

    let mut comms = pi.comms.to_vec();
    comms.insert(0, C.comm.to_affine());

    let q = Self::get_batch_challenge(&pi.evals, transcript);
    let R_x = UniPoly::new(pi.R_x.clone());

    let verification_failed = pi.evals.iter().zip_eq(u.iter()).any(|(evals_i, u_i)| {
      // here we check correlation between R polynomial and batched evals, e.g.:
      // 1) R(r) == eval at r
      // 2) R(-r) == eval at -r
      // 3) R(r^2) == eval at r^2
      let batched_eval = UniPoly::ref_cast(evals_i).evaluate(&q);
      batched_eval != R_x.evaluate(u_i)
    });
    if verification_failed {
      return Err(NovaError::ProofVerifyError);
    }

    // here we check that Pi polynomials were correctly constructed by the prover, using 'r' as a random point, e.g:
    // P_i_even = P_i(r) + P_i(-r) * 1/2
    // P_i_odd = P_i(r) - P_i(-r) * 1/2*r
    // P_i+1(r^2) == (1 - point_i) * P_i_even + point_i * P_i_odd -> should hold, according to Gemini transformation
    let mut point = point.to_vec();
    point.reverse();

    let r_mul_2 = E::Fr::from(2) * r;
    #[allow(clippy::disallowed_methods)]
    let verification_failed = pi.evals[0]
      .par_iter()
      .chain(&[*P_of_x])
      .zip_eq(pi.evals[1].par_iter().chain(&[*P_of_x]))
      .zip(pi.evals[2][1..].par_iter().chain(&[*P_of_x]))
      .enumerate()
      .any(|(index, ((eval_r, eval_minus_r), eval_r_squared))| {
        // some optimisation to avoid using expensive inversions:
        // P_i+1(r^2) == (1 - point_i) * (P_i(r) + P_i(-r)) * 1/2 + point_i * (P_i(r) - P_i(-r)) * 1/2 * r
        // is equivalent to:
        // 2 * r * P_i+1(r^2) == r * (1 - point_i) * (P_i(r) + P_i(-r)) + point_i * (P_i(r) - P_i(-r))

        let even = *eval_r + eval_minus_r;
        let odd = *eval_r - eval_minus_r;
        let right = r * ((E::Fr::ONE - point[index]) * even) + (point[index] * odd);
        let left = *eval_r_squared * r_mul_2;
        left != right
      });

    if verification_failed {
      return Err(NovaError::ProofVerifyError);
    }

    let C_P: E::G1 = comms.par_iter().map(|comm| comm.to_curve()).rlc(&q);
    let C_Q = pi.C_Q;
    let C_H = pi.C_H;
    let r_squared = u[2];

    // D = (x - r) * (x + r) * (x - r^2) = 1 * x^3 - r^2 * x^2 - r^2 * x + r^4
    let D = UniPoly::new(vec![
      r_squared * r_squared,
      -r_squared,
      -r_squared,
      E::Fr::from(1),
    ]);

    let a = Self::compute_a(&C_Q, transcript);

    let C_K = C_P - (C_Q * D.evaluate(&a) + vk.g * R_x.evaluate(&a));

    let pairing_inputs: Vec<(E::G1Affine, E::G2Prepared)> = vec![
      (C_H, vk.beta_h.into()),
      ((C_H * (-a) - C_K).to_affine(), vk.h.into()),
    ];

    #[allow(clippy::map_identity)]
    let pairing_input_refs = pairing_inputs
      .iter()
      .map(|(a, b)| (a, b))
      .collect::<Vec<_>>();

    let pairing_result = E::multi_miller_loop(pairing_input_refs.as_slice()).final_exponentiation();
    let successful: bool = pairing_result.is_identity().into();
    if !successful {
      return Err(NovaError::ProofVerifyError);
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::util::iterators::DoubleEndedIteratorExt as _;
  use crate::provider::util::test_utils::prove_verify_from_num_vars;
  use crate::spartan::powers;
  use crate::traits::TranscriptEngineTrait;
  use crate::zip_with;
  use crate::{provider::keccak::Keccak256Transcript, CommitmentEngineTrait, CommitmentKey};
  use bincode::Options;
  use expect_test::expect;
  use halo2curves::bn256::G1;
  use itertools::Itertools;

  type E = halo2curves::bn256::Bn256;
  type NE = crate::provider::Bn256EngineKZG;
  type Fr = <NE as NovaEngine>::Scalar;

  fn test_commitment_to_k_polynomial_correctness(
    ck: &CommitmentKey<NE>,
    C: &Commitment<NE>,
    poly: &[Fr],
    point: &[Fr],
    _eval: &Fr,
  ) {
    let polys = EvaluationEngine::<E, NE>::compute_pi_polynomials(poly, point);
    let mut comms = EvaluationEngine::<E, NE>::compute_commitments(ck, C, &polys);
    comms.insert(0, C.comm.to_affine());

    let q = Fr::from(8165763);
    let q_powers = batch_challenge_powers(q, polys.len());
    let batched_Pi: UniPoly<Fr> = polys.clone().into_iter().map(UniPoly::new).rlc(&q);

    let r = Fr::from(1354678);
    let r_squared = r * r;

    let divident = batched_Pi.clone();
    let D = UniPoly::new(vec![
      r_squared * r_squared,
      -r_squared,
      -r_squared,
      Fr::from(1),
    ]);
    let (Q_x, R_x) = divident.divide_with_q_and_r(&D).unwrap();

    let a = Fr::from(938576);

    let K_x = EvaluationEngine::<E, NE>::compute_k_polynomial(&batched_Pi, &Q_x, &D, &R_x, a);

    let mut C_P = G1::identity();
    q_powers.iter().zip_eq(comms.iter()).for_each(|(q_i, C_i)| {
      C_P += *C_i * q_i;
    });

    let C_Q =
      <<crate::provider::Bn256EngineKZG as NovaEngine>::CE as CommitmentEngineTrait<NE>>::commit(
        ck,
        &Q_x.coeffs,
      )
      .comm
      .to_affine();

    // Check that Cp - Cq * D(a) - g1 * R(a) == MSM(ck, K(x))
    let C_K = C_P - C_Q * D.evaluate(&a) - ck.powers_of_g[0] * R_x.evaluate(&a);

    let C_K_expected =
      <<crate::provider::Bn256EngineKZG as NovaEngine>::CE as CommitmentEngineTrait<NE>>::commit(
        ck,
        &K_x.coeffs,
      )
      .comm
      .to_affine();

    assert_eq!(C_K_expected, C_K.to_affine());
  }

  fn test_k_polynomial_correctness(poly: &[Fr], point: &[Fr], _eval: &Fr) {
    let polys = EvaluationEngine::<E, NE>::compute_pi_polynomials(poly, point);
    let q = Fr::from(8165763);
    let batched_Pi: UniPoly<Fr> = polys.clone().into_iter().map(UniPoly::new).rlc(&q);

    let r = Fr::from(56263);
    let r_squared = r * r;

    let divident = batched_Pi.clone();
    let D = UniPoly::new(vec![
      r_squared * r_squared,
      -r_squared,
      -r_squared,
      Fr::from(1),
    ]);
    let (Q_x, R_x) = divident.divide_with_q_and_r(&D).unwrap();

    let a = Fr::from(190837645);

    let K_x = EvaluationEngine::<E, NE>::compute_k_polynomial(&batched_Pi, &Q_x, &D, &R_x, a);

    assert_eq!(Fr::from(0), K_x.evaluate(&a));
  }

  fn test_d_polynomial_correctness(poly: &[Fr], point: &[Fr], _eval: &Fr) {
    let polys = EvaluationEngine::<E, NE>::compute_pi_polynomials(poly, point);
    let q = Fr::from(8165763);
    let batched_Pi: UniPoly<Fr> = polys.clone().into_iter().map(UniPoly::new).rlc(&q);

    let r = Fr::from(2895776832);
    let r_squared = r * r;

    let divident = batched_Pi.clone();
    // D(x) = (x - r) * (x + r) * (x - r^2)
    let D = UniPoly::new(vec![
      r_squared * r_squared,
      -r_squared,
      -r_squared,
      Fr::from(1),
    ]);
    let (Q_x, R_x) = divident.divide_with_q_and_r(&D).unwrap();

    let evaluation_scalar = Fr::from(182746);
    assert_eq!(
      batched_Pi.evaluate(&evaluation_scalar),
      D.evaluate(&evaluation_scalar) * Q_x.evaluate(&evaluation_scalar)
        + R_x.evaluate(&evaluation_scalar)
    );

    // Check that Q(x) = (P(x) - R(x)) / D(x)
    let mut P_x = batched_Pi.clone();
    let minus_R_x = UniPoly::new(
      R_x
        .clone()
        .coeffs
        .into_iter()
        .map(|coeff| -coeff)
        .collect::<Vec<Fr>>(),
    );
    P_x += &minus_R_x;

    let divident = P_x.clone();
    let (Q_x_recomputed, _) = divident.divide_with_q_and_r(&D).unwrap();

    assert_eq!(Q_x, Q_x_recomputed);
  }

  fn test_batching_property_on_evaluation(poly: &[Fr], point: &[Fr], _eval: &Fr) {
    let polys = EvaluationEngine::<E, NE>::compute_pi_polynomials(poly, point);

    let q = Fr::from(97652);
    let u = [Fr::from(10), Fr::from(20), Fr::from(50)];

    let batched_Pi: UniPoly<Fr> = polys.clone().into_iter().map(UniPoly::new).rlc(&q);

    let q_powers = batch_challenge_powers(q, polys.len());
    for evaluation_scalar in u.iter() {
      let evals = polys
        .clone()
        .into_iter()
        .map(|poly| UniPoly::new(poly).evaluate(evaluation_scalar))
        .collect::<Vec<Fr>>();

      let expected = zip_with!((evals.iter(), q_powers.iter()), |eval, q| eval * q)
        .collect::<Vec<Fr>>()
        .into_iter()
        .sum::<Fr>();

      let actual = batched_Pi.evaluate(evaluation_scalar);
      assert_eq!(expected, actual);
    }
  }

  #[test]
  fn test_hyperkzg_shplonk_unit_tests() {
    // poly = [1, 2, 1, 4, 1, 2, 1, 4]
    let poly = vec![
      Fr::ONE,
      Fr::from(2),
      Fr::from(1),
      Fr::from(4),
      Fr::ONE,
      Fr::from(2),
      Fr::from(1),
      Fr::from(4),
    ];

    // point = [4,3,8]
    let point = vec![Fr::from(4), Fr::from(3), Fr::from(8)];

    // eval = 57
    let eval = Fr::from(57);

    let ck: CommitmentKey<NE> =
      <KZGCommitmentEngine<E> as CommitmentEngineTrait<NE>>::setup(b"test", poly.len());

    let ck = Arc::new(ck);
    let C: Commitment<NE> = KZGCommitmentEngine::commit(&ck, &poly);

    test_batching_property_on_evaluation(&poly, &point, &eval);
    test_d_polynomial_correctness(&poly, &point, &eval);
    test_k_polynomial_correctness(&poly, &point, &eval);
    test_commitment_to_k_polynomial_correctness(&ck, &C, &poly, &point, &eval);
  }

  #[test]
  fn test_hyperkzg_shplonk_pcs() {
    let n = 8;

    // poly = [1, 2, 1, 4, 1, 2, 1, 4]
    let poly = vec![
      Fr::ONE,
      Fr::from(2),
      Fr::from(1),
      Fr::from(4),
      Fr::ONE,
      Fr::from(2),
      Fr::from(1),
      Fr::from(4),
    ];

    // point = [4,3,8]
    let point = vec![Fr::from(4), Fr::from(3), Fr::from(8)];

    // eval = 57
    let eval = Fr::from(57);

    let ck: CommitmentKey<NE> =
      <KZGCommitmentEngine<E> as CommitmentEngineTrait<NE>>::setup(b"test", n);
    let ck = Arc::new(ck);
    let (pk, vk): (KZGProverKey<E>, KZGVerifierKey<E>) =
      EvaluationEngine::<E, NE>::setup(ck.clone());

    // make a commitment
    let C: Commitment<NE> = KZGCommitmentEngine::commit(&ck, &poly);

    let mut prover_transcript = Keccak256Transcript::new(b"TestEval");
    let proof =
      EvaluationEngine::<E, NE>::prove(&ck, &pk, &mut prover_transcript, &C, &poly, &point, &eval)
        .unwrap();

    let mut verifier_transcript = Keccak256Transcript::<NE>::new(b"TestEval");
    EvaluationEngine::<E, NE>::verify(&vk, &mut verifier_transcript, &C, &point, &eval, &proof)
      .unwrap();
  }

  #[test]
  fn test_hyperkzg_shplonk_pcs_negative() {
    let n = 8;
    // poly = [1, 2, 1, 4, 1, 2, 1, 4]
    let poly = vec![
      Fr::ONE,
      Fr::from(2),
      Fr::from(1),
      Fr::from(4),
      Fr::ONE,
      Fr::from(2),
      Fr::from(1),
      Fr::from(4),
    ];
    // point = [4,3,8]
    let point = vec![Fr::from(4), Fr::from(3), Fr::from(8)];
    // eval = 57
    let eval = Fr::from(57);

    // eval = 57
    let eval1 = Fr::from(56); // wrong eval
    test_negative_inner(n, &poly, &point, &eval1);

    // point = [4,3,8]
    let point1 = vec![Fr::from(4), Fr::from(3), Fr::from(7)]; // wrong point
    test_negative_inner(n, &poly, &point1, &eval);

    // poly = [1, 2, 1, 4, 1, 2, 1, 4]
    let poly1 = vec![
      Fr::ONE,
      Fr::from(2),
      Fr::from(1),
      Fr::from(4),
      Fr::ONE,
      Fr::from(2),
      Fr::from(200),
      Fr::from(100),
    ]; // wrong poly
    test_negative_inner(n, &poly1, &point, &eval);
  }

  fn test_negative_inner(n: usize, poly: &[Fr], point: &[Fr], eval: &Fr) {
    let ck: CommitmentKey<NE> =
      <KZGCommitmentEngine<E> as CommitmentEngineTrait<NE>>::setup(b"test", n);
    let ck = Arc::new(ck);
    let (pk, vk): (KZGProverKey<E>, KZGVerifierKey<E>) =
      EvaluationEngine::<E, NE>::setup(ck.clone());

    // make a commitment
    let C: Commitment<NE> = KZGCommitmentEngine::commit(&ck, poly);

    let mut prover_transcript = Keccak256Transcript::new(b"TestEval");
    let mut verifier_transcript = Keccak256Transcript::<NE>::new(b"TestEval");

    let proof =
      EvaluationEngine::<E, NE>::prove(&ck, &pk, &mut prover_transcript, &C, poly, point, eval)
        .unwrap();

    assert!(EvaluationEngine::<E, NE>::verify(
      &vk,
      &mut verifier_transcript,
      &C,
      point,
      eval,
      &proof
    )
    .is_err());
  }

  #[test]
  fn test_hyperkzg_shplonk_pcs_negative_wrong_commitment() {
    let n = 8;
    // poly = [1, 2, 1, 4, 1, 2, 1, 4]
    let poly = vec![
      Fr::ONE,
      Fr::from(2),
      Fr::from(1),
      Fr::from(4),
      Fr::ONE,
      Fr::from(2),
      Fr::from(1),
      Fr::from(4),
    ];
    // point = [4,3,8]
    let point = vec![Fr::from(4), Fr::from(3), Fr::from(8)];
    // eval = 57
    let eval = Fr::from(57);

    // altered_poly = [85, 84, 83, 82, 81, 80, 79, 78]
    let altered_poly = vec![
      Fr::from(85),
      Fr::from(84),
      Fr::from(83),
      Fr::from(82),
      Fr::from(81),
      Fr::from(80),
      Fr::from(79),
      Fr::from(78),
    ];

    let ck: CommitmentKey<NE> =
      <KZGCommitmentEngine<E> as CommitmentEngineTrait<NE>>::setup(b"test", n);

    let C1: Commitment<NE> = KZGCommitmentEngine::commit(&ck, &poly); // correct commitment
    let C2: Commitment<NE> = KZGCommitmentEngine::commit(&ck, &altered_poly); // wrong commitment

    test_negative_inner_commitment(&poly, &point, &eval, &ck, &C1, &C2); // here we check detection when proof and commitment do not correspond
    test_negative_inner_commitment(&poly, &point, &eval, &ck, &C2, &C2); // here we check detection when proof was built with wrong commitment
  }

  fn test_negative_inner_commitment(
    poly: &[Fr],
    point: &[Fr],
    eval: &Fr,
    ck: &CommitmentKey<NE>,
    C_prover: &Commitment<NE>,
    C_verifier: &Commitment<NE>,
  ) {
    let ck = Arc::new(ck.clone());
    let (pk, vk): (KZGProverKey<E>, KZGVerifierKey<E>) =
      EvaluationEngine::<E, NE>::setup(ck.clone());

    let mut prover_transcript = Keccak256Transcript::new(b"TestEval");
    let mut verifier_transcript = Keccak256Transcript::<NE>::new(b"TestEval");

    let proof = EvaluationEngine::<E, NE>::prove(
      &ck,
      &pk,
      &mut prover_transcript,
      C_prover,
      poly,
      point,
      eval,
    )
    .unwrap();

    assert!(EvaluationEngine::<E, NE>::verify(
      &vk,
      &mut verifier_transcript,
      C_verifier,
      point,
      eval,
      &proof
    )
    .is_err());
  }

  #[test]
  fn test_hyperkzg_shplonk_eval() {
    // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
    let n = 4;
    let ck: CommitmentKey<NE> =
      <KZGCommitmentEngine<E> as CommitmentEngineTrait<NE>>::setup(b"test", n);
    let ck = Arc::new(ck);
    let (pk, vk): (KZGProverKey<E>, KZGVerifierKey<E>) =
      EvaluationEngine::<E, NE>::setup(ck.clone());

    // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
    let poly = vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4)];

    let C = <KZGCommitmentEngine<E> as CommitmentEngineTrait<NE>>::commit(&ck, &poly);

    let test_inner = |point: Vec<Fr>, eval: Fr| -> Result<(), NovaError> {
      let mut tr = Keccak256Transcript::<NE>::new(b"TestEval");
      let proof =
        EvaluationEngine::<E, NE>::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).unwrap();
      let mut tr = Keccak256Transcript::new(b"TestEval");
      EvaluationEngine::<E, NE>::verify(&vk, &mut tr, &C, &point, &eval, &proof)
    };

    // Call the prover with a (point, eval) pair.
    // The prover does not recompute so it may produce a proof, but it should not verify
    let point = vec![Fr::from(0), Fr::from(0)];
    let eval = Fr::ONE;
    test_inner(point, eval).unwrap();

    let point = vec![Fr::from(0), Fr::from(1)];
    let eval = Fr::from(2);
    test_inner(point, eval).unwrap();

    let point = vec![Fr::from(1), Fr::from(1)];
    let eval = Fr::from(4);
    test_inner(point, eval).unwrap();

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(3);
    test_inner(point, eval).unwrap();

    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(9);
    test_inner(point, eval).unwrap();

    // Try a couple incorrect evaluations and expect failure
    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(50);
    assert!(test_inner(point, eval).is_err());

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(4);
    assert!(test_inner(point, eval).is_err());
  }

  #[test]
  fn test_hyperkzg_shplonk_transcript_correctness() {
    let n = 4;

    // poly = [1, 2, 1, 4]
    let poly = vec![Fr::ONE, Fr::from(2), Fr::from(1), Fr::from(4)];

    // point = [4,3]
    let point = vec![Fr::from(4), Fr::from(3)];

    // eval = 28
    let eval = Fr::from(28);

    let ck: CommitmentKey<NE> =
      <KZGCommitmentEngine<E> as CommitmentEngineTrait<NE>>::setup(b"test", n);
    let ck = Arc::new(ck);
    let (pk, vk): (KZGProverKey<E>, KZGVerifierKey<E>) =
      EvaluationEngine::<E, NE>::setup(ck.clone());

    // make a commitment
    let C = KZGCommitmentEngine::commit(&ck, &poly);

    // prove an evaluation
    let mut prover_transcript = Keccak256Transcript::new(b"TestEval");
    let proof =
      EvaluationEngine::<E, NE>::prove(&ck, &pk, &mut prover_transcript, &C, &poly, &point, &eval)
        .unwrap();
    let post_c_p = prover_transcript.squeeze(b"c").unwrap();

    // verify the evaluation
    let mut verifier_transcript = Keccak256Transcript::<NE>::new(b"TestEval");
    EvaluationEngine::<E, NE>::verify(&vk, &mut verifier_transcript, &C, &point, &eval, &proof)
      .unwrap();
    let post_c_v = verifier_transcript.squeeze(b"c").unwrap();

    // check if the prover transcript and verifier transcript are kept in the
    // same state
    assert_eq!(post_c_p, post_c_v);

    let proof_bytes = bincode::DefaultOptions::new()
      .with_big_endian()
      .with_fixint_encoding()
      .serialize(&proof)
      .unwrap();
    expect!["432"].assert_eq(&proof_bytes.len().to_string());

    // Change the proof and expect verification to fail
    let mut bad_proof = proof.clone();
    bad_proof.comms[0] = (bad_proof.comms[0] + bad_proof.comms[0] * Fr::from(123)).to_affine();
    let mut verifier_transcript2 = Keccak256Transcript::<NE>::new(b"TestEval");
    assert!(EvaluationEngine::<E, NE>::verify(
      &vk,
      &mut verifier_transcript2,
      &C,
      &point,
      &eval,
      &bad_proof
    )
    .is_err());
  }

  #[test]
  fn test_hyperkzg_shplonk_more() {
    // test the hyperkzg prover and verifier with random instances (derived from a seed)
    for num_vars in [4, 5, 6] {
      prove_verify_from_num_vars::<_, EvaluationEngine<E, NE>>(num_vars);
    }
  }

  /// Compute powers of q : (1, q, q^2, ..., q^(k-1))
  fn batch_challenge_powers(q: Fr, k: usize) -> Vec<Fr> {
    powers(&q, k)
  }
}
