//! This module implements `RelaxedR1CSSNARKTrait` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! This version of Spartan does not use preprocessing so the verifier keeps the entire
//! description of R1CS matrices. This is essentially optimal for the verifier when using
//! an IPA-based polynomial commitment scheme.

use crate::{
  errors::NovaError,
  spartan::{
    polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
    powers,
    sumcheck::SumcheckProof,
    PolyEvalInstance, PolyEvalWitness,
  },
  traits::{Engine, TranscriptEngineTrait},
};
use itertools::Itertools as _;

/// A type that represents the prover's key
pub type ProverKey<E, EE> = crate::spartan::batched::ProverKey<E, EE>;

/// A type that represents the verifier's key
pub type VerifierKey<E, EE> = crate::spartan::batched::VerifierKey<E, EE>;

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
pub type RelaxedR1CSSNARK<E, EE> = crate::spartan::batched::BatchedRelaxedR1CSSNARK<E, EE>;

/// Proves a batch of polynomial evaluation claims using Sumcheck
/// reducing them to a single claim at the same point.
///
/// # Details
///
/// We are given as input a list of instance/witness pairs
/// u = [(Cᵢ, xᵢ, eᵢ)], w = [Pᵢ], such that
/// - nᵢ = |xᵢ|
/// - Cᵢ = Commit(Pᵢ)
/// - eᵢ = Pᵢ(xᵢ)
/// - |Pᵢ| = 2^nᵢ
///
/// We allow the polynomial Pᵢ to have different sizes, by appropriately scaling
/// the claims and resulting evaluations from Sumcheck.
pub(in crate::spartan) fn batch_eval_prove<E: Engine>(
  u_vec: Vec<PolyEvalInstance<E>>,
  w_vec: &[PolyEvalWitness<E>],
  transcript: &mut E::TE,
) -> Result<
  (
    PolyEvalInstance<E>,
    PolyEvalWitness<E>,
    SumcheckProof<E>,
    Vec<E::Scalar>,
  ),
  NovaError,
> {
  let num_claims = u_vec.len();
  assert_eq!(w_vec.len(), num_claims);

  // Compute nᵢ and n = maxᵢ{nᵢ}
  let num_rounds = u_vec.iter().map(|u| u.x.len()).collect::<Vec<_>>();

  // Check polynomials match number of variables, i.e. |Pᵢ| = 2^nᵢ
  zip_with_for_each!(iter, (w_vec, num_rounds), |w, num_vars| assert_eq!(
    w.p.len(),
    1 << num_vars
  ));

  // generate a challenge, and powers of it for random linear combination
  let rho = transcript.squeeze(b"r")?;
  let powers_of_rho = powers(&rho, num_claims);

  let (claims, u_xs, comms): (Vec<_>, Vec<_>, Vec<_>) =
    u_vec.into_iter().map(|u| (u.e, u.x, u.c)).multiunzip();

  // Create clones of polynomials to be given to Sumcheck
  // Pᵢ(X)
  let polys_P: Vec<MultilinearPolynomial<E::Scalar>> = w_vec
    .iter()
    .map(|w| MultilinearPolynomial::new(w.p.clone()))
    .collect();
  // eq(xᵢ, X)
  let polys_eq: Vec<MultilinearPolynomial<E::Scalar>> = u_xs
    .into_iter()
    .map(|ux| MultilinearPolynomial::new(EqPolynomial::evals_from_points(&ux)))
    .collect();

  // For each i, check eᵢ = ∑ₓ Pᵢ(x)eq(xᵢ,x), where x ∈ {0,1}^nᵢ
  let comb_func = |poly_P: &E::Scalar, poly_eq: &E::Scalar| -> E::Scalar { *poly_P * *poly_eq };
  let (sc_proof_batch, r, claims_batch) = SumcheckProof::prove_quad_batch(
    &claims,
    &num_rounds,
    polys_P,
    polys_eq,
    &powers_of_rho,
    comb_func,
    transcript,
  )?;

  let (claims_batch_left, _): (Vec<E::Scalar>, Vec<E::Scalar>) = claims_batch;

  transcript.absorb(b"l", &claims_batch_left.as_slice());

  // we now combine evaluation claims at the same point r into one
  let gamma = transcript.squeeze(b"g")?;

  let u_joint =
    PolyEvalInstance::batch_diff_size(&comms, &claims_batch_left, &num_rounds, r, gamma);

  // P = ∑ᵢ γⁱ⋅Pᵢ
  let w_joint = PolyEvalWitness::batch_diff_size(&w_vec.iter().by_ref().collect::<Vec<_>>(), gamma);

  Ok((u_joint, w_joint, sc_proof_batch, claims_batch_left))
}

/// Verifies a batch of polynomial evaluation claims using Sumcheck
/// reducing them to a single claim at the same point.
pub(in crate::spartan) fn batch_eval_verify<E: Engine>(
  u_vec: Vec<PolyEvalInstance<E>>,
  transcript: &mut E::TE,
  sc_proof_batch: &SumcheckProof<E>,
  evals_batch: &[E::Scalar],
) -> Result<PolyEvalInstance<E>, NovaError> {
  let num_claims = u_vec.len();
  assert_eq!(evals_batch.len(), num_claims);

  // generate a challenge
  let rho = transcript.squeeze(b"r")?;
  let powers_of_rho = powers(&rho, num_claims);

  // Compute nᵢ and n = maxᵢ{nᵢ}
  let num_rounds = u_vec.iter().map(|u| u.x.len()).collect::<Vec<_>>();
  let num_rounds_max = *num_rounds.iter().max().unwrap();

  let claims = u_vec.iter().map(|u| u.e).collect::<Vec<_>>();

  let (claim_batch_final, r) =
    sc_proof_batch.verify_batch(&claims, &num_rounds, &powers_of_rho, 2, transcript)?;

  let claim_batch_final_expected = {
    let evals_r = u_vec.iter().map(|u| {
      let (_, r_hi) = r.split_at(num_rounds_max - u.x.len());
      EqPolynomial::new(r_hi.to_vec()).evaluate(&u.x)
    });

    zip_with!(
      (evals_r, evals_batch.iter(), powers_of_rho.iter()),
      |e_i, p_i, rho_i| e_i * *p_i * rho_i
    )
    .sum()
  };

  if claim_batch_final != claim_batch_final_expected {
    return Err(NovaError::InvalidSumcheckProof);
  }

  transcript.absorb(b"l", &evals_batch);

  // we now combine evaluation claims at the same point r into one
  let gamma = transcript.squeeze(b"g")?;

  let comms = u_vec.into_iter().map(|u| u.c).collect::<Vec<_>>();

  let u_joint = PolyEvalInstance::batch_diff_size(&comms, evals_batch, &num_rounds, r, gamma);

  Ok(u_joint)
}
