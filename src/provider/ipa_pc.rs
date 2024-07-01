//! This module implements `EvaluationEngine` using an IPA-based polynomial commitment scheme
use crate::{
  digest::SimpleDigestible,
  errors::{NovaError},
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup, },
  spartan::polys::eq::EqPolynomial,
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    evaluation::{EvaluationEngineTrait, GetEvalCommitmentsTrait},
    Engine, TranscriptEngineTrait, TranscriptReprTrait,
  },
  zip_with, Commitment, CommitmentKey, CompressedCommitment, CE,
};
use core::{cmp::max, iter};
use ff::Field;
use group::prime::PrimeCurve;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;

/// Provides an implementation of the prover key
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProverKey<E: Engine> {
  ck_s: CommitmentKey<E>,
}

/// Provides an implementation of the verifier key
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine> {
  pub(in crate::provider) ck_v: Arc<CommitmentKey<E>>,
  pub(in crate::provider) ck_s: CommitmentKey<E>,
}

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct EvaluationArgument<E: Engine>
where
  E::GE: DlogGroup,
{
  nifs: Vec<NIFSForInnerProduct<E>>,
  ipa: InnerProductArgument<E>,
  eval_commitments: Vec<CompressedCommitment<E>>,
}

impl<E: Engine> GetEvalCommitmentsTrait<E> for EvaluationArgument<E>
where
  E::GE: DlogGroup,
{
  fn get_eval_commitment(&self, index: usize) -> CompressedCommitment<E> {
    assert!(self.eval_commitments.len() > index);
    self.eval_commitments[index].clone()
  }
}

impl<E: Engine> SimpleDigestible for VerifierKey<E> {}

/// Provides an implementation of a polynomial evaluation engine using IPA
#[derive(Clone, Debug)]
pub struct EvaluationEngine<E> {
  _p: PhantomData<E>,
}

impl<E> EvaluationEngineTrait<E> for EvaluationEngine<E>
where
  E: Engine + Serialize + for<'de> Deserialize<'de>,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  type ProverKey = ProverKey<E>;
  type VerifierKey = VerifierKey<E>;
  type EvaluationArgument = EvaluationArgument<E>;

  fn setup(
    ck: Arc<<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey>,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    // let ck_c = E::CE::setup(b"ipa", 1);
    let ck_c = E::CE::setup_with_blinding(b"ipa", 1, &E::CE::get_blinding_gen(&ck));

    let pk = ProverKey { ck_s: ck_c.clone() };
    let vk = VerifierKey {
      ck_v: ck.clone(),
      ck_s: ck_c,
    };

    (pk, vk)
  }

  fn get_scalar_gen_pk(pk: Self::ProverKey) -> CommitmentKey<E> {
    pk.ck_s
  }

  fn get_scalar_gen_vk(vk: Self::VerifierKey) -> CommitmentKey<E> {
    vk.ck_s
  }

  fn get_vector_gen_vk(vk: Self::VerifierKey) -> Arc<CommitmentKey<E>> {
    vk.ck_v
  }

  fn prove_batch(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    transcript: &mut E::TE,
    comms_x_vec: &[Commitment<E>],
    polys: &[Vec<E::Scalar>],
    rand_polys: &[E::Scalar],
    points: &[Vec<E::Scalar>],
    y_vec: &[E::Scalar],
    rand_y_vec: &[E::Scalar],
    comm_y_vec: &[CompressedCommitment<E>],
  ) -> Result<Self::EvaluationArgument, NovaError> {
    // sanity checks (these should never fail)
    assert!(polys.len() >= 2);
    assert_eq!(comms_x_vec.len(), polys.len());
    assert_eq!(comms_x_vec.len(), points.len());
    assert_eq!(comms_x_vec.len(), y_vec.len());
    assert_eq!(y_vec.len(), rand_y_vec.len());

    let mut comms_y_vec = Vec::new();

    let mut U_folded = InnerProductInstance::new(
      &comms_x_vec[0],
      &EqPolynomial::new(points[0].clone()).evals(),
      &Commitment::<E>::decompress(&comm_y_vec[0])?,
    );

    comms_y_vec.push(comm_y_vec[0].clone());

    // Record value of eval and randomness used in commitment in the witness
    let mut W_folded =
      InnerProductWitness::new(&polys[0], &rand_polys[0], &y_vec[0], &rand_y_vec[0]);
    let mut nifs = Vec::new();

    for i in 1..polys.len() {
      // perform the folding
      let (n, u, w) = NIFSForInnerProduct::prove(
        pk,
        &U_folded,
        &W_folded,
        &InnerProductInstance::new(
          &comms_x_vec[i],
          &EqPolynomial::new(points[i].clone()).evals(),
          &Commitment::<E>::decompress(&comm_y_vec[i])?,
        ),
        &InnerProductWitness::new(&polys[i], &rand_polys[i], &y_vec[i], &rand_y_vec[i]),
        transcript,
      );

      nifs.push(n);
      comms_y_vec.push(comm_y_vec[i].clone());

      U_folded = u;
      W_folded = w;
    }

    let ipa = InnerProductArgument::prove(ck, &pk.ck_s, &U_folded, &W_folded, transcript)?;

    Ok(EvaluationArgument {
      nifs,
      ipa,
      eval_commitments: comms_y_vec,
    })
  }

  fn verify_batch(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comms_x_vec: &[Commitment<E>],
    points: &[Vec<E::Scalar>],
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let comms_y_vec = &arg.eval_commitments;

    // sanity checks (these should never fail)
    assert!(comms_x_vec.len() >= 2);
    assert_eq!(comms_x_vec.len(), points.len());
    assert_eq!(comms_x_vec.len(), comms_y_vec.len());

    let mut U_folded = InnerProductInstance::new(
      &comms_x_vec[0],
      &EqPolynomial::new(points[0].clone()).evals(),
      &Commitment::<E>::decompress(&comms_y_vec[0])?,
    );
    let mut num_vars = points[0].len();
    for i in 1..comms_x_vec.len() {
      let u = arg.nifs[i - 1].verify(
        &U_folded,
        &InnerProductInstance::new(
          &comms_x_vec[i],
          &EqPolynomial::new(points[i].clone()).evals(),
          &Commitment::<E>::decompress(&comms_y_vec[i])?,
        ),
        transcript,
      );
      U_folded = u;
      num_vars = max(num_vars, points[i].len());
    }

    arg.ipa.verify(
      &vk.ck_v,
      &vk.ck_s,
      (2_usize).pow(num_vars as u32),
      &U_folded,
      transcript,
    )?;

    Ok(())
  }
}

fn inner_product<T: Field + Send + Sync>(a: &[T], b: &[T]) -> T {
  zip_with!(par_iter, (a, b), |x, y| *x * y).sum()
}

/// An inner product instance consists of a commitment to a vector `x` and another vector `a`
/// and the claim that y = <x, a>.
pub struct InnerProductInstance<E: Engine> {
  /// Commitment to vector
  pub comm_x_vec: Commitment<E>,
  /// Public vector
  pub a_vec: Vec<E::Scalar>,
  /// Commitment to scalar
  pub comm_y: Commitment<E>, // commitment to scalar y
}

impl<E> InnerProductInstance<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  /// new inner product instance
  pub fn new(comm_x_vec: &Commitment<E>, a_vec: &[E::Scalar], comm_y: &Commitment<E>) -> Self {
    InnerProductInstance {
      comm_x_vec: *comm_x_vec,
      a_vec: a_vec.to_vec(),
      comm_y: *comm_y,
    }
  }

  fn pad(&self, n: usize) -> InnerProductInstance<E> {
    let mut a_vec = self.a_vec.clone();
    a_vec.resize(n, E::Scalar::ZERO);
    InnerProductInstance {
      comm_x_vec: self.comm_x_vec,
      a_vec,
      comm_y: self.comm_y,
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for InnerProductInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // we do not need to include self.a_vec as in our context it is produced from the transcript
    [
      self.comm_x_vec.to_transcript_bytes(),
      self.comm_y.to_transcript_bytes(),
    ]
    .concat()
  }
}

/// an inner product witness
pub struct InnerProductWitness<E: Engine> {
  x_vec: Vec<E::Scalar>,
  r_x: E::Scalar,
  y: E::Scalar,
  r_y: E::Scalar,
}

impl<E: Engine> InnerProductWitness<E> {
  /// new inner product witness
  pub fn new(x_vec: &[E::Scalar], r_x: &E::Scalar, y: &E::Scalar, r_y: &E::Scalar) -> Self {
    InnerProductWitness {
      x_vec: x_vec.to_vec(),
      r_x: *r_x,
      y: *y,
      r_y: *r_y,
    }
  }

  fn pad(&self, n: usize) -> InnerProductWitness<E> {
    let mut x_vec = self.x_vec.clone();
    x_vec.resize(n, E::Scalar::ZERO);
    InnerProductWitness {
      x_vec,
      r_x: self.r_x,
      y: self.y,
      r_y: self.r_y,
    }
  }
}

/// A non-interactive folding scheme (NIFS) for inner product relations
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFSForInnerProduct<E: Engine>
where
  E::GE: DlogGroup,
{
  comm_cross_term: Commitment<E>, // commitment to cross term (which is a scalar)
}

impl<E: Engine + Serialize + for<'de> Deserialize<'de>> NIFSForInnerProduct<E>
where
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  fn protocol_name() -> &'static [u8] {
    b"NIFSForInnerProduct"
  }

  fn prove(
    pk: &<EvaluationEngine<E> as EvaluationEngineTrait<E>>::ProverKey,
    U1: &InnerProductInstance<E>,
    W1: &InnerProductWitness<E>,
    U2: &InnerProductInstance<E>,
    W2: &InnerProductWitness<E>,
    transcript: &mut E::TE,
  ) -> (Self, InnerProductInstance<E>, InnerProductWitness<E>)
  where
    E::GE: DlogGroup,
    CommitmentKey<E>: CommitmentKeyExtTrait<E>,
  {
    transcript.dom_sep(Self::protocol_name());

    // pad the instances and witness so they are of the same length
    let U1 = U1.pad(max(U1.a_vec.len(), U2.a_vec.len()));
    let U2 = U2.pad(max(U1.a_vec.len(), U2.a_vec.len()));
    let W1 = W1.pad(max(U1.a_vec.len(), U2.a_vec.len()));
    let W2 = W2.pad(max(U1.a_vec.len(), U2.a_vec.len()));

    // add the two commitments and two public vectors to the transcript
    transcript.absorb(b"U1_comm_x_vec", &U1.comm_x_vec);
    transcript.absorb(b"U1_a_vec", &U1.a_vec.as_slice());
    transcript.absorb(b"U2_comm_x_vec", &U2.comm_x_vec);
    transcript.absorb(b"U2_a_vec", &U2.a_vec.as_slice());

    // compute the cross-term
    let cross_term = inner_product(&W1.x_vec, &U2.a_vec) + inner_product(&W2.x_vec, &U1.a_vec);

    // commit to the cross-term
    let r_cross = E::Scalar::random(&mut OsRng);
    let comm_cross = CE::<E>::commit(&pk.ck_s, &[cross_term], &r_cross);

    // add the commitment of the cross-term to the transcript
    transcript.absorb(b"cross_term", &comm_cross);

    // obtain a random challenge
    let r = transcript.squeeze(b"r").unwrap();

    // fold the vectors and their inner product
    let x_vec = W1
      .x_vec
      .par_iter()
      .zip(W2.x_vec.par_iter())
      .map(|(x1, x2)| *x1 + r * x2)
      .collect::<Vec<E::Scalar>>();
    let a_vec = U1
      .a_vec
      .par_iter()
      .zip(U2.a_vec.par_iter())
      .map(|(a1, a2)| *a1 + r * a2)
      .collect::<Vec<E::Scalar>>();

    // fold using the cross term and fold x_vec as well
    let y = W1.y + r * r * W2.y + r * cross_term;
    let comm_x_vec = U1.comm_x_vec + U2.comm_x_vec * r;
    let r_x = W1.r_x + W2.r_x * r;

    // generate commitment to y
    let r_y = W1.r_y + W2.r_y * r * r + r_cross * r;
    let comm_y = CE::<E>::commit(&pk.ck_s, &[y], &r_y);

    let W = InnerProductWitness { x_vec, r_x, y, r_y };

    let U = InnerProductInstance {
      comm_x_vec,
      a_vec,
      comm_y,
    };

    (
      NIFSForInnerProduct {
        comm_cross_term: comm_cross,
      },
      U,
      W,
    )
  }

  fn verify(
    &self,
    U1: &InnerProductInstance<E>,
    U2: &InnerProductInstance<E>,
    transcript: &mut E::TE,
  ) -> InnerProductInstance<E> {
    transcript.dom_sep(Self::protocol_name());

    // pad the instances so they are of the same length
    let U1 = U1.pad(max(U1.a_vec.len(), U2.a_vec.len()));
    let U2 = U2.pad(max(U1.a_vec.len(), U2.a_vec.len()));

    // add the two commitments and two public vectors to the transcript
    transcript.absorb(b"U1_comm_x_vec", &U1.comm_x_vec);
    transcript.absorb(b"U1_a_vec", &U1.a_vec.as_slice());
    transcript.absorb(b"U2_comm_x_vec", &U2.comm_x_vec);
    transcript.absorb(b"U2_a_vec", &U2.a_vec.as_slice());

    // add the commitment to the cross-term to the transcript
    transcript.absorb(b"cross_term", &self.comm_cross_term);

    // obtain a random challenge
    let r = transcript.squeeze(b"r").unwrap();

    // fold the vectors and their inner product
    let a_vec = U1
      .a_vec
      .par_iter()
      .zip(U2.a_vec.par_iter())
      .map(|(x1, x2)| *x1 + r * x2)
      .collect::<Vec<E::Scalar>>();

    let comm_y = U1.comm_y + U2.comm_y * r * r + self.comm_cross_term * r;
    let comm_x_vec = U1.comm_x_vec + U2.comm_x_vec * r;

    InnerProductInstance {
      comm_x_vec,
      a_vec,
      comm_y,
    }
  }
}

/// An inner product argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgument<E: Engine> {
  P_L_vec: Vec<CompressedCommitment<E>>,
  P_R_vec: Vec<CompressedCommitment<E>>,
  delta: CompressedCommitment<E>,
  beta: CompressedCommitment<E>,
  z_1: E::Scalar,
  z_2: E::Scalar,
  _p: PhantomData<E>,
}

impl<E> InnerProductArgument<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  fn protocol_name() -> &'static [u8] {
    b"inner product argument"
  }

  fn bullet_reduce_prover(
    r_P: &E::Scalar,
    x_vec: &[E::Scalar],
    a_vec: &[E::Scalar],
    y: &E::Scalar,
    ck: &CommitmentKey<E>,
    ck_y: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      E::Scalar,        // r_P'
      Commitment<E>,    // P_L
      Commitment<E>,    // P_R
      E::Scalar,        // y_prime
      Vec<E::Scalar>,   // x_vec'
      Vec<E::Scalar>,   // a_vec'
      CommitmentKey<E>, // gens'
    ),
    NovaError,
  > {
    let n = x_vec.len();
    let (gens_L, gens_R) = ck.split_at(n / 2);

    let y_L = inner_product(&x_vec[0..n / 2], &a_vec[n / 2..n]);
    let y_R = inner_product(&x_vec[n / 2..n], &a_vec[0..n / 2]);

    let r_L = E::Scalar::random(&mut OsRng);
    let r_R = E::Scalar::random(&mut OsRng);

    let P_L = CE::<E>::commit(
      &gens_R.combine(ck_y),
      &x_vec[0..n / 2]
        .iter()
        .chain(iter::once(&y_L))
        .copied()
        .collect::<Vec<E::Scalar>>(),
      &r_L,
    );

    let P_R = CE::<E>::commit(
      &gens_L.combine(ck_y),
      &x_vec[n / 2..n]
        .iter()
        .chain(iter::once(&y_R))
        .copied()
        .collect::<Vec<E::Scalar>>(),
      &r_R,
    );

    transcript.absorb(b"L", &P_L.compress());
    transcript.absorb(b"R", &P_R.compress());

    let chal = transcript.squeeze(b"challenge_r")?;

    let chal_square = chal * chal;
    let chal_inverse = chal.invert().unwrap();
    let chal_inverse_square = chal_inverse * chal_inverse;

    // fold the left half and the right half
    let x_vec_prime = x_vec[0..n / 2]
      .par_iter()
      .zip(x_vec[n / 2..n].par_iter())
      .map(|(x_L, x_R)| *x_L * chal + chal_inverse * *x_R)
      .collect::<Vec<E::Scalar>>();

    let a_vec_prime = a_vec[0..n / 2]
      .par_iter()
      .zip(a_vec[n / 2..n].par_iter())
      .map(|(a_L, a_R)| *a_L * chal_inverse + chal * *a_R)
      .collect::<Vec<E::Scalar>>();

    let gens_folded = ck.fold(&chal_inverse, &chal);

    let y_prime = chal_square * y_L + y + chal_inverse_square * y_R;
    let r_P_prime = r_L * chal_square + r_P + r_R * chal_inverse_square;

    Ok((
      r_P_prime,
      P_L,
      P_R,
      y_prime,
      x_vec_prime,
      a_vec_prime,
      gens_folded,
    ))
  }

  /// prover inner product argument
  pub fn prove(
    ck: &CommitmentKey<E>,
    ck_y: &CommitmentKey<E>,
    U: &InnerProductInstance<E>,
    W: &InnerProductWitness<E>,
    transcript: &mut E::TE,
  ) -> Result<Self, NovaError> {
    // The goal here is to prove that x_vec * a_vec = y.
    // We have a hiding vector commitment to x_vec, and a hiding commitment to y.
    // a_vec is a vector in the clear.

    // We will prove this based on Hyrax's Figure 7 and 8.
    // Translation of variables from this code to Hyrax's paper
    //
    // Code                     Hyrax
    // ------------------------------
    // x_vec                    \vec{x}
    // r_x                      r_\Xi
    // comm_x_vec               \Xi
    //
    // a_vec                    \vec{a}
    //
    // y                        y
    // comm_y                   \tau
    // r_y                      r_\tau

    // P                        \Upsilon
    // r_P                      r_\Upsilon
    //
    // ck                     vec<g_i>
    // ck_y                   g

    transcript.dom_sep(Self::protocol_name());

    if U.a_vec.len() != W.x_vec.len() {
      return Err(NovaError::InvalidInputLength);
    }

    transcript.absorb(b"comm_x_vec", &U.comm_x_vec);
    transcript.absorb(b"a_vec", &U.a_vec.as_slice());
    transcript.absorb(b"y", &U.comm_y);

    // Scale generator to be consistent with Bulletproofs Figure 1 (in the Bulletproofs
    // figure, ck_y is "u" and chal is "x").
    let chal = transcript.squeeze(b"r")?;
    let ck_y = ck_y.scale(&chal);

    // two vectors to hold the logarithmic number of group elements, and their masks
    let mut P_L_vec: Vec<CompressedCommitment<E>> = Vec::new();
    let mut P_R_vec: Vec<CompressedCommitment<E>> = Vec::new();

    // Step 1 in Hyrax's Figure 7. The prover doesn't need P explicitly, so we don't
    // need to compute it. We just compute the randomness used in the commitment.
    let mut r_P = W.r_x + W.r_y * chal;

    // we create mutable copies of vectors and generators
    let mut x_vec = W.x_vec.to_vec();
    let mut a_vec = U.a_vec.to_vec();
    let mut ck = ck.clone();
    let mut y = W.y;

    for _i in 0..(U.a_vec.len() as f64).log2() as usize {
      let (r_P_prime, P_L, P_R, y_prime, x_vec_prime, a_vec_prime, ck_prime) =
        Self::bullet_reduce_prover(&r_P, &x_vec, &a_vec, &y, &ck, &ck_y, transcript)?;

      P_L_vec.push(P_L.compress());
      P_R_vec.push(P_R.compress());

      r_P = r_P_prime;
      y = y_prime;
      x_vec = x_vec_prime;
      a_vec = a_vec_prime;
      ck = ck_prime;
    }

    assert_eq!(a_vec.len(), 1);

    // This is after the recursive calls to bullet_reduce in Hyrax
    let r_P_hat = r_P;
    let y_hat = y;
    let a_hat = a_vec[0];
    let c_hat = ck;

    let d = E::Scalar::random(&mut OsRng);
    let r_delta = E::Scalar::random(&mut OsRng);
    let r_beta = E::Scalar::random(&mut OsRng);

    let delta = CE::<E>::commit(&c_hat, &[d], &r_delta).compress();
    let beta = CE::<E>::commit(&ck_y, &[d], &r_beta).compress();

    transcript.absorb(b"beta", &beta);
    transcript.absorb(b"delta", &delta);

    let chal = transcript.squeeze(b"chal_z")?;

    let z_1 = d + chal * y_hat;
    let z_2 = a_hat * ((chal * r_P_hat) + r_beta) + r_delta;

    Ok(InnerProductArgument {
      P_L_vec,
      P_R_vec,
      delta,
      beta,
      z_1,
      z_2,
      _p: Default::default(),
    })
  }

  // from Spartan, notably without the zeroizing buffer
  fn batch_invert(inputs: &mut [E::Scalar]) -> E::Scalar {
    // This code is essentially identical to the FieldElement
    // implementation, and is documented there.  Unfortunately,
    // it's not easy to write it generically, since here we want
    // to use `UnpackedScalar`s internally, and `Scalar`s
    // externally, but there's no corresponding distinction for
    // field elements.

    let n = inputs.len();
    let one = E::Scalar::ONE;

    // Place scratch storage in a Zeroizing wrapper to wipe it when
    // we pass out of scope.
    let mut scratch = vec![one; n];
    //let mut scratch = Zeroizing::new(scratch_vec);

    // Keep an accumulator of all of the previous products
    let mut acc = E::Scalar::ONE;

    // Pass through the input vector, recording the previous
    // products in the scratch space
    for (input, scratch) in inputs.iter().zip(scratch.iter_mut()) {
      *scratch = acc;

      acc *= input;
    }

    // acc is nonzero iff all inputs are nonzero
    debug_assert!(acc != E::Scalar::ZERO);

    // Compute the inverse of all products
    acc = acc.invert().unwrap();

    // We need to return the product of all inverses later
    let ret = acc;

    // Pass through the vector backwards to compute the inverses
    // in place
    for (input, scratch) in inputs.iter_mut().rev().zip(scratch.iter().rev()) {
      let tmp = acc * *input;
      *input = acc * scratch;
      acc = tmp;
    }

    ret
  }

  // copied almost directly from the Spartan method, with some type massaging
  fn verification_scalars(
    &self,
    n: usize,
    transcript: &mut E::TE,
  ) -> Result<(Vec<<E::GE as DlogGroup>::ScalarExt>, Vec<<E::GE as DlogGroup>::ScalarExt>, Vec<<E::GE as DlogGroup>::ScalarExt>), NovaError> {
    let lg_n = self.P_L_vec.len();
    if lg_n >= 32 {
      // 4 billion multiplications should be enough for anyone
      // and this check prevents overflow in 1<<lg_n below.
      return Err(NovaError::ProofVerifyError);
    }
    if n != (1 << lg_n) {
      return Err(NovaError::ProofVerifyError);
    }

    let mut challenges: Vec<<E::GE as DlogGroup>::ScalarExt> = Vec::with_capacity(lg_n);

    // Recompute x_k,...,x_1 based on the proof transcript
    for (P_L, P_R) in self.P_L_vec.iter().zip(self.P_R_vec.iter()) {
      transcript.absorb(b"L", P_L);
      transcript.absorb(b"R", P_R);

      challenges.push(transcript.squeeze(b"challenge_r")?);
    }

    // inverses
    let mut challenges_inv = challenges.clone();
    let prod_all_inv = Self::batch_invert(&mut challenges_inv);

    // squares of challenges & inverses
    for i in 0..lg_n {
      challenges[i] = challenges[i].square();
      challenges_inv[i] = challenges_inv[i].square();
    }
    let challenges_sq = challenges;
    let challenges_inv_sq = challenges_inv;

    // s values inductively
    let mut s = Vec::with_capacity(n);
    s.push(prod_all_inv);
    for i in 1..n {
      let lg_i = (32 - 1 - (i as u32).leading_zeros()) as usize;
      let k = 1 << lg_i;
      // The challenges are stored in "creation order" as [u_k,...,u_1],
      // so u_{lg(i)+1} = is indexed by (lg_n-1) - lg_i
      let u_lg_i_sq = challenges_sq[(lg_n - 1) - lg_i];
      s.push(s[i - k] * u_lg_i_sq);
    }

    Ok((challenges_sq, challenges_inv_sq, s))
  }

  /// verify inner product argument
  pub fn verify(
    &self,
    ck: &CommitmentKey<E>,
    ck_y: &CommitmentKey<E>,
    n: usize,
    U: &InnerProductInstance<E>,
    transcript: &mut E::TE,
  ) -> Result<(), NovaError> {
    transcript.dom_sep(Self::protocol_name());
    if U.a_vec.len() != n
      || n != (1 << self.P_L_vec.len())
      || self.P_L_vec.len() != self.P_R_vec.len()
      || self.P_L_vec.len() >= 32
    {
      return Err(NovaError::InvalidInputLength);
    }

    transcript.absorb(b"comm_x_vec", &U.comm_x_vec);
    transcript.absorb(b"a_vec", &U.a_vec.as_slice());
    transcript.absorb(b"y", &U.comm_y);

    // Scaling to be compatible with Bulletproofs figure 1
    let chal = transcript.squeeze(b"r")?; // sample a random challenge for scaling commitment
    let ck_y = ck_y.scale(&chal);
    let P = U.comm_x_vec + U.comm_y * chal;

    let a_vec = U.a_vec.clone();

    // calculate all the exponent challenges (s) and inverses at once
    let (mut u_sq, mut u_inv_sq, s) = self.verification_scalars(n, transcript)?;

    // do all the exponentiations at once (Hyrax, Fig. 7, step 4, all rounds)
    let c_hat = E::GE::vartime_multiscalar_mul(&s, &CE::<E>::get_gens(ck));
    let a = inner_product(&a_vec[..], &s[..]);

    let mut Ls: Vec<<E::GE as DlogGroup>::AffineExt> = self
      .P_L_vec
      .iter()
      .map(|p| {
        Commitment::<E>::decompress(p)
          .unwrap()
          .reinterpret_as_generator()
      })
      .collect();
    let mut Rs: Vec<<E::GE as DlogGroup>::AffineExt> = self
      .P_R_vec
      .iter()
      .map(|p| {
        Commitment::<E>::decompress(p)
          .unwrap()
          .reinterpret_as_generator()
      })
      .collect();

    Ls.append(&mut Rs);
    Ls.push(P.reinterpret_as_generator());

    u_sq.append(&mut u_inv_sq);
    u_sq.push(E::Scalar::ONE);

    let P_comm = E::GE::vartime_multiscalar_mul(&u_sq, &Ls[..]);

    // Step 3 in Hyrax's Figure 8
    transcript.absorb(b"beta", &self.beta);
    transcript.absorb(b"delta", &self.delta);

    let chal = transcript.squeeze(b"chal_z")?;

    // Step 5 in Hyrax's Figure 8
    // P^(chal*a) * beta^a * delta^1
    let left_hand_side = E::GE::vartime_multiscalar_mul(
      &[(chal * a), a, E::Scalar::ONE],
      &[
        P_comm.preprocessed(),
        Commitment::<E>::decompress(&self.beta)
          .unwrap()
          .reinterpret_as_generator(),
        Commitment::<E>::decompress(&self.delta)
          .unwrap()
          .reinterpret_as_generator(),
      ],
    );

    // c_hat^z1 * g^(a*z1) * h^z2
    let right_hand_side = E::GE::vartime_multiscalar_mul(
      &[self.z_1, (self.z_1 * a), self.z_2],
      &[
        c_hat.preprocessed(),
        CE::<E>::get_gens(&ck_y)[0].clone(),
        CE::<E>::get_blinding_gen(&ck_y),
      ],
    );

    if left_hand_side == right_hand_side {
      Ok(())
    } else {
      println!("Invalid IPA! whoops");
      Err(NovaError::InvalidIPA)
    }
  }
}

#[cfg(test)]
mod test {
  use crate::provider::ipa_pc::EvaluationEngine;
  // use crate::provider::util::test_utils::prove_verify_from_num_vars;
  use crate::provider::GrumpkinEngine;

  // #[test]
  // fn test_multiple_polynomial_size() {
  //   for num_vars in [4, 5, 6] {
  //     prove_verify_from_num_vars::<_, EvaluationEngine<GrumpkinEngine>>(num_vars);
  //   }
  // }
}
