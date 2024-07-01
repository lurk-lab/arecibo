//! This module defines nizk proofs
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
use crate::errors::NovaError;
use crate::traits::{
  commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
  Engine, TranscriptEngineTrait,
};
use crate::{Commitment, CommitmentKey, CompressedCommitment, CE};
use ff::Field;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

/// KnowledgeProof
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KnowledgeProof<E: Engine> {
  alpha: CompressedCommitment<E>,
  z1: E::Scalar,
  z2: E::Scalar,
}

/// EqualityProof
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct EqualityProof<E: Engine> {
  /// alpha
  pub alpha: CompressedCommitment<E>,
  /// z
  pub z: E::Scalar,
}

/// ProductProof
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProductProof<E: Engine> {
  alpha: CompressedCommitment<E>,
  beta: CompressedCommitment<E>,
  delta: CompressedCommitment<E>,
  z: [E::Scalar; 5],
}

/// DocProductProof
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DotProductProof<E: Engine> {
  delta: CompressedCommitment<E>,
  beta: CompressedCommitment<E>,
  z: Vec<E::Scalar>,
  z_delta: E::Scalar,
  z_beta: E::Scalar,
}

/// KnowledgeProof
impl<E: Engine> KnowledgeProof<E> {
  /// protocol name
  pub fn protocol_name() -> &'static [u8] {
    b"knowledge proof"
  }

  /// prove
  pub fn prove(
    ck_n: &CommitmentKey<E>,
    transcript: &mut E::TE,
    x: &E::Scalar,
    r: &E::Scalar,
  ) -> Result<(KnowledgeProof<E>, CompressedCommitment<E>), NovaError> {
    transcript.dom_sep(Self::protocol_name());

    // produce two random scalars
    let t1 = E::Scalar::random(&mut OsRng);
    let t2 = E::Scalar::random(&mut OsRng);

    let C = CE::<E>::commit(ck_n, &[*x], r).compress();
    transcript.absorb(b"C", &C);

    let alpha = CE::<E>::commit(ck_n, &[t1], &t2).compress();
    transcript.absorb(b"alpha", &alpha);

    let c = transcript.squeeze(b"c")?;

    let z1 = *x * c + t1;
    let z2 = *r * c + t2;

    Ok((Self { alpha, z1, z2 }, C))
  }

  /// verify
  pub fn verify(
    &self,
    ck_n: &CommitmentKey<E>,
    transcript: &mut E::TE,
    C: &CompressedCommitment<E>,
  ) -> Result<(), NovaError> {
    transcript.dom_sep(Self::protocol_name());
    transcript.absorb(b"C", C);
    transcript.absorb(b"alpha", &self.alpha);

    let c = transcript.squeeze(b"c")?;

    let lhs = CE::<E>::commit(ck_n, &[self.z1], &self.z2).compress();
    let rhs =
      (Commitment::<E>::decompress(C)? * c + Commitment::<E>::decompress(&self.alpha)?).compress();

    if lhs == rhs {
      Ok(())
    } else {
      Err(NovaError::InvalidZkKnowledgeProof)
    }
  }
}

/// EqualityProof
impl<E: Engine> EqualityProof<E> {
  /// protocol name
  pub fn protocol_name() -> &'static [u8] {
    b"equality proof"
  }

  /// prove
  pub fn prove(
    ck_n: &CommitmentKey<E>,
    transcript: &mut E::TE,
    v1: &E::Scalar,
    s1: &E::Scalar,
    v2: &E::Scalar,
    s2: &E::Scalar,
  ) -> Result<
    (
      EqualityProof<E>,
      CompressedCommitment<E>,
      CompressedCommitment<E>,
    ),
    NovaError,
  > {
    transcript.dom_sep(Self::protocol_name());

    // produce a random scalar
    let r = E::Scalar::random(&mut OsRng);

    let C1 = CE::<E>::commit(ck_n, &[*v1], s1).compress();
    transcript.absorb(b"C1", &C1);

    let C2 = CE::<E>::commit(ck_n, &[*v2], s2).compress();
    transcript.absorb(b"C2", &C2);

    let alpha = CE::<E>::commit(ck_n, &[E::Scalar::ZERO], &r).compress(); // h^r
    transcript.absorb(b"alpha", &alpha);

    let c = transcript.squeeze(b"c")?;

    let z = c * (*s1 - *s2) + r;

    Ok((Self { alpha, z }, C1, C2))
  }

  /// verify
  pub fn verify(
    &self,
    gens_n: &CommitmentKey<E>,
    transcript: &mut E::TE,
    C1: &CompressedCommitment<E>,
    C2: &CompressedCommitment<E>,
  ) -> Result<(), NovaError> {
    transcript.dom_sep(Self::protocol_name());
    transcript.absorb(b"C1", C1);
    transcript.absorb(b"C2", C2);
    transcript.absorb(b"alpha", &self.alpha);

    let c = transcript.squeeze(b"c")?;

    let rhs = {
      let C = Commitment::<E>::decompress(C1)? - Commitment::<E>::decompress(C2)?;
      (C * c + Commitment::<E>::decompress(&self.alpha)?).compress()
    };

    let lhs = CE::<E>::commit(gens_n, &[E::Scalar::ZERO], &self.z).compress(); // h^z

    if lhs == rhs {
      Ok(())
    } else {
      Err(NovaError::InvalidZkEqualityProof)
    }
  }
}

/// product proof
impl<E: Engine> ProductProof<E> {
  /// protocol name
  pub fn protocol_name() -> &'static [u8] {
    b"product proof"
  }

  /// prove
  pub fn prove(
    ck_n: &CommitmentKey<E>,
    transcript: &mut E::TE,
    x: &E::Scalar,
    rX: &E::Scalar,
    y: &E::Scalar,
    rY: &E::Scalar,
    z: &E::Scalar,
    rZ: &E::Scalar,
  ) -> Result<
    (
      ProductProof<E>,
      CompressedCommitment<E>,
      CompressedCommitment<E>,
      CompressedCommitment<E>,
    ),
    NovaError,
  > {
    transcript.dom_sep(Self::protocol_name());

    // produce 5 random scalars
    let b1 = E::Scalar::random(&mut OsRng);
    let b2 = E::Scalar::random(&mut OsRng);
    let b3 = E::Scalar::random(&mut OsRng);
    let b4 = E::Scalar::random(&mut OsRng);
    let b5 = E::Scalar::random(&mut OsRng);

    let X = CE::<E>::commit(ck_n, &[*x], rX).compress();
    transcript.absorb(b"X", &X);

    let Y = CE::<E>::commit(ck_n, &[*y], rY).compress();
    transcript.absorb(b"Y", &Y);

    let Z = CE::<E>::commit(ck_n, &[*z], rZ).compress();
    transcript.absorb(b"Z", &Z);

    let alpha = CE::<E>::commit(ck_n, &[b1], &b2).compress();
    transcript.absorb(b"alpha", &alpha);

    let beta = CE::<E>::commit(ck_n, &[b3], &b4).compress();
    transcript.absorb(b"beta", &beta);

    let delta = {
      let h_to_b5 = CE::<E>::commit(ck_n, &[E::Scalar::ZERO], &b5); // h^b5
      (Commitment::<E>::decompress(&X)? * b3 + h_to_b5).compress() // X^b3*h^b5
    };

    transcript.absorb(b"delta", &delta);

    let c = transcript.squeeze(b"c")?;

    let z1 = b1 + c * *x;
    let z2 = b2 + c * *rX;
    let z3 = b3 + c * *y;
    let z4 = b4 + c * *rY;
    let z5 = b5 + c * (*rZ - *rX * *y);
    let z = [z1, z2, z3, z4, z5];

    Ok((
      Self {
        alpha,
        beta,
        delta,
        z,
      },
      X,
      Y,
      Z,
    ))
  }

  /// check_equality
  fn check_equality(
    P: &CompressedCommitment<E>,
    X: &CompressedCommitment<E>,
    c: &E::Scalar,
    ck_n: &CommitmentKey<E>,
    z1: &E::Scalar,
    z2: &E::Scalar,
  ) -> Result<bool, NovaError> {
    let lhs = (Commitment::<E>::decompress(P)? + Commitment::<E>::decompress(X)? * *c).compress();
    let rhs = CE::<E>::commit(ck_n, &[*z1], z2).compress();

    Ok(lhs == rhs)
  }

  /// verify
  pub fn verify(
    &self,
    ck_n: &CommitmentKey<E>,
    transcript: &mut E::TE,
    X: &CompressedCommitment<E>,
    Y: &CompressedCommitment<E>,
    Z: &CompressedCommitment<E>,
  ) -> Result<(), NovaError> {
    transcript.dom_sep(Self::protocol_name());

    transcript.absorb(b"X", X);
    transcript.absorb(b"Y", Y);
    transcript.absorb(b"Z", Z);
    transcript.absorb(b"alpha", &self.alpha);
    transcript.absorb(b"beta", &self.beta);
    transcript.absorb(b"delta", &self.delta);

    let z1 = self.z[0];
    let z2 = self.z[1];
    let z3 = self.z[2];
    let z4 = self.z[3];
    let z5 = self.z[4];

    let c = transcript.squeeze(b"c")?;

    let res = ProductProof::<E>::check_equality(&self.alpha, X, &c, ck_n, &z1, &z2)?
      && ProductProof::<E>::check_equality(&self.beta, Y, &c, ck_n, &z3, &z4)?;

    let res2 = {
      let lhs = (Commitment::<E>::decompress(&self.delta)? + Commitment::<E>::decompress(Z)? * c)
        .compress();

      let h_to_z5 = CE::<E>::commit(ck_n, &[E::Scalar::ZERO], &z5); // h^z5
      let rhs = (Commitment::<E>::decompress(X)? * z3 + h_to_z5).compress(); // X^z3*h^z5
      lhs == rhs
    };

    if res && res2 {
      Ok(())
    } else {
      Err(NovaError::InvalidZkProductProof)
    }
  }
}

/// DotProductProof
impl<E: Engine> DotProductProof<E> {
  /// protocol name
  pub fn protocol_name() -> &'static [u8] {
    b"dot product proof"
  }

  /// comppute dot product
  pub fn compute_dotproduct(a: &[E::Scalar], b: &[E::Scalar]) -> E::Scalar {
    assert_eq!(a.len(), b.len());
    let mut result = E::Scalar::ZERO;

    for i in 0..a.len() {
      result += a[i] * b[i];
    }

    result
  }

  /// prove
  pub fn prove(
    ck_1: &CommitmentKey<E>, // generator of size 1
    ck_n: &CommitmentKey<E>, // generators of size n
    transcript: &mut E::TE,
    x_vec: &[E::Scalar],
    blind_x: &E::Scalar,
    a_vec: &[E::Scalar],
    y: &E::Scalar,
    blind_y: &E::Scalar,
  ) -> Result<(Self, CompressedCommitment<E>, CompressedCommitment<E>), NovaError> {
    transcript.dom_sep(Self::protocol_name());

    let n = x_vec.len();
    assert_eq!(x_vec.len(), a_vec.len());
    assert_eq!(ck_n.length(), a_vec.len());
    assert_eq!(ck_1.length(), 1);

    // produce randomness for the proofs
    let d_vec = (0..n)
      .map(|_i| E::Scalar::random(&mut OsRng))
      .collect::<Vec<E::Scalar>>();

    let r_delta = E::Scalar::random(&mut OsRng);
    let r_beta = E::Scalar::random(&mut OsRng);

    let Cx = CE::<E>::commit(ck_n, x_vec, blind_x).compress();
    transcript.absorb(b"Cx", &Cx);

    let Cy = CE::<E>::commit(ck_1, &[*y], blind_y).compress();
    transcript.absorb(b"Cy", &Cy);

    transcript.absorb(b"a", &a_vec);

    let delta = CE::<E>::commit(ck_n, &d_vec, &r_delta).compress();
    transcript.absorb(b"delta", &delta);

    let dotproduct_a_d = DotProductProof::<E>::compute_dotproduct(a_vec, &d_vec);

    let beta = CE::<E>::commit(ck_1, &[dotproduct_a_d], &r_beta).compress();
    transcript.absorb(b"beta", &beta);

    let c = transcript.squeeze(b"c")?;

    let z = (0..d_vec.len())
      .map(|i| c * x_vec[i] + d_vec[i])
      .collect::<Vec<E::Scalar>>();

    let z_delta = c * blind_x + r_delta;
    let z_beta = c * blind_y + r_beta;

    Ok((
      DotProductProof {
        delta,
        beta,
        z,
        z_delta,
        z_beta,
      },
      Cx,
      Cy,
    ))
  }

  /// verify
  pub fn verify(
    &self,
    ck_1: &CommitmentKey<E>, // generator of size 1
    ck_n: &CommitmentKey<E>, // generator of size n
    transcript: &mut E::TE,
    a_vec: &[E::Scalar],
    Cx: &CompressedCommitment<E>,
    Cy: &CompressedCommitment<E>,
  ) -> Result<(), NovaError> {
    assert_eq!(ck_n.length(), a_vec.len());
    assert_eq!(ck_1.length(), 1);

    transcript.dom_sep(Self::protocol_name());

    transcript.absorb(b"Cx", Cx);
    transcript.absorb(b"Cy", Cy);
    transcript.absorb(b"a", &a_vec);
    transcript.absorb(b"delta", &self.delta);
    transcript.absorb(b"beta", &self.beta);

    let c = transcript.squeeze(b"c")?;

    let mut result = Commitment::<E>::decompress(Cx)? * c
      + Commitment::<E>::decompress(&self.delta)?
      == CE::<E>::commit(ck_n, &self.z, &self.z_delta);

    let dotproduct_z_a = DotProductProof::<E>::compute_dotproduct(&self.z, a_vec);
    result &= Commitment::<E>::decompress(Cy)? * c + Commitment::<E>::decompress(&self.beta)?
      == CE::<E>::commit(ck_1, &[dotproduct_z_a], &self.z_beta);

    if result {
      Ok(())
    } else {
      Err(NovaError::InvalidZkDotProductProof)
    }
  }
}
