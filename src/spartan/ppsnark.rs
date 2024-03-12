//! This module implements `RelaxedR1CSSNARK` traits using a spark-based approach to prove evaluations of
//! sparse multilinear polynomials involved in Spartan's sum-check protocol, thereby providing a preprocessing SNARK
//! The verifier in this preprocessing SNARK maintains a commitment to R1CS matrices. This is beneficial when using a
//! polynomial commitment scheme in which the verifier's costs is succinct.
//! This code includes experimental optimizations to reduce runtimes and proof sizes.
use crate::{
  r1cs::R1CSShape,
  traits::{commitment::CommitmentEngineTrait, Engine, TranscriptReprTrait},
  Commitment, CommitmentKey,
};
use core::cmp::max;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that holds `R1CSShape` in a form amenable to memory checking
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSShapeSparkRepr<E: Engine> {
  pub(in crate::spartan) N: usize, // size of the vectors

  // dense representation
  pub(in crate::spartan) row: Vec<E::Scalar>,
  pub(in crate::spartan) col: Vec<E::Scalar>,
  pub(in crate::spartan) val_A: Vec<E::Scalar>,
  pub(in crate::spartan) val_B: Vec<E::Scalar>,
  pub(in crate::spartan) val_C: Vec<E::Scalar>,

  // timestamp polynomials
  pub(in crate::spartan) ts_row: Vec<E::Scalar>,
  pub(in crate::spartan) ts_col: Vec<E::Scalar>,
}

/// A type that holds a commitment to a sparse polynomial
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSShapeSparkCommitment<E: Engine> {
  pub(in crate::spartan) N: usize, // size of each vector

  // commitments to the dense representation
  pub(in crate::spartan) comm_row: Commitment<E>,
  pub(in crate::spartan) comm_col: Commitment<E>,
  pub(in crate::spartan) comm_val_A: Commitment<E>,
  pub(in crate::spartan) comm_val_B: Commitment<E>,
  pub(in crate::spartan) comm_val_C: Commitment<E>,

  // commitments to the timestamp polynomials
  pub(in crate::spartan) comm_ts_row: Commitment<E>,
  pub(in crate::spartan) comm_ts_col: Commitment<E>,
}

impl<E: Engine> TranscriptReprTrait<E::GE> for R1CSShapeSparkCommitment<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_row,
      self.comm_col,
      self.comm_val_A,
      self.comm_val_B,
      self.comm_val_C,
      self.comm_ts_row,
      self.comm_ts_col,
    ]
    .as_slice()
    .to_transcript_bytes()
  }
}

impl<E: Engine> R1CSShapeSparkRepr<E> {
  /// represents `R1CSShape` in a Spark-friendly format amenable to memory checking
  pub fn new(S: &R1CSShape<E>) -> Self {
    let N = {
      let total_nz = S.A.len() + S.B.len() + S.C.len();
      max(total_nz, max(2 * S.num_vars, S.num_cons)).next_power_of_two()
    };

    // we make col lookup into the last entry of z, so we commit to zeros
    let (mut row, mut col, mut val_A, mut val_B, mut val_C) = (
      vec![0; N],
      vec![N - 1; N],
      vec![E::Scalar::ZERO; N],
      vec![E::Scalar::ZERO; N],
      vec![E::Scalar::ZERO; N],
    );

    for (i, entry) in S.A.iter().enumerate() {
      let (r, c, v) = entry;
      row[i] = r;
      col[i] = c;
      val_A[i] = v;
    }

    let b_offset = S.A.len();
    for (i, entry) in S.B.iter().enumerate() {
      let (r, c, v) = entry;
      row[b_offset + i] = r;
      col[b_offset + i] = c;
      val_B[b_offset + i] = v;
    }

    let c_offset = S.A.len() + S.B.len();
    for (i, entry) in S.C.iter().enumerate() {
      let (r, c, v) = entry;
      row[c_offset + i] = r;
      col[c_offset + i] = c;
      val_C[c_offset + i] = v;
    }

    // timestamp calculation routine
    let timestamp_calc = |num_ops: usize, num_cells: usize, addr_trace: &[usize]| -> Vec<usize> {
      let mut ts = vec![0usize; num_cells];

      assert!(num_ops >= addr_trace.len());
      for addr in addr_trace {
        assert!(*addr < num_cells);
        ts[*addr] += 1;
      }
      ts
    };

    // timestamp polynomials for row
    let (ts_row, ts_col) =
      rayon::join(|| timestamp_calc(N, N, &row), || timestamp_calc(N, N, &col));

    // a routine to turn a vector of usize into a vector scalars
    let to_vec_scalar = |v: &[usize]| -> Vec<E::Scalar> {
      v.iter()
        .map(|x| E::Scalar::from(*x as u64))
        .collect::<Vec<_>>()
    };

    Self {
      N,

      // dense representation
      row: to_vec_scalar(&row),
      col: to_vec_scalar(&col),
      val_A,
      val_B,
      val_C,

      // timestamp polynomials
      ts_row: to_vec_scalar(&ts_row),
      ts_col: to_vec_scalar(&ts_col),
    }
  }

  pub(in crate::spartan) fn commit(&self, ck: &CommitmentKey<E>) -> R1CSShapeSparkCommitment<E> {
    let comm_vec: Vec<Commitment<E>> = [
      &self.row,
      &self.col,
      &self.val_A,
      &self.val_B,
      &self.val_C,
      &self.ts_row,
      &self.ts_col,
    ]
    .par_iter()
    .map(|v| E::CE::commit(ck, v))
    .collect();

    R1CSShapeSparkCommitment {
      N: self.row.len(),
      comm_row: comm_vec[0],
      comm_col: comm_vec[1],
      comm_val_A: comm_vec[2],
      comm_val_B: comm_vec[3],
      comm_val_C: comm_vec[4],
      comm_ts_row: comm_vec[5],
      comm_ts_col: comm_vec[6],
    }
  }
}

/// A type that represents the prover's key
pub type ProverKey<E, EE> = crate::spartan::batched_ppsnark::ProverKey<E, EE>;
/// A type that represents the verifier's key
pub type VerifierKey<E, EE> = crate::spartan::batched_ppsnark::VerifierKey<E, EE>;

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
pub type RelaxedR1CSSNARK<E, EE> = crate::spartan::batched_ppsnark::BatchedRelaxedR1CSSNARK<E, EE>;
