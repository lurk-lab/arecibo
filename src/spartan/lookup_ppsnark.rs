//! This module implements `RelaxedR1CSSNARK` traits using a spark-based approach to prove evaluations of
//! sparse multilinear polynomials involved in Spartan's sum-check protocol, thereby providing a preprocessing SNARK
//! The verifier in this preprocessing SNARK maintains a commitment to R1CS matrices. This is beneficial when using a
//! polynomial commitment scheme in which the verifier's costs is succinct.
//! This code includes experimental optimizations to reduce runtimes and proof sizes.
use crate::gadgets::lookup::Lookup;
use crate::spartan::sumcheck::engine::NaturalNumVec;
use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  spartan::{
    math::Math,
    polys::{
      eq::EqPolynomial,
      identity::IdentityPolynomial,
      multilinear::MultilinearPolynomial,
      power::PowPolynomial,
      univariate::{CompressedUniPoly, UniPoly},
    },
    powers,
    sumcheck::{
      engine::{
        InnerSumcheckInstance,
        MemorySumcheckInstance,
        OuterSumcheckInstance,
        SumcheckEngine,
        LookupSumcheckInstance,
        WitnessBoundSumcheck,
      },
      SumcheckProof,
    },
    PolyEvalInstance, PolyEvalWitness,
  },
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    evaluation::EvaluationEngineTrait,
    snark::{DigestHelperTrait, RelaxedR1CSLookupSNARKTrait},
    Engine, TranscriptEngineTrait, TranscriptReprTrait,
  },
  zip_with, Commitment, CommitmentKey, CompressedCommitment,
};
use core::cmp::max;
use ff::Field;
use itertools::Itertools as _;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::Mutex;
use crate::traits::AbsorbInROTrait;
use crate::ROTrait;
use crate::scalar_as_base;
use crate::constants::NUM_CHALLENGE_BITS;

use super::polys::{masked_eq::MaskedEqPolynomial, multilinear::SparsePolynomial};

fn padded<E: Engine>(v: &[E::Scalar], n: usize, e: &E::Scalar) -> Vec<E::Scalar> {
  let mut v_padded = vec![*e; n];
  v_padded[..v.len()].copy_from_slice(v);
  v_padded
}

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

  // lookup
  pub(in crate::spartan) init_values_lookup: Vec<E::Scalar>,
  pub(in crate::spartan) const_ts_lookup: Vec<E::Scalar>,
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

  // commitments to lookup table
  pub(in crate::spartan) comm_init_values_lookup: Commitment<E>,
  pub(in crate::spartan) comm_const_ts_lookup: Commitment<E>,
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
  pub fn new(S: &R1CSShape<E>, initial_table: &Lookup<E::Scalar>) -> Self
    where
    <E as Engine>::Scalar: Ord,
  {

    let N = {
      let total_nz = S.A.len() + S.B.len() + S.C.len();
      max(
        max(total_nz, max(2 * S.num_vars, S.num_cons)),
        initial_table.table_size(),
      )
      .next_power_of_two()
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

    let mut initial_table = initial_table.clone();
    initial_table.padding(N);
    // lookup
    let init_values_lookup: Vec<_> = initial_table
      .into_iter()
      .map(|(_, (value, _))| *value)
      .collect();
    let const_ts_lookup = vec![E::Scalar::from(1u64); N];

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

      // lookup
      init_values_lookup,
      const_ts_lookup,
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
      &self.init_values_lookup,
      &self.const_ts_lookup,
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
      // lookup
      comm_init_values_lookup: comm_vec[7],
      comm_const_ts_lookup: comm_vec[8],
    }
  }

  // computes evaluation oracles
  fn evaluation_oracles(
    &self,
    S: &R1CSShape<E>,
    r_x: &E::Scalar,
    z: &[E::Scalar],
  ) -> (
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
  ) {
    let mem_row = PowPolynomial::new(r_x, self.N.log_2()).evals();
    let mem_col = padded::<E>(z, self.N, &E::Scalar::ZERO);

    let (L_row, L_col) = {
      let mut L_row = vec![mem_row[0]; self.N]; // we place mem_row[0] since resized row is appended with 0s
      let mut L_col = vec![mem_col[self.N - 1]; self.N]; // we place mem_col[N-1] since resized col is appended with N-1

      for (i, (val_r, val_c)) in S
        .A
        .iter()
        .chain(S.B.iter())
        .chain(S.C.iter())
        .map(|(r, c, _)| (mem_row[r], mem_col[c]))
        .enumerate()
      {
        L_row[i] = val_r;
        L_col[i] = val_c;
      }
      (L_row, L_col)
    };

    (mem_row, mem_col, L_row, L_col)
  }
}

/// A type that represents the prover's key
#[derive(Debug, Clone)]
pub struct ProverKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  pk_ee: EE::ProverKey,
  S_repr: R1CSShapeSparkRepr<E>,
  S_comm: R1CSShapeSparkCommitment<E>,
  vk_digest: E::Scalar, // digest of verifier's key
}

/// A type that represents the verifier's key
#[derive(Debug, Clone, Serialize)]
#[serde(bound = "EE::VerifierKey: Serialize")]
pub struct VerifierKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  num_cons: usize,
  num_vars: usize,
  vk_ee: EE::VerifierKey,
  S_comm: R1CSShapeSparkCommitment<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> SimpleDigestible for VerifierKey<E, EE> where
  EE::VerifierKey: Serialize
{
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSSNARK<E: Engine, EE: EvaluationEngineTrait<E>> {
  // commitment to oracles: the first three are for Az, Bz, Cz,
  // and the last two are for memory reads
  comm_Az: CompressedCommitment<E>,
  comm_Bz: CompressedCommitment<E>,
  comm_Cz: CompressedCommitment<E>,
  comm_L_row: CompressedCommitment<E>,
  comm_L_col: CompressedCommitment<E>,

  // commitments to aid the memory checks
  comm_t_plus_r_inv_row: CompressedCommitment<E>,
  comm_w_plus_r_inv_row: CompressedCommitment<E>,
  comm_t_plus_r_inv_col: CompressedCommitment<E>,
  comm_w_plus_r_inv_col: CompressedCommitment<E>,

  // claims about Az, Bz, and Cz polynomials
  eval_Az_at_tau: E::Scalar,
  eval_Bz_at_tau: E::Scalar,
  eval_Cz_at_tau: E::Scalar,

  // sum-check
  sc: SumcheckProof<E>,

  // claims from the end of sum-check
  eval_Az: E::Scalar,
  eval_Bz: E::Scalar,
  eval_Cz: E::Scalar,
  eval_E: E::Scalar,
  eval_L_row: E::Scalar,
  eval_L_col: E::Scalar,
  eval_val_A: E::Scalar,
  eval_val_B: E::Scalar,
  eval_val_C: E::Scalar,

  eval_W: E::Scalar,

  eval_t_plus_r_inv_row: E::Scalar,
  eval_row: E::Scalar, // address
  eval_w_plus_r_inv_row: E::Scalar,
  eval_ts_row: E::Scalar,

  eval_t_plus_r_inv_col: E::Scalar,
  eval_col: E::Scalar, // address
  eval_w_plus_r_inv_col: E::Scalar,
  eval_ts_col: E::Scalar,

  // lookup
  comm_t_plus_r_inv_lookup: CompressedCommitment<E>,
  comm_w_plus_r_inv_lookup: CompressedCommitment<E>,
  comm_final_value: CompressedCommitment<E>,
  comm_final_ts: CompressedCommitment<E>,
  eval_t_plus_r_inv_lookup: E::Scalar,
  eval_w_plus_r_inv_lookup: E::Scalar,
  eval_ts_lookup: E::Scalar,
  eval_init_value_lookup: E::Scalar,
  eval_final_value_lookup: E::Scalar,
  eval_final_ts_lookup: E::Scalar,

  // a PCS evaluation argument
  eval_arg: EE::EvaluationArgument,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> RelaxedR1CSSNARK<E, EE> {
  fn prove_helper<T1, T2, T3, T4, T5>(
    mem: &mut T1,
    outer: &mut T2,
    inner: &mut T3,
    witness: &mut T4,
    lookup: &mut T5,
    transcript: &mut E::TE,
  ) -> Result<
    (
      SumcheckProof<E>,
      Vec<E::Scalar>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
    ),
    NovaError,
  >
  where
    T1: SumcheckEngine<E>,
    T2: SumcheckEngine<E>,
    T3: SumcheckEngine<E>,
    T4: SumcheckEngine<E>,
    T5: SumcheckEngine<E>,
  {
    // sanity checks
    assert_eq!(mem.size(), outer.size());
    assert_eq!(mem.size(), inner.size());
    assert_eq!(mem.size(), witness.size());
    assert_eq!(mem.size(), lookup.size());

    assert_eq!(mem.degree(), outer.degree());
    assert_eq!(mem.degree(), inner.degree());
    assert_eq!(mem.degree(), witness.degree());
    assert_eq!(mem.degree(), lookup.degree());

    // these claims are already added to the transcript, so we do not need to add
    let claims = mem
      .initial_claims()
      .into_iter()
      .chain(outer.initial_claims())
      .chain(inner.initial_claims())
      .chain(witness.initial_claims())
      .chain(lookup.initial_claims())
      .collect::<Vec<E::Scalar>>();

    let s = transcript.squeeze(b"r")?;
    let coeffs = powers(&s, claims.len());

    // compute the joint claim
    let claim = zip_with!(iter, (claims, coeffs), |c_1, c_2| *c_1 * c_2).sum();

    let mut e = claim;
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();
    let num_rounds = mem.size().log_2();
    
    for _ in 0..num_rounds {

      let (((evals_mem, evals_outer), (evals_inner, evals_witness)), evals_lookup) = rayon::join(
        || {
          rayon::join(
            || rayon::join(|| mem.evaluation_points(), || outer.evaluation_points()),
            || rayon::join(|| inner.evaluation_points(), || witness.evaluation_points()),
          )
        },
        || lookup.evaluation_points(),
      );

      let evals: Vec<Vec<E::Scalar>> = evals_mem
        .into_iter()
        .chain(evals_outer.into_iter())
        .chain(evals_inner.into_iter())
        .chain(evals_witness.into_iter())
        .chain(evals_lookup.into_iter())
        .collect::<Vec<Vec<E::Scalar>>>();
        
      assert_eq!(evals.len(), claims.len());

      let evals_combined_0 = (0..evals.len()).map(|i| evals[i][0] * coeffs[i]).sum();
      let evals_combined_2 = (0..evals.len()).map(|i| evals[i][1] * coeffs[i]).sum();
      let evals_combined_3 = (0..evals.len()).map(|i| evals[i][2] * coeffs[i]).sum();

      let evals = vec![
        evals_combined_0,
        e - evals_combined_0,
        evals_combined_2,
        evals_combined_3,
      ];
      let poly = UniPoly::from_evals(&evals);

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      // derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);

      let _ = rayon::join(
        || {
          rayon::join(
            || rayon::join(|| mem.bound(&r_i), || outer.bound(&r_i)),
            || rayon::join(|| inner.bound(&r_i), || witness.bound(&r_i)),
          )
        },
        || lookup.bound(&r_i),
      );

      e = poly.evaluate(&r_i);
      cubic_polys.push(poly.compress());
    }

    let mem_claims = mem.final_claims();
    let outer_claims = outer.final_claims();
    let inner_claims = inner.final_claims();
    let witness_claims = witness.final_claims();
    let lookup_claims = lookup.final_claims();

    Ok((
      SumcheckProof::new(cubic_polys),
      r,
      mem_claims,
      outer_claims,
      inner_claims,
      witness_claims,
      lookup_claims,
    ))
  }

  fn verify_challenge<E2: Engine>(
    comm_final_value: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    comm_final_ts: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    lookup_intermediate_gamma: E::Scalar,
    challenges: (E::Scalar, E::Scalar),
  ) -> Result<(), NovaError>
  where
    E: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E as Engine>::Scalar>,
  {
    // verify fingerprint challenge
    let (lookup_r, lookup_gamma) = challenges;

    let ro_consts =
      <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants::default();

    let mut hasher = <E as Engine>::RO::new(ro_consts.clone(), 7);
    let lookup_intermediate_gamma: E2::Scalar = scalar_as_base::<E>(lookup_intermediate_gamma);
    hasher.absorb(lookup_intermediate_gamma);
    comm_final_value.absorb_in_ro(&mut hasher);
    comm_final_ts.absorb_in_ro(&mut hasher);
    let computed_gamma = hasher.squeeze(NUM_CHALLENGE_BITS);
    if lookup_gamma != computed_gamma {
      return Err(NovaError::InvalidMultisetProof);
    }
    let mut hasher = <E as Engine>::RO::new(ro_consts, 1);
    hasher.absorb(scalar_as_base::<E>(computed_gamma));
    let computed_r = hasher.squeeze(NUM_CHALLENGE_BITS);
    if lookup_r != computed_r {
      return Err(NovaError::InvalidMultisetProof);
    }
    Ok(())
  }
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> VerifierKey<E, EE> {
  fn new(
    num_cons: usize,
    num_vars: usize,
    S_comm: R1CSShapeSparkCommitment<E>,
    vk_ee: EE::VerifierKey,
  ) -> Self {
    Self {
      num_cons,
      num_vars,
      S_comm,
      vk_ee,
      digest: Default::default(),
    }
  }
}
impl<E: Engine, EE: EvaluationEngineTrait<E>> DigestHelperTrait<E> for VerifierKey<E, EE> {
  /// Returns the digest of the verifier's key
  fn digest(&self) -> E::Scalar {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::new(self);
        dc.digest()
      })
      .cloned()
      .expect("Failure to retrieve digest!")
  }
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> RelaxedR1CSLookupSNARKTrait<E> 
  for RelaxedR1CSSNARK<E, EE>
where
  <E as Engine>::Scalar: Ord,
{
  type ProverKey = ProverKey<E, EE>;
  type VerifierKey = VerifierKey<E, EE>;

  fn ck_floor() -> Box<dyn for<'a> Fn(&'a R1CSShape<E>) -> usize> {
    Box::new(|shape: &R1CSShape<E>| -> usize {
      // the commitment key should be large enough to commit to the R1CS matrices
      shape.A.len() + shape.B.len() + shape.C.len()
    })
  }

  fn setup(
    ck: Arc<CommitmentKey<E>>,
    S: &R1CSShape<E>,
    initial_table: &Lookup<E::Scalar>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError>
    where
    <E as Engine>::Scalar: Ord,
  {
    // check the provided commitment key meets minimal requirements
    if ck.length() < Self::ck_floor()(S) {
      return Err(NovaError::InvalidCommitmentKeyLength);
    }
    let (pk_ee, vk_ee) = EE::setup(ck.clone());

    // pad the R1CS matrices
    let S = S.pad();

    let S_repr = R1CSShapeSparkRepr::new(&S, initial_table);
    let S_comm = S_repr.commit(&*ck);

    let vk = VerifierKey::new(S.num_cons, S.num_vars, S_comm.clone(), vk_ee);

    let pk = ProverKey {
      pk_ee,
      S_repr,
      S_comm,
      vk_digest: vk.digest(),
    };

    Ok((pk, vk))
  }

  
  /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
  #[tracing::instrument(skip_all, name = "LOOKUP_PPSNARK::prove")]
  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: &R1CSShape<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
    challenges: (E::Scalar, E::Scalar),
    RW_acc: E::Scalar,
    mut initial_table: Lookup<E::Scalar>,
    mut final_table: Lookup<E::Scalar>,
  ) -> Result<Self, NovaError> {
    // pad the R1CSShape
    let S = S.pad();
    // sanity check that R1CSShape has all required size characteristics
    assert!(S.is_regular_shape());

    let W = W.pad(&S); // pad the witness
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // chalenges for lookup
    let (lookup_r, lookup_gamma) = challenges;

    // append the verifier key (which includes commitment to R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    // add lookup table to transcript
    transcript.absorb(b"RW_acc", &RW_acc);
    transcript.absorb(b"r", &lookup_r);
    transcript.absorb(b"gamma", &lookup_gamma);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let z = [W.W.clone(), vec![U.u], U.X.clone()].concat();

    // compute Az, Bz, Cz
    let (mut Az, mut Bz, mut Cz) = S.multiply_vec(&z)?;

    // commit to Az, Bz, Cz
    let (comm_Az, (comm_Bz, comm_Cz)) = rayon::join(
      || E::CE::commit(ck, &Az),
      || rayon::join(|| E::CE::commit(ck, &Bz), || E::CE::commit(ck, &Cz)),
    );

    transcript.absorb(b"c", &[comm_Az, comm_Bz, comm_Cz].as_slice());


    // commit general table init/final value
    // padding init table/final table to same length as N
    initial_table.padding(pk.S_repr.N);
    final_table.padding(pk.S_repr.N);
    let init_values: Vec<<E as Engine>::Scalar> = initial_table
      .into_iter()
      .map(|(_, (value, _))| *value)
      .collect();
    let (final_values, final_ts): (Vec<_>, Vec<_>) = final_table.values().copied().unzip();
    let (comm_final_value, comm_final_ts) = rayon::join(
      || E::CE::commit(ck, &final_values),
      || E::CE::commit(ck, &final_ts),
    );

    // absorb lookup table value/ts commitment
    transcript.absorb(b"c", &vec![comm_final_value, comm_final_ts].as_slice());

    // number of rounds of sum-check
    let num_rounds_sc = pk.S_repr.N.log_2();
    let tau = transcript.squeeze(b"t")?;
    let tau_coords = PowPolynomial::new(&tau, num_rounds_sc).coordinates();

    // (1) send commitments to Az, Bz, and Cz along with their evaluations at tau
    let (Az, Bz, Cz, W, E) = {
      Az.resize(pk.S_repr.N, E::Scalar::ZERO);
      Bz.resize(pk.S_repr.N, E::Scalar::ZERO);
      Cz.resize(pk.S_repr.N, E::Scalar::ZERO);
      let E = padded::<E>(&W.E, pk.S_repr.N, &E::Scalar::ZERO);
      let W = padded::<E>(&W.W, pk.S_repr.N, &E::Scalar::ZERO);

      (Az, Bz, Cz, W, E)
    };
    let chis_taus = EqPolynomial::evals_from_points(&tau_coords);
    let (eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau) = {
      let evals_at_tau = [&Az, &Bz, &Cz]
        .into_par_iter()
        .map(|p| MultilinearPolynomial::evaluate_with_chis(p, &chis_taus))
        .collect::<Vec<E::Scalar>>();
      (evals_at_tau[0], evals_at_tau[1], evals_at_tau[2])
    };

    // (2) send commitments to the following two oracles
    // L_row(i) = eq(tau, row(i)) for all i
    // L_col(i) = z(col(i)) for all i
    let (mem_row, mem_col, L_row, L_col) = pk.S_repr.evaluation_oracles(&S, &tau, &z);
    let (comm_L_row, comm_L_col) =
      rayon::join(|| E::CE::commit(ck, &L_row), || E::CE::commit(ck, &L_col));

    // since all the three polynomials are opened at tau,
    // we can combine them into a single polynomial opened at tau
    let eval_vec = vec![eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau];

    // absorb the claimed evaluations into the transcript
    transcript.absorb(b"e", &eval_vec.as_slice());
    // absorb commitments to L_row and L_col in the transcript
    transcript.absorb(b"e", &vec![comm_L_row, comm_L_col].as_slice());
    let comm_vec = vec![comm_Az, comm_Bz, comm_Cz];
    let poly_vec = vec![&Az, &Bz, &Cz];
    let c = transcript.squeeze(b"c")?;
    let w: PolyEvalWitness<E> = PolyEvalWitness::batch(&poly_vec, &c);
    let u: PolyEvalInstance<E> =
      PolyEvalInstance::batch(&comm_vec, tau_coords.clone(), &eval_vec, &c);

    // we now need to prove four claims
    // (1) 0 = \sum_x poly_tau(x) * (poly_Az(x) * poly_Bz(x) - poly_uCz_E(x)), and eval_Az_at_tau + r * eval_Bz_at_tau + r^2 * eval_Cz_at_tau = (Az+r*Bz+r^2*Cz)(tau)
    // (2) eval_Az_at_tau + c * eval_Bz_at_tau + c^2 * eval_Cz_at_tau = \sum_y L_row(y) * (val_A(y) + c * val_B(y) + c^2 * val_C(y)) * L_col(y)
    // (3) L_row(i) = eq(tau, row(i)) and L_col(i) = z(col(i))
    // (4) Check that the witness polynomial W is well-formed e.g., it is padded with only zeros
    let gamma = transcript.squeeze(b"g")?;
    let r = transcript.squeeze(b"r")?;

    let transcript_parallel = Mutex::new(transcript);

    let ((mut outer_sc_inst, mut inner_sc_inst), (mem_res, lookup_res)) = rayon::join(
      || {
        // a sum-check instance to prove the first claim
        let outer_sc_inst = OuterSumcheckInstance::new(
          PowPolynomial::new(&tau, num_rounds_sc).evals(),
          Az.clone(),
          Bz.clone(),
          (0..Cz.len())
            .map(|i| U.u * Cz[i] + E[i])
            .collect::<Vec<E::Scalar>>(),
          w.p.clone(), // Mz = Az + r * Bz + r^2 * Cz
          &u.e,        // eval_Az_at_tau + r * eval_Az_at_tau + r^2 * eval_Cz_at_tau
        );

        // a sum-check instance to prove the second claim
        let val = zip_with!(
          par_iter,
          (pk.S_repr.val_A, pk.S_repr.val_B, pk.S_repr.val_C),
          |v_a, v_b, v_c| *v_a + c * *v_b + c * c * *v_c
        )
        .collect::<Vec<E::Scalar>>();
        let inner_sc_inst = InnerSumcheckInstance {
          claim: eval_Az_at_tau + c * eval_Bz_at_tau + c * c * eval_Cz_at_tau,
          poly_L_row: MultilinearPolynomial::new(L_row.clone()),
          poly_L_col: MultilinearPolynomial::new(L_col.clone()),
          poly_val: MultilinearPolynomial::new(val),
        };

        (outer_sc_inst, inner_sc_inst)
      },
      || rayon::join(
        || {
          // a third sum-check instance to prove the read-only memory claim
          // we now need to prove that L_row and L_col are well-formed

          // hash the tuples of (addr,val) memory contents and read responses into a single field element using `hash_func`

          let (comm_mem_oracles, mem_oracles, mem_aux) =
            MemorySumcheckInstance::<E>::compute_oracles(
              ck,
              &r,
              &gamma,
              &mem_row,
              &pk.S_repr.row,
              &L_row,
              &pk.S_repr.ts_row,
              &mem_col,
              &pk.S_repr.col,
              &L_col,
              &pk.S_repr.ts_col,
            )?;

          // absorb the commitments
          let mut transcript_guard = transcript_parallel.lock().unwrap();
          transcript_guard.absorb(b"l", &comm_mem_oracles.as_slice());

          let rho = transcript_guard.squeeze(b"r")?;
          let poly_eq = MultilinearPolynomial::new(PowPolynomial::new(&rho, num_rounds_sc).evals());

          Ok::<_, NovaError>((
            MemorySumcheckInstance::<E>::new(
              mem_oracles.clone(),
              mem_aux,
              poly_eq.Z,
              pk.S_repr.ts_row.clone(),
              pk.S_repr.ts_col.clone(),
            ),
            comm_mem_oracles,
            mem_oracles,
          ))
        },
        || {
          // 4th sumcheck instance to prove lookup

          let table_size = pk.S_repr.const_ts_lookup.len();
          let const_one = vec![E::Scalar::from(1u64); table_size];
          let table_size: u64 = table_size.try_into().unwrap();

          let (comm_lookup_oracles, lookup_oracles, lookup_aux) =
          LookupSumcheckInstance::<E>::compute_oracles(
            ck,
            &lookup_r,
            &lookup_gamma,
            vec![
              Box::new(NaturalNumVec::<E>::new(table_size)), // address
              Box::new(init_values.clone().into_iter()),
              // init_ts is zero vector
            ], // t set
            vec![
              Box::new(NaturalNumVec::<E>::new(table_size)), // address
              Box::new(final_values.clone().into_iter()),
              Box::new(final_ts.clone().into_iter()),
            ], // w set
            &const_one, // ts
          )?;

          // absorb the commitments
          let mut transcript_guard = transcript_parallel.lock().unwrap();
          transcript_guard.absorb(b"l", &comm_lookup_oracles.as_slice());

          let rho = transcript_guard.squeeze(b"r")?;
          let poly_eq = MultilinearPolynomial::new(PowPolynomial::new(&rho, num_rounds_sc).evals());

          Ok::<_, NovaError>((
            LookupSumcheckInstance::new(
              lookup_oracles[0..2].to_vec().try_into().unwrap(),
              lookup_aux[0..2].to_vec().try_into().unwrap(),
              poly_eq.Z,
              pk.S_repr.const_ts_lookup.clone(),
              Some(RW_acc),
            ),
            comm_lookup_oracles,
            lookup_oracles,
          ))
        },
      ),
    );

    let (mut mem_sc_inst, comm_mem_oracles, mem_oracles) = mem_res?;
    let (mut lookup_sc_inst, comm_lookup_oracles, lookup_oracles) = lookup_res?;

    let mut witness_sc_inst = WitnessBoundSumcheck::new(tau, W.clone(), S.num_vars);

    let (sc, rand_sc, claims_mem, claims_outer, claims_inner, claims_witness, claims_lookup) = {
      let mut transcript_guard = transcript_parallel.lock().unwrap();

      Self::prove_helper(
        &mut mem_sc_inst,
        &mut outer_sc_inst,
        &mut inner_sc_inst,
        &mut witness_sc_inst,
        &mut lookup_sc_inst,
        &mut *transcript_guard,
      )?
    };

    // claims from the end of the sum-check
    let eval_Az = claims_outer[0][0];
    let eval_Bz = claims_outer[0][1];

    let eval_L_row = claims_inner[0][0];
    let eval_L_col = claims_inner[0][1];

    let eval_t_plus_r_inv_row = claims_mem[0][0];
    let eval_w_plus_r_inv_row = claims_mem[0][1];
    let eval_ts_row = claims_mem[0][2];

    let eval_t_plus_r_inv_col = claims_mem[1][0];
    let eval_w_plus_r_inv_col = claims_mem[1][1];
    let eval_ts_col = claims_mem[1][2];
    let eval_W = claims_witness[0][0];

    // lookup
    let eval_t_plus_r_inv_lookup = claims_lookup[0][0];
    let eval_w_plus_r_inv_lookup = claims_lookup[0][1];
    let eval_ts_lookup = claims_lookup[0][2];

    // compute the remaining claims that did not come for free from the sum-check prover
    let (
      eval_Cz,
      eval_E,
      eval_val_A,
      eval_val_B,
      eval_val_C,
      eval_row,
      eval_col,
      // lookup
      eval_init_value_lookup,
      eval_final_value_lookup,
      eval_final_ts_lookup,
    ) = {
      let e = [
        &Cz,
        &E,
        &pk.S_repr.val_A,
        &pk.S_repr.val_B,
        &pk.S_repr.val_C,
        &pk.S_repr.row,
        &pk.S_repr.col,
        // lookup
        &init_values,
        &final_values,
        &final_ts,
      ]
      .into_par_iter()
      .map(|p| MultilinearPolynomial::evaluate_with(p, &rand_sc))
      .collect::<Vec<E::Scalar>>();
      (e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9])
    };

    // all the evaluations are at rand_sc, we can fold them into one claim
    let eval_vec = vec![
      eval_W,
      eval_Az,
      eval_Bz,
      eval_Cz,
      eval_E,
      eval_L_row,
      eval_L_col,
      eval_val_A,
      eval_val_B,
      eval_val_C,
      eval_t_plus_r_inv_row,
      eval_row,
      eval_w_plus_r_inv_row,
      eval_ts_row,
      eval_t_plus_r_inv_col,
      eval_col,
      eval_w_plus_r_inv_col,
      eval_ts_col,
      // lookup
      eval_t_plus_r_inv_lookup,
      eval_w_plus_r_inv_lookup,
      eval_ts_lookup,
      eval_init_value_lookup,
      eval_final_value_lookup,
      eval_final_ts_lookup,
    ];

    let comm_vec = [
      U.comm_W,
      comm_Az,
      comm_Bz,
      comm_Cz,
      U.comm_E,
      comm_L_row,
      comm_L_col,
      pk.S_comm.comm_val_A,
      pk.S_comm.comm_val_B,
      pk.S_comm.comm_val_C,
      comm_mem_oracles[0],
      pk.S_comm.comm_row,
      comm_mem_oracles[1],
      pk.S_comm.comm_ts_row,
      comm_mem_oracles[2],
      pk.S_comm.comm_col,
      comm_mem_oracles[3],
      pk.S_comm.comm_ts_col,
      // lookup
      comm_lookup_oracles[0],
      comm_lookup_oracles[1],
      pk.S_comm.comm_const_ts_lookup, // ts
      pk.S_comm.comm_init_values_lookup,
      comm_final_value,
      comm_final_ts,
    ];
    let poly_vec = [
      &W,
      &Az,
      &Bz,
      &Cz,
      &E,
      &L_row,
      &L_col,
      &pk.S_repr.val_A,
      &pk.S_repr.val_B,
      &pk.S_repr.val_C,
      mem_oracles[0].as_ref(),
      &pk.S_repr.row,
      mem_oracles[1].as_ref(),
      &pk.S_repr.ts_row,
      mem_oracles[2].as_ref(),
      &pk.S_repr.col,
      mem_oracles[3].as_ref(),
      &pk.S_repr.ts_col,
      // lookup
      &lookup_oracles[0],
      &lookup_oracles[1],
      &pk.S_repr.const_ts_lookup,
      &pk.S_repr.init_values_lookup,
      &final_values,
      &final_ts,
    ];

    let mut transcript_guard = transcript_parallel.lock().unwrap();
    transcript_guard.absorb(b"e", &eval_vec.as_slice()); // comm_vec is already in the transcript
    let c = transcript_guard.squeeze(b"c")?;
    let w: PolyEvalWitness<E> = PolyEvalWitness::batch(&poly_vec, &c);
    let u: PolyEvalInstance<E> = PolyEvalInstance::batch(&comm_vec, rand_sc.clone(), &eval_vec, &c);

    let eval_arg = EE::prove(ck, &pk.pk_ee, &mut *transcript_guard, &u.c, &w.p, &rand_sc, &u.e)?;

    Ok(Self {
      comm_Az: comm_Az.compress(),
      comm_Bz: comm_Bz.compress(),
      comm_Cz: comm_Cz.compress(),
      comm_L_row: comm_L_row.compress(),
      comm_L_col: comm_L_col.compress(),

      comm_t_plus_r_inv_row: comm_mem_oracles[0].compress(),
      comm_w_plus_r_inv_row: comm_mem_oracles[1].compress(),
      comm_t_plus_r_inv_col: comm_mem_oracles[2].compress(),
      comm_w_plus_r_inv_col: comm_mem_oracles[3].compress(),

      eval_Az_at_tau,
      eval_Bz_at_tau,
      eval_Cz_at_tau,

      sc,

      eval_Az,
      eval_Bz,
      eval_Cz,
      eval_E,
      eval_L_row,
      eval_L_col,
      eval_val_A,
      eval_val_B,
      eval_val_C,

      eval_W,

      eval_t_plus_r_inv_row,
      eval_row,
      eval_w_plus_r_inv_row,
      eval_ts_row,

      eval_col,
      eval_t_plus_r_inv_col,
      eval_w_plus_r_inv_col,
      eval_ts_col,

      // lookup
      comm_t_plus_r_inv_lookup: comm_lookup_oracles[0].compress(),
      comm_w_plus_r_inv_lookup: comm_lookup_oracles[1].compress(),
      comm_final_value: comm_final_value.compress(),
      comm_final_ts: comm_final_ts.compress(),
      eval_t_plus_r_inv_lookup,
      eval_w_plus_r_inv_lookup,
      eval_ts_lookup,
      eval_init_value_lookup,
      eval_final_value_lookup,
      eval_final_ts_lookup,

      eval_arg,
    })
  }

  
  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify<E2: Engine>(
    &self,
    vk: &Self::VerifierKey,
    U: &RelaxedR1CSInstance<E>,
    lookup_intermediate_gamma: E::Scalar,
    RW_acc: E::Scalar,
    challenges: (E::Scalar, E::Scalar),
  ) -> Result<(), NovaError>
  where
    E: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E as Engine>::Scalar>,
  {
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");
    let (lookup_r, lookup_gamma) = challenges;

    // append the verifier key (including commitment to R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", U);

    // add lookup table to transcript
    transcript.absorb(b"RW_acc", &RW_acc);
    transcript.absorb(b"r", &lookup_r);
    transcript.absorb(b"gamma", &lookup_gamma);

    let comm_Az = Commitment::<E>::decompress(&self.comm_Az)?;
    let comm_Bz = Commitment::<E>::decompress(&self.comm_Bz)?;
    let comm_Cz = Commitment::<E>::decompress(&self.comm_Cz)?;
    let comm_L_row = Commitment::<E>::decompress(&self.comm_L_row)?;
    let comm_L_col = Commitment::<E>::decompress(&self.comm_L_col)?;
    let comm_t_plus_r_inv_row = Commitment::<E>::decompress(&self.comm_t_plus_r_inv_row)?;
    let comm_w_plus_r_inv_row = Commitment::<E>::decompress(&self.comm_w_plus_r_inv_row)?;
    let comm_t_plus_r_inv_col = Commitment::<E>::decompress(&self.comm_t_plus_r_inv_col)?;
    let comm_w_plus_r_inv_col = Commitment::<E>::decompress(&self.comm_w_plus_r_inv_col)?;

    let comm_t_plus_r_inv_lookup = Commitment::<E>::decompress(&self.comm_t_plus_r_inv_lookup)?;
    let comm_w_plus_r_inv_lookup = Commitment::<E>::decompress(&self.comm_w_plus_r_inv_lookup)?;
    let comm_final_value = Commitment::<E>::decompress(&self.comm_final_value)?;
    let comm_final_ts = Commitment::<E>::decompress(&self.comm_final_ts)?;

    // verify lookup challenge
    Self::verify_challenge::<E2>(
      comm_final_value,
      comm_final_ts,
      lookup_intermediate_gamma,
      challenges,
    )?;

    transcript.absorb(b"c", &[comm_Az, comm_Bz, comm_Cz].as_slice());
    transcript.absorb(b"c", &[comm_final_value, comm_final_ts].as_slice());

    let num_rounds_sc = vk.S_comm.N.log_2();
    let tau = transcript.squeeze(b"t")?;
    let tau_coords = PowPolynomial::new(&tau, num_rounds_sc).coordinates();

    // add claims about Az, Bz, and Cz to be checked later
    // since all the three polynomials are opened at tau,
    // we can combine them into a single polynomial opened at tau
    let eval_vec = vec![
      self.eval_Az_at_tau,
      self.eval_Bz_at_tau,
      self.eval_Cz_at_tau,
    ];

    transcript.absorb(b"e", &eval_vec.as_slice());

    transcript.absorb(b"e", &vec![comm_L_row, comm_L_col].as_slice());
    let comm_vec = vec![comm_Az, comm_Bz, comm_Cz];
    let c = transcript.squeeze(b"c")?;
    let u: PolyEvalInstance<E> =
      PolyEvalInstance::batch(&comm_vec, tau_coords.clone(), &eval_vec, &c);
    let claim = u.e;

    let gamma = transcript.squeeze(b"g")?;

    let r = transcript.squeeze(b"r")?;

    transcript.absorb(
      b"l",
      &vec![
        comm_t_plus_r_inv_row,
        comm_w_plus_r_inv_row,
        comm_t_plus_r_inv_col,
        comm_w_plus_r_inv_col,
        comm_t_plus_r_inv_lookup,
        comm_w_plus_r_inv_lookup,
      ]
      .as_slice(),
    );

    let rho = transcript.squeeze(b"r")?;

    let num_claims = 13;
    let s = transcript.squeeze(b"r")?;
    let coeffs = powers(&s, num_claims);
    let claim = (coeffs[7] + coeffs[8]) * claim + coeffs[10] * RW_acc; // rest are zeros

    // verify sc
    let (claim_sc_final, rand_sc) = self.sc.verify(claim, num_rounds_sc, 3, &mut transcript)?;

    // verify claim_sc_final
    let claim_sc_final_expected = {
      let rand_eq_bound_rand_sc = PowPolynomial::new(&rho, num_rounds_sc).evaluate(&rand_sc);
      let eq_tau: EqPolynomial<_> = PowPolynomial::new(&tau, num_rounds_sc).into();

      let taus_bound_rand_sc = eq_tau.evaluate(&rand_sc);
      let taus_masked_bound_rand_sc =
        MaskedEqPolynomial::new(&eq_tau, vk.num_vars.log_2()).evaluate(&rand_sc);

      let eval_t_plus_r_row = {
        let eval_addr_row = IdentityPolynomial::new(num_rounds_sc).evaluate(&rand_sc);
        let eval_val_row = taus_bound_rand_sc;
        let eval_t = eval_addr_row + gamma * eval_val_row;
        eval_t + r
      };

      let eval_w_plus_r_row = {
        let eval_addr_row = self.eval_row;
        let eval_val_row = self.eval_L_row;
        let eval_w = eval_addr_row + gamma * eval_val_row;
        eval_w + r
      };

      let eval_t_plus_r_col = {
        let eval_addr_col = IdentityPolynomial::new(num_rounds_sc).evaluate(&rand_sc);

        // memory contents is z, so we compute eval_Z from eval_W and eval_X
        let eval_val_col = {
          // rand_sc was padded, so we now remove the padding
          let (factor, rand_sc_unpad) = {
            let l = vk.S_comm.N.log_2() - (2 * vk.num_vars).log_2();

            let mut factor = E::Scalar::ONE;
            for r_p in rand_sc.iter().take(l) {
              factor *= E::Scalar::ONE - r_p
            }

            let rand_sc_unpad = rand_sc[l..].to_vec();

            (factor, rand_sc_unpad)
          };

          let eval_X = {
            // public IO is (u, X)
            let X = vec![U.u]
              .into_iter()
              .chain(U.X.iter().cloned())
              .collect::<Vec<E::Scalar>>();

            // evaluate the sparse polynomial at rand_sc_unpad[1..]
            let poly_X = SparsePolynomial::new(rand_sc_unpad.len() - 1, X);
            poly_X.evaluate(&rand_sc_unpad[1..])
          };

          self.eval_W + factor * rand_sc_unpad[0] * eval_X
        };
        let eval_t = eval_addr_col + gamma * eval_val_col;
        eval_t + r
      };

      let eval_w_plus_r_col = {
        let eval_addr_col = self.eval_col;
        let eval_val_col = self.eval_L_col;
        let eval_w = eval_addr_col + gamma * eval_val_col;
        eval_w + r
      };

      let eval_addr = IdentityPolynomial::new(num_rounds_sc).evaluate(&rand_sc);
      let eval_t_plus_r_lookup = {
        let eval_val = self.eval_init_value_lookup;
        let eval_t = eval_addr + lookup_gamma * eval_val;
        eval_t + lookup_r
      };

      let eval_w_plus_r_lookup = {
        let eval_val = self.eval_final_value_lookup;
        let eval_ts = self.eval_final_ts_lookup;
        let eval_w = eval_addr + lookup_gamma * eval_val + lookup_gamma * lookup_gamma * eval_ts;
        eval_w + lookup_r
      };

      let claim_mem_final_expected: E::Scalar = coeffs[0]
        * (self.eval_t_plus_r_inv_row - self.eval_w_plus_r_inv_row)
        + coeffs[1] * (self.eval_t_plus_r_inv_col - self.eval_w_plus_r_inv_col)
        + coeffs[2]
          * (rand_eq_bound_rand_sc
            * (self.eval_t_plus_r_inv_row * eval_t_plus_r_row - self.eval_ts_row))
        + coeffs[3]
          * (rand_eq_bound_rand_sc
            * (self.eval_w_plus_r_inv_row * eval_w_plus_r_row - E::Scalar::ONE))
        + coeffs[4]
          * (rand_eq_bound_rand_sc
            * (self.eval_t_plus_r_inv_col * eval_t_plus_r_col - self.eval_ts_col))
        + coeffs[5]
          * (rand_eq_bound_rand_sc
            * (self.eval_w_plus_r_inv_col * eval_w_plus_r_col - E::Scalar::ONE));

      let claim_outer_final_expected = coeffs[6]
        * taus_bound_rand_sc
        * (self.eval_Az * self.eval_Bz - U.u * self.eval_Cz - self.eval_E)
        + coeffs[7] * taus_bound_rand_sc * (self.eval_Az + c * self.eval_Bz + c * c * self.eval_Cz);
      let claim_inner_final_expected = coeffs[8]
        * self.eval_L_row
        * self.eval_L_col
        * (self.eval_val_A + c * self.eval_val_B + c * c * self.eval_val_C);

      let claim_witness_final_expected = coeffs[9] * taus_masked_bound_rand_sc * self.eval_W;

      let claim_lookup_expected = coeffs[10]
      * (self.eval_t_plus_r_inv_lookup - self.eval_w_plus_r_inv_lookup)
      + coeffs[11]
        * (rand_eq_bound_rand_sc
          * (self.eval_t_plus_r_inv_lookup * eval_t_plus_r_lookup - self.eval_ts_lookup))
      + coeffs[12]
        * (rand_eq_bound_rand_sc
          * (self.eval_w_plus_r_inv_lookup * eval_w_plus_r_lookup - E::Scalar::ONE));

      claim_mem_final_expected
        + claim_outer_final_expected
        + claim_inner_final_expected
        + claim_witness_final_expected
        + claim_lookup_expected
    };

    if claim_sc_final_expected != claim_sc_final {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let eval_vec = vec![
      self.eval_W,
      self.eval_Az,
      self.eval_Bz,
      self.eval_Cz,
      self.eval_E,
      self.eval_L_row,
      self.eval_L_col,
      self.eval_val_A,
      self.eval_val_B,
      self.eval_val_C,
      self.eval_t_plus_r_inv_row,
      self.eval_row,
      self.eval_w_plus_r_inv_row,
      self.eval_ts_row,
      self.eval_t_plus_r_inv_col,
      self.eval_col,
      self.eval_w_plus_r_inv_col,
      self.eval_ts_col,

      // lookup
      self.eval_t_plus_r_inv_lookup,
      self.eval_w_plus_r_inv_lookup,
      self.eval_ts_lookup,
      self.eval_init_value_lookup,
      self.eval_final_value_lookup,
      self.eval_final_ts_lookup,
    ];

    let comm_vec = [
      U.comm_W,
      comm_Az,
      comm_Bz,
      comm_Cz,
      U.comm_E,
      comm_L_row,
      comm_L_col,
      vk.S_comm.comm_val_A,
      vk.S_comm.comm_val_B,
      vk.S_comm.comm_val_C,
      comm_t_plus_r_inv_row,
      vk.S_comm.comm_row,
      comm_w_plus_r_inv_row,
      vk.S_comm.comm_ts_row,
      comm_t_plus_r_inv_col,
      vk.S_comm.comm_col,
      comm_w_plus_r_inv_col,
      vk.S_comm.comm_ts_col,
      // lookup
      comm_t_plus_r_inv_lookup,
      comm_w_plus_r_inv_lookup,
      vk.S_comm.comm_const_ts_lookup, // ts
      vk.S_comm.comm_init_values_lookup,
      comm_final_value,
      comm_final_ts,
    ];
    transcript.absorb(b"e", &eval_vec.as_slice()); // comm_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let u: PolyEvalInstance<E> = PolyEvalInstance::batch(&comm_vec, rand_sc.clone(), &eval_vec, &c);

    // verify
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &u.c,
      &rand_sc,
      &u.e,
      &self.eval_arg,
    )?;

    Ok(())
  }
  
}

#[cfg(test)]
mod tests {
  use crate::provider::PallasEngine;

  use super::*;
  use ff::Field;
  use pasta_curves::Fq as Scalar;

  #[test]
  fn test_padded_lookup_ppsnark() {
    let mut rng = rand::thread_rng();
    let e = Scalar::random(&mut rng);
    let v: Vec<Scalar> = (0..10).map(|_| Scalar::random(&mut rng)).collect();
    let n = 20;

    let result = padded::<PallasEngine>(&v, n, &e);

    assert_eq!(result.len(), n);
    assert_eq!(&result[0..10], &v[..]);
    assert!(result[10..].iter().all(|&i| i == e));
  }
}