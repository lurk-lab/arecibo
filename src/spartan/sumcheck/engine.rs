use ff::Field;
use rayon::prelude::*;

use crate::provider::util::field::batch_invert;
use crate::spartan::math::Math;
use crate::spartan::polys::{
  eq::EqPolynomial, masked_eq::MaskedEqPolynomial, multilinear::MultilinearPolynomial,
  power::PowPolynomial,
};
use crate::spartan::sumcheck::SumcheckProof;
use crate::traits::commitment::CommitmentEngineTrait;
use crate::{Commitment, CommitmentKey, Engine, NovaError};
use crate::spartan::powers;
use itertools::Itertools;
use std::marker::PhantomData;

// Define a iterator over Natural number without allocated a whole size vector
// it started from 0 instead of 1
pub(in crate::spartan) struct NaturalNumVec<E: Engine> {
  curr: u64,
  size: u64,
  _phatom: PhantomData<E>,
}
impl<E: Engine> NaturalNumVec<E> {
  pub fn new(size: u64) -> Self {
    NaturalNumVec {
      curr: 0,
      size,
      _phatom: PhantomData,
    }
  }
}

impl<E: Engine> Iterator for NaturalNumVec<E> {
  // We can refer to this type using Self::Item
  type Item = E::Scalar;

  fn next(&mut self) -> Option<Self::Item> {
    let next = if self.curr < self.size {
      Some(E::Scalar::from(self.curr))
    } else {
      None
    };
    self.curr += 1;
    next
  }
}

impl<E: Engine> ExactSizeIterator for NaturalNumVec<E> {
  // We can easily calculate the remaining number of iterations.
  fn len(&self) -> usize {
    (self.size - self.curr).try_into().unwrap()
  }
}

/// Defines a trait for implementing sum-check in a generic manner
pub trait SumcheckEngine<E: Engine>: Send + Sync {
  /// returns the initial claims
  fn initial_claims(&self) -> Vec<E::Scalar>;

  /// degree of the sum-check polynomial
  fn degree(&self) -> usize;

  /// the size of the polynomials
  fn size(&self) -> usize;

  /// returns evaluation points at 0, 2, d-1 (where d is the degree of the sum-check polynomial)
  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>>;

  /// bounds a variable in the constituent polynomials
  fn bound(&mut self, r: &E::Scalar);

  /// returns the final claims
  fn final_claims(&self) -> Vec<Vec<E::Scalar>>;
}

/// The [`WitnessBoundSumcheck`] ensures that the witness polynomial W defined over n = log(N) variables,
/// is zero outside of the first `num_vars = 2^m` entries.
///
/// # Details
///
/// The `W` polynomial is padded with zeros to size N = 2^n.
/// The `masked_eq` polynomials is defined as with regards to a random challenge `tau` as
/// the eq(tau) polynomial, where the first 2^m evaluations to 0.
///
/// The instance is given by
///  `0 = ∑_{0≤i<2^n} masked_eq[i] * W[i]`.
/// It is equivalent to the expression
///  `0 = ∑_{2^m≤i<2^n} eq[i] * W[i]`
/// Since `eq` is random, the instance is only satisfied if `W[2^{m}..] = 0`.
pub(in crate::spartan) struct WitnessBoundSumcheck<E: Engine> {
  poly_W: MultilinearPolynomial<E::Scalar>,
  poly_masked_eq: MultilinearPolynomial<E::Scalar>,
}

impl<E: Engine> WitnessBoundSumcheck<E> {
  pub fn new(tau: E::Scalar, poly_W_padded: Vec<E::Scalar>, num_vars: usize) -> Self {
    let num_vars_log = num_vars.log_2();
    // When num_vars = num_rounds, we shouldn't have to prove anything
    // but we still want this instance to compute the evaluation of W
    let num_rounds = poly_W_padded.len().log_2();
    assert!(num_vars_log < num_rounds);

    let tau_coords = PowPolynomial::new(&tau, num_rounds).coordinates();
    let poly_masked_eq_evals =
      MaskedEqPolynomial::new(&EqPolynomial::new(tau_coords), num_vars_log).evals();

    Self {
      poly_W: MultilinearPolynomial::new(poly_W_padded),
      poly_masked_eq: MultilinearPolynomial::new(poly_masked_eq_evals),
    }
  }
}
impl<E: Engine> SumcheckEngine<E> for WitnessBoundSumcheck<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
    vec![E::Scalar::ZERO]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    assert_eq!(self.poly_W.len(), self.poly_masked_eq.len());
    self.poly_W.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>> {
    let comb_func = |poly_A_comp: &E::Scalar,
                     poly_B_comp: &E::Scalar,
                     _: &E::Scalar|
     -> E::Scalar { *poly_A_comp * *poly_B_comp };

    let (eval_point_0, eval_point_2, eval_point_3) = SumcheckProof::<E>::compute_eval_points_cubic(
      &self.poly_masked_eq,
      &self.poly_W,
      &self.poly_W, // unused
      &comb_func,
    );

    vec![vec![eval_point_0, eval_point_2, eval_point_3]]
  }

  fn bound(&mut self, r: &E::Scalar) {
    [&mut self.poly_W, &mut self.poly_masked_eq]
      .par_iter_mut()
      .for_each(|poly| poly.bind_poly_var_top(r));
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
    vec![vec![self.poly_W[0], self.poly_masked_eq[0]]]
  }
}

pub(in crate::spartan) struct MemorySumcheckInstance<E: Engine> {
  // row
  w_plus_r_row: MultilinearPolynomial<E::Scalar>,
  t_plus_r_row: MultilinearPolynomial<E::Scalar>,
  t_plus_r_inv_row: MultilinearPolynomial<E::Scalar>,
  w_plus_r_inv_row: MultilinearPolynomial<E::Scalar>,
  ts_row: MultilinearPolynomial<E::Scalar>,

  // col
  w_plus_r_col: MultilinearPolynomial<E::Scalar>,
  t_plus_r_col: MultilinearPolynomial<E::Scalar>,
  t_plus_r_inv_col: MultilinearPolynomial<E::Scalar>,
  w_plus_r_inv_col: MultilinearPolynomial<E::Scalar>,
  ts_col: MultilinearPolynomial<E::Scalar>,

  // eq
  poly_eq: MultilinearPolynomial<E::Scalar>,

  // zero polynomial
  poly_zero: MultilinearPolynomial<E::Scalar>,
}

impl<E: Engine> MemorySumcheckInstance<E> {
  /// Computes witnesses for `MemoryInstanceSumcheck`
  ///
  /// # Description
  /// We use the logUp protocol to prove that
  /// ∑ TS[i]/(T[i] + r) - 1/(W[i] + r) = 0
  /// where
  ///   T_row[i] = mem_row[i]      * gamma + i
  ///            = eq(tau)[i]      * gamma + i
  ///   W_row[i] = L_row[i]        * gamma + addr_row[i]
  ///            = eq(tau)[row[i]] * gamma + addr_row[i]
  ///   T_col[i] = mem_col[i]      * gamma + i
  ///            = z[i]            * gamma + i
  ///   W_col[i] = addr_col[i]     * gamma + addr_col[i]
  ///            = z[col[i]]       * gamma + addr_col[i]
  /// and
  ///   `TS_row`, `TS_col` are integer-valued vectors representing the number of reads
  ///   to each memory cell of `L_row`, `L_col`
  ///
  /// The function returns oracles for the polynomials TS[i]/(T[i] + r), 1/(W[i] + r),
  /// as well as auxiliary polynomials T[i] + r, W[i] + r
  pub fn compute_oracles(
    ck: &CommitmentKey<E>,
    r: &E::Scalar,
    gamma: &E::Scalar,
    mem_row: &[E::Scalar],
    addr_row: &[E::Scalar],
    L_row: &[E::Scalar],
    ts_row: &[E::Scalar],
    mem_col: &[E::Scalar],
    addr_col: &[E::Scalar],
    L_col: &[E::Scalar],
    ts_col: &[E::Scalar],
  ) -> Result<([Commitment<E>; 4], [Vec<E::Scalar>; 4], [Vec<E::Scalar>; 4]), NovaError> {
    // hash the tuples of (addr,val) memory contents and read responses into a single field element using `hash_func`
    let hash_func_vec = |mem: &[E::Scalar],
                         addr: &[E::Scalar],
                         lookups: &[E::Scalar]|
     -> (Vec<E::Scalar>, Vec<E::Scalar>) {
      let hash_func = |addr: &E::Scalar, val: &E::Scalar| -> E::Scalar { *val * gamma + *addr };
      assert_eq!(addr.len(), lookups.len());
      rayon::join(
        || {
          (0..mem.len())
            .map(|i| hash_func(&E::Scalar::from(i as u64), &mem[i]))
            .collect::<Vec<E::Scalar>>()
        },
        || {
          (0..addr.len())
            .map(|i| hash_func(&addr[i], &lookups[i]))
            .collect::<Vec<E::Scalar>>()
        },
      )
    };

    let ((T_row, W_row), (T_col, W_col)) = rayon::join(
      || hash_func_vec(mem_row, addr_row, L_row),
      || hash_func_vec(mem_col, addr_col, L_col),
    );

    // compute vectors TS[i]/(T[i] + r) and 1/(W[i] + r)
    let helper = |T: &[E::Scalar],
                  W: &[E::Scalar],
                  TS: &[E::Scalar],
                  r: &E::Scalar|
     -> (
      (
        Result<Vec<E::Scalar>, NovaError>,
        Result<Vec<E::Scalar>, NovaError>,
      ),
      (Vec<E::Scalar>, Vec<E::Scalar>),
    ) {
      rayon::join(
        || {
          rayon::join(
            || {
              let inv = batch_invert(T.par_iter().map(|e| *e + *r).collect::<Vec<_>>())?;

              // compute inv[i] * TS[i] in parallel
              Ok(
                zip_with!((inv.into_par_iter(), TS.par_iter()), |e1, e2| e1 * *e2)
                  .collect::<Vec<_>>(),
              )
            },
            || batch_invert(W.par_iter().map(|e| *e + *r).collect::<Vec<_>>()),
          )
        },
        || {
          rayon::join(
            || T.par_iter().map(|e| *e + *r).collect(),
            || W.par_iter().map(|e| *e + *r).collect(),
          )
        },
      )
    };

    let (
      ((t_plus_r_inv_row, w_plus_r_inv_row), (t_plus_r_row, w_plus_r_row)),
      ((t_plus_r_inv_col, w_plus_r_inv_col), (t_plus_r_col, w_plus_r_col)),
    ) = rayon::join(
      || helper(&T_row, &W_row, ts_row, r),
      || helper(&T_col, &W_col, ts_col, r),
    );

    let t_plus_r_inv_row = t_plus_r_inv_row?;
    let w_plus_r_inv_row = w_plus_r_inv_row?;
    let t_plus_r_inv_col = t_plus_r_inv_col?;
    let w_plus_r_inv_col = w_plus_r_inv_col?;

    let (
      (comm_t_plus_r_inv_row, comm_w_plus_r_inv_row),
      (comm_t_plus_r_inv_col, comm_w_plus_r_inv_col),
    ) = rayon::join(
      || {
        rayon::join(
          || E::CE::commit(ck, &t_plus_r_inv_row),
          || E::CE::commit(ck, &w_plus_r_inv_row),
        )
      },
      || {
        rayon::join(
          || E::CE::commit(ck, &t_plus_r_inv_col),
          || E::CE::commit(ck, &w_plus_r_inv_col),
        )
      },
    );

    let comm_vec = [
      comm_t_plus_r_inv_row,
      comm_w_plus_r_inv_row,
      comm_t_plus_r_inv_col,
      comm_w_plus_r_inv_col,
    ];

    let poly_vec = [
      t_plus_r_inv_row,
      w_plus_r_inv_row,
      t_plus_r_inv_col,
      w_plus_r_inv_col,
    ];

    let aux_poly_vec = [t_plus_r_row, w_plus_r_row, t_plus_r_col, w_plus_r_col];

    Ok((comm_vec, poly_vec, aux_poly_vec))
  }

  pub fn new(
    polys_oracle: [Vec<E::Scalar>; 4],
    polys_aux: [Vec<E::Scalar>; 4],
    poly_eq: Vec<E::Scalar>,
    ts_row: Vec<E::Scalar>,
    ts_col: Vec<E::Scalar>,
  ) -> Self {
    let [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col] = polys_oracle;
    let [t_plus_r_row, w_plus_r_row, t_plus_r_col, w_plus_r_col] = polys_aux;

    let zero = vec![E::Scalar::ZERO; poly_eq.len()];

    Self {
      w_plus_r_row: MultilinearPolynomial::new(w_plus_r_row),
      t_plus_r_row: MultilinearPolynomial::new(t_plus_r_row),
      t_plus_r_inv_row: MultilinearPolynomial::new(t_plus_r_inv_row),
      w_plus_r_inv_row: MultilinearPolynomial::new(w_plus_r_inv_row),
      ts_row: MultilinearPolynomial::new(ts_row),
      w_plus_r_col: MultilinearPolynomial::new(w_plus_r_col),
      t_plus_r_col: MultilinearPolynomial::new(t_plus_r_col),
      t_plus_r_inv_col: MultilinearPolynomial::new(t_plus_r_inv_col),
      w_plus_r_inv_col: MultilinearPolynomial::new(w_plus_r_inv_col),
      ts_col: MultilinearPolynomial::new(ts_col),
      poly_eq: MultilinearPolynomial::new(poly_eq),
      poly_zero: MultilinearPolynomial::new(zero),
    }
  }
}

impl<E: Engine> SumcheckEngine<E> for MemorySumcheckInstance<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
    vec![E::Scalar::ZERO; 6]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    // sanity checks
    assert_eq!(self.w_plus_r_row.len(), self.t_plus_r_row.len());
    assert_eq!(self.w_plus_r_row.len(), self.ts_row.len());
    assert_eq!(self.w_plus_r_row.len(), self.w_plus_r_col.len());
    assert_eq!(self.w_plus_r_row.len(), self.t_plus_r_col.len());
    assert_eq!(self.w_plus_r_row.len(), self.ts_col.len());

    self.w_plus_r_row.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>> {
    let comb_func = |poly_A_comp: &E::Scalar,
                     poly_B_comp: &E::Scalar,
                     _poly_C_comp: &E::Scalar|
     -> E::Scalar { *poly_A_comp - *poly_B_comp };

    let comb_func2 =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       _poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - E::Scalar::ONE) };

    let comb_func3 =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    // inv related evaluation points
    // 0 = ∑ TS[i]/(T[i] + r) - 1/(W[i] + r)
    let (eval_inv_0_row, eval_inv_2_row, eval_inv_3_row) =
      SumcheckProof::<E>::compute_eval_points_cubic(
        &self.t_plus_r_inv_row,
        &self.w_plus_r_inv_row,
        &self.poly_zero,
        &comb_func,
      );

    let (eval_inv_0_col, eval_inv_2_col, eval_inv_3_col) =
      SumcheckProof::<E>::compute_eval_points_cubic(
        &self.t_plus_r_inv_col,
        &self.w_plus_r_inv_col,
        &self.poly_zero,
        &comb_func,
      );

    // row related evaluation points
    // 0 = ∑ eq[i] * (inv_T[i] * (T[i] + r) - TS[i]))
    let (eval_T_0_row, eval_T_2_row, eval_T_3_row) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.t_plus_r_inv_row,
        &self.t_plus_r_row,
        &self.ts_row,
        &comb_func3,
      );
    // 0 = ∑ eq[i] * (inv_W[i] * (T[i] + r) - 1))
    let (eval_W_0_row, eval_W_2_row, eval_W_3_row) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.w_plus_r_inv_row,
        &self.w_plus_r_row,
        &self.poly_zero,
        &comb_func2,
      );

    // column related evaluation points
    let (eval_T_0_col, eval_T_2_col, eval_T_3_col) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.t_plus_r_inv_col,
        &self.t_plus_r_col,
        &self.ts_col,
        &comb_func3,
      );
    let (eval_W_0_col, eval_W_2_col, eval_W_3_col) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.w_plus_r_inv_col,
        &self.w_plus_r_col,
        &self.poly_zero,
        &comb_func2,
      );

    vec![
      vec![eval_inv_0_row, eval_inv_2_row, eval_inv_3_row],
      vec![eval_inv_0_col, eval_inv_2_col, eval_inv_3_col],
      vec![eval_T_0_row, eval_T_2_row, eval_T_3_row],
      vec![eval_W_0_row, eval_W_2_row, eval_W_3_row],
      vec![eval_T_0_col, eval_T_2_col, eval_T_3_col],
      vec![eval_W_0_col, eval_W_2_col, eval_W_3_col],
    ]
  }

  fn bound(&mut self, r: &E::Scalar) {
    [
      &mut self.t_plus_r_row,
      &mut self.t_plus_r_inv_row,
      &mut self.w_plus_r_row,
      &mut self.w_plus_r_inv_row,
      &mut self.ts_row,
      &mut self.t_plus_r_col,
      &mut self.t_plus_r_inv_col,
      &mut self.w_plus_r_col,
      &mut self.w_plus_r_inv_col,
      &mut self.ts_col,
      &mut self.poly_eq,
    ]
    .par_iter_mut()
    .for_each(|poly| poly.bind_poly_var_top(r));
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
    let poly_row_final = vec![
      self.t_plus_r_inv_row[0],
      self.w_plus_r_inv_row[0],
      self.ts_row[0],
    ];

    let poly_col_final = vec![
      self.t_plus_r_inv_col[0],
      self.w_plus_r_inv_col[0],
      self.ts_col[0],
    ];

    vec![poly_row_final, poly_col_final]
  }
}

pub(in crate::spartan) struct OuterSumcheckInstance<E: Engine> {
  poly_tau: MultilinearPolynomial<E::Scalar>,
  poly_Az: MultilinearPolynomial<E::Scalar>,
  poly_Bz: MultilinearPolynomial<E::Scalar>,
  poly_uCz_E: MultilinearPolynomial<E::Scalar>,

  poly_Mz: MultilinearPolynomial<E::Scalar>,
  eval_Mz_at_tau: E::Scalar,

  poly_zero: MultilinearPolynomial<E::Scalar>,
}

impl<E: Engine> OuterSumcheckInstance<E> {
  pub fn new(
    tau: Vec<E::Scalar>,
    Az: Vec<E::Scalar>,
    Bz: Vec<E::Scalar>,
    uCz_E: Vec<E::Scalar>,
    Mz: Vec<E::Scalar>,
    eval_Mz_at_tau: &E::Scalar,
  ) -> Self {
    let zero = vec![E::Scalar::ZERO; tau.len()];
    Self {
      poly_tau: MultilinearPolynomial::new(tau),
      poly_Az: MultilinearPolynomial::new(Az),
      poly_Bz: MultilinearPolynomial::new(Bz),
      poly_uCz_E: MultilinearPolynomial::new(uCz_E),
      poly_Mz: MultilinearPolynomial::new(Mz),
      eval_Mz_at_tau: *eval_Mz_at_tau,
      poly_zero: MultilinearPolynomial::new(zero),
    }
  }
}

impl<E: Engine> SumcheckEngine<E> for OuterSumcheckInstance<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
    vec![E::Scalar::ZERO, self.eval_Mz_at_tau]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    assert_eq!(self.poly_tau.len(), self.poly_Az.len());
    assert_eq!(self.poly_tau.len(), self.poly_Bz.len());
    assert_eq!(self.poly_tau.len(), self.poly_uCz_E.len());
    assert_eq!(self.poly_tau.len(), self.poly_Mz.len());
    self.poly_tau.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>> {
    let comb_func =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    let (eval_point_h_0, eval_point_h_2, eval_point_h_3) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_tau,
        &self.poly_Az,
        &self.poly_Bz,
        &self.poly_uCz_E,
        &comb_func,
      );

    let comb_func2 = |poly_A_comp: &E::Scalar,
                      poly_B_comp: &E::Scalar,
                      _poly_C_comp: &E::Scalar|
     -> E::Scalar { *poly_A_comp * *poly_B_comp };

    let (eval_point_e_0, eval_point_e_2, eval_point_e_3) =
      SumcheckProof::<E>::compute_eval_points_cubic(
        &self.poly_tau,
        &self.poly_Mz,
        &self.poly_zero,
        &comb_func2,
      );

    vec![
      vec![eval_point_h_0, eval_point_h_2, eval_point_h_3],
      vec![eval_point_e_0, eval_point_e_2, eval_point_e_3],
    ]
  }

  fn bound(&mut self, r: &E::Scalar) {
    [
      &mut self.poly_tau,
      &mut self.poly_Az,
      &mut self.poly_Bz,
      &mut self.poly_uCz_E,
      &mut self.poly_Mz,
    ]
    .par_iter_mut()
    .for_each(|poly| poly.bind_poly_var_top(r));
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
    vec![vec![self.poly_Az[0], self.poly_Bz[0]]]
  }
}

pub(in crate::spartan) struct InnerSumcheckInstance<E: Engine> {
  pub(in crate::spartan) claim: E::Scalar,
  pub(in crate::spartan) poly_L_row: MultilinearPolynomial<E::Scalar>,
  pub(in crate::spartan) poly_L_col: MultilinearPolynomial<E::Scalar>,
  pub(in crate::spartan) poly_val: MultilinearPolynomial<E::Scalar>,
}
impl<E: Engine> InnerSumcheckInstance<E> {
  pub fn new(
    claim: E::Scalar,
    poly_L_row: MultilinearPolynomial<E::Scalar>,
    poly_L_col: MultilinearPolynomial<E::Scalar>,
    poly_val: MultilinearPolynomial<E::Scalar>,
  ) -> Self {
    Self {
      claim,
      poly_L_row,
      poly_L_col,
      poly_val,
    }
  }
}
impl<E: Engine> SumcheckEngine<E> for InnerSumcheckInstance<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
    vec![self.claim]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    assert_eq!(self.poly_L_row.len(), self.poly_val.len());
    assert_eq!(self.poly_L_row.len(), self.poly_L_col.len());
    self.poly_L_row.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>> {
    let (poly_A, poly_B, poly_C) = (&self.poly_L_row, &self.poly_L_col, &self.poly_val);
    let comb_func = |poly_A_comp: &E::Scalar,
                     poly_B_comp: &E::Scalar,
                     poly_C_comp: &E::Scalar|
     -> E::Scalar { *poly_A_comp * *poly_B_comp * *poly_C_comp };

    let (eval_point_0, eval_point_2, eval_point_3) =
      SumcheckProof::<E>::compute_eval_points_cubic(poly_A, poly_B, poly_C, &comb_func);

    vec![vec![eval_point_0, eval_point_2, eval_point_3]]
  }

  fn bound(&mut self, r: &E::Scalar) {
    [
      &mut self.poly_L_row,
      &mut self.poly_L_col,
      &mut self.poly_val,
    ]
    .par_iter_mut()
    .for_each(|poly| poly.bind_poly_var_top(r));
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
    vec![vec![self.poly_L_row[0], self.poly_L_col[0]]]
  }
}

pub(in crate::spartan) struct LookupSumcheckInstance<E: Engine> {
  w_plus_r: MultilinearPolynomial<E::Scalar>,
  t_plus_r: MultilinearPolynomial<E::Scalar>,
  w_plus_r_inv: MultilinearPolynomial<E::Scalar>,
  t_plus_r_inv: MultilinearPolynomial<E::Scalar>,
  ts: MultilinearPolynomial<E::Scalar>,

  // eq
  poly_eq: MultilinearPolynomial<E::Scalar>,

  // zero polynomial
  poly_zero: MultilinearPolynomial<E::Scalar>,

  // initial claim
  initial_claim: Option<E::Scalar>,
}

impl<E: Engine> LookupSumcheckInstance<E> {
  /// Computes witnesses for MemoryInstanceSumcheck
  ///
  /// # Description
  /// We use the logUp protocol to prove that
  /// ∑ TS[i]/(T[i] + r) - 1/(W[i] + r) = 0
  /// where
  ///   T[i] = t_sets[0][i] + t_sets[1][i] * gamma + t_sets[2][i] * gamma * gamma + ...
  ///   W[i] = w_sets[0][i] + w_sets[1][i] * gamma + t_sets[2][i] * gamma * gamma + ...
  /// and
  ///   TS are integer-valued vectors representing the number of reads
  ///   to each memory cell of L
  ///

  /// The function returns oracles for the polynomials TS[i]/(T[i] + r), 1/(W[i] + r),
  /// as well as auxiliary polynomials T[i] + r, W[i] + r
  // such that caller pass vector of iterators with respective to T and W part
  // and this function will return the oracle version of it
  pub fn compute_oracles<'a>(
    ck: &CommitmentKey<E>,
    r: &E::Scalar,
    gamma: &E::Scalar,
    mut t_sets: Vec<Box<dyn ExactSizeIterator<Item = E::Scalar> + Sync + Send + 'a>>,
    mut w_sets: Vec<Box<dyn ExactSizeIterator<Item = E::Scalar> + Sync + Send + 'a>>,
    ts: &[E::Scalar],
  ) -> Result<([Commitment<E>; 2], [Vec<E::Scalar>; 2], [Vec<E::Scalar>; 2]), NovaError> {
    // get power of gamma
    let hash_func_vec = |vectors: &mut Vec<
      Box<dyn ExactSizeIterator<Item = E::Scalar> + Sync + Send>,
    >|
     -> Result<Vec<E::Scalar>, NovaError> {
      let vector_len = vectors.len();
      assert!(vector_len > 0);
      let vector_size = vectors[0].len();
      let power_of_gamma = powers(gamma, vector_len);
      let hash_func = |values: Vec<E::Scalar>| -> E::Scalar {
        values
          .into_iter()
          .zip_eq(power_of_gamma.iter())
          .map(|(value, gamma)| value * *gamma)
          .sum()
      };
      (0..vector_size)
        .map(move |_| {
          let values = vectors
            .iter_mut()
            .map(|iterator| iterator.next().ok_or(NovaError::InternalError))
            .collect::<Result<Vec<_>, _>>()?;
          let value = hash_func(values);
          Ok(value)
        })
        .collect::<Result<Vec<_>, _>>()
    };

    let (T, W) = {
      let (t_sets, w_sets) =
        rayon::join(|| hash_func_vec(&mut t_sets), || hash_func_vec(&mut w_sets));
      (t_sets?, w_sets?)
    };

    let batch_invert = |v: &[E::Scalar]| -> Result<Vec<E::Scalar>, NovaError> {
      let mut products = vec![E::Scalar::ZERO; v.len()];
      let mut acc = E::Scalar::ONE;

      for i in 0..v.len() {
        products[i] = acc;
        acc *= v[i];
      }

      // we can compute an inversion only if acc is non-zero
      if acc == E::Scalar::ZERO {
        return Err(NovaError::InternalError);
      }

      // compute the inverse once for all entries
      acc = acc.invert().unwrap();

      let mut inv = vec![E::Scalar::ZERO; v.len()];
      for i in 0..v.len() {
        let tmp = acc * v[v.len() - 1 - i];
        inv[v.len() - 1 - i] = products[v.len() - 1 - i] * acc;
        acc = tmp;
      }

      Ok(inv)
    };

    // compute vectors TS[i]/(T[i] + r) and 1/(W[i] + r)
    let helper = |T: &[E::Scalar],
                  W: &[E::Scalar],
                  TS: &[E::Scalar],
                  r: &E::Scalar|
     -> (
      (
        Result<Vec<E::Scalar>, NovaError>, // T inv
        Result<Vec<E::Scalar>, NovaError>, // W inv
      ),
      (
        Result<Vec<E::Scalar>, NovaError>, // T
        Result<Vec<E::Scalar>, NovaError>, // W
      ),
    ) {
      rayon::join(
        || {
          rayon::join(
            || {
              let inv = batch_invert(&T.par_iter().map(|e| *e + *r).collect::<Vec<E::Scalar>>())?;

              // compute inv[i] * TS[i] in parallel
              Ok(
                zip_with!((inv.into_par_iter(), TS.into_par_iter()), |e1, e2| e1 * *e2)
                  .collect::<Vec<_>>(),
              )
            },
            || batch_invert(&W.par_iter().map(|e| *e + *r).collect::<Vec<E::Scalar>>()),
          )
        },
        || {
          rayon::join(
            || Ok(T.par_iter().map(|e| *e + *r).collect::<Vec<E::Scalar>>()),
            || Ok(W.par_iter().map(|e| *e + *r).collect::<Vec<E::Scalar>>()),
          )
        },
      )
    };

    let ((t_inv, w_inv), (t, w)) = {
      let ((t_inv, w_inv), (t, w)) = helper(&T, &W, ts, r);
      ((t_inv?, w_inv?), (t?, w?))
    };

    let comm_vec = {
      let (t_inv_comm, w_inv_comm) =
        rayon::join(|| E::CE::commit(ck, &t_inv), || E::CE::commit(ck, &w_inv));
      [t_inv_comm, w_inv_comm]
    };
    let poly_vec = [t_inv, w_inv];
    let aux_poly_vec = [t, w];

    Ok((comm_vec, poly_vec, aux_poly_vec))
  }

  pub fn new(
    polys_oracle: [Vec<E::Scalar>; 2],
    polys_aux: [Vec<E::Scalar>; 2],
    poly_eq: Vec<E::Scalar>,
    ts_row: Vec<E::Scalar>,
    initial_claim: Option<E::Scalar>,
  ) -> Self {
    let [t_plus_r_inv, w_plus_r_inv] = polys_oracle;
    let [t_plus_r, w_plus_r] = polys_aux;

    let zero = vec![E::Scalar::ZERO; poly_eq.len()];

    Self {
      initial_claim,
      t_plus_r: MultilinearPolynomial::new(t_plus_r),
      w_plus_r: MultilinearPolynomial::new(w_plus_r),
      t_plus_r_inv: MultilinearPolynomial::new(t_plus_r_inv),
      w_plus_r_inv: MultilinearPolynomial::new(w_plus_r_inv),
      ts: MultilinearPolynomial::new(ts_row),
      poly_eq: MultilinearPolynomial::new(poly_eq),
      poly_zero: MultilinearPolynomial::new(zero),
    }
  }
}

impl<E: Engine> SumcheckEngine<E> for LookupSumcheckInstance<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
    vec![
      self.initial_claim.unwrap_or_default(),
      E::Scalar::ZERO,
      E::Scalar::ZERO,
    ]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    // sanity checks
    assert_eq!(self.t_plus_r.len(), self.w_plus_r.len());
    assert_eq!(self.w_plus_r.len(), self.ts.len());

    self.w_plus_r.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>> {
    let comb_func = |poly_A_comp: &E::Scalar,
                     poly_B_comp: &E::Scalar,
                     _poly_C_comp: &E::Scalar|
     -> E::Scalar { *poly_A_comp - *poly_B_comp };

    let comb_func2 =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       _poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - E::Scalar::ONE) };

    let comb_func3 =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    // inv related evaluation points
    // v = ∑ TS[i]/(T[i] + r) - 1/(W[i] + r)
    let (eval_inv_0, eval_inv_2, eval_inv_3) = SumcheckProof::<E>::compute_eval_points_cubic(
      &self.t_plus_r_inv,
      &self.w_plus_r_inv,
      &self.poly_zero,
      &comb_func,
    );

    // evaluation points
    // 0 = ∑ eq[i] * (inv_T[i] * (T[i] + r) - TS[i]))
    let (eval_T_0, eval_T_2, eval_T_3) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.t_plus_r_inv,
        &self.t_plus_r,
        &self.ts,
        &comb_func3,
      );
    // 0 = ∑ eq[i] * (inv_W[i] * (W[i] + r) - 1))
    let (eval_W_0, eval_W_2, eval_W_3) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.w_plus_r_inv,
        &self.w_plus_r,
        &self.poly_zero,
        &comb_func2,
      );

    vec![
      vec![eval_inv_0, eval_inv_2, eval_inv_3],
      vec![eval_T_0, eval_T_2, eval_T_3],
      vec![eval_W_0, eval_W_2, eval_W_3],
    ]
  }

  fn bound(&mut self, r: &E::Scalar) {
    [
      &mut self.t_plus_r,
      &mut self.t_plus_r_inv,
      &mut self.w_plus_r,
      &mut self.w_plus_r_inv,
      &mut self.ts,
      &mut self.poly_eq,
    ]
    .par_iter_mut()
    .for_each(|poly| poly.bind_poly_var_top(r));
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
    let poly_final = vec![self.t_plus_r_inv[0], self.w_plus_r_inv[0], self.ts[0]];

    vec![poly_final]
  }
}
