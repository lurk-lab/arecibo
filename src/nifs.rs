//! This module implements a non-interactive folding scheme
#![allow(non_snake_case)]

use crate::{
  constants::{NUM_CHALLENGE_BITS, NUM_FE_FOR_RO, NUM_FE_WITHOUT_IO_FOR_NOVA_FOLD},
  errors::NovaError,
  r1cs::{
    R1CSInstance, R1CSResult, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
  },
  scalar_as_base,
  traits::{commitment::CommitmentTrait, AbsorbInROTrait, Engine, ROConstants, ROTrait},
  Commitment, CommitmentKey, CompressedCommitment,
};
use ff::Field;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

/// A SNARK that holds the proof of a step of an incremental computation
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E: Engine> {
  pub(crate) comm_T: CompressedCommitment<E>,
}

impl<E: Engine> NIFS<E> {
  /// Takes as input a Relaxed R1CS instance-witness tuple `(U1, W1)` and
  /// an R1CS instance-witness tuple `(U2, W2)` with the same structure `shape`
  /// and defined with respect to the same `ck`, and outputs
  /// a folded Relaxed R1CS instance-witness tuple `(U, W)` of the same shape `shape`,
  /// with the guarantee that the folded witness `W` satisfies the folded instance `U`
  /// if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  ///
  /// Note that this code is tailored for use with Nova's IVC scheme, which enforces
  /// certain requirements between the two instances that are folded.
  /// In particular, it requires that `U1` and `U2` are such that the hash of `U1` is stored in the public IO of `U2`.
  /// In this particular setting, this means that if `U2` is absorbed in the RO, it implicitly absorbs `U1` as well.
  /// So the code below avoids absorbing `U1` in the RO.
  #[allow(clippy::too_many_arguments)]
  #[tracing::instrument(skip_all, level = "trace", name = "NIFS::prove")]
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &ROConstants<E>,
    pp_digest: &E::Scalar,
    S: &R1CSShape<E>,
    U1: &RelaxedR1CSInstance<E>,
    W1: &RelaxedR1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) -> Result<
    (
      Self,
      (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>),
      E::Scalar,
    ),
    NovaError,
  > {
    // Check `U1` and `U2` have the same arity
    let io_arity = U1.X.len();
    if io_arity != U2.X.len() {
      return Err(NovaError::InvalidInputLength);
    }

    // initialize a new RO
    let mut ro = E::RO::new(
      ro_consts.clone(),
      NUM_FE_WITHOUT_IO_FOR_NOVA_FOLD + io_arity,
    );

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<E>(*pp_digest));

    // append U2 to transcript, U1 does not need to absorbed since U2.X[0] = Hash(params, U1, i, z0, zi)
    U2.absorb_in_ro(&mut ro);

    // compute a commitment to the cross-term
    let r_T = E::Scalar::random(&mut OsRng);
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2, &r_T)?;

    // append `comm_T` to the transcript and obtain a challenge
    comm_T.absorb_in_ro(&mut ro);

    // compute a challenge from the RO
    let r = ro.squeeze(NUM_CHALLENGE_BITS);

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &comm_T, &r);

    // fold the witness using `r` and `T`
    let W = W1.fold(W2, &T, &r_T, &r)?;

    // return the folded instance and witness
    Ok((
      Self {
        comm_T: comm_T.compress(),
      },
      (U, W),
      r,
    ))
  }

  /// Takes as input a Relaxed R1CS instance-witness tuple `(U1, W1)` and
  /// an R1CS instance-witness tuple `(U2, W2)` with the same structure `shape`
  /// and defined with respect to the same `ck`, and updates `(U1, W1)` by folding
  /// `(U2, W2)` into it with the guarantee that the updated witness `W` satisfies
  /// the updated instance `U` if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  #[allow(clippy::too_many_arguments)]
  #[tracing::instrument(skip_all, level = "trace", name = "NIFS::prove_mut")]
  pub fn prove_mut(
    ck: &CommitmentKey<E>,
    ro_consts: &ROConstants<E>,
    pp_digest: &E::Scalar,
    S: &R1CSShape<E>,
    U1: &mut RelaxedR1CSInstance<E>,
    W1: &mut RelaxedR1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    T: &mut Vec<E::Scalar>,
    ABC_Z_1: &mut R1CSResult<E>,
    ABC_Z_2: &mut R1CSResult<E>,
  ) -> Result<(Self, E::Scalar), NovaError> {
    // initialize a new RO
    let mut ro = E::RO::new(ro_consts.clone(), NUM_FE_FOR_RO);

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<E>(*pp_digest));

    // append U2 to transcript, U1 does not need to absorbed since U2.X[0] = Hash(params, U1, i, z0, zi)
    U2.absorb_in_ro(&mut ro);

    // compute a commitment to the cross-term
    let r_T = E::Scalar::random(&mut OsRng);
    let comm_T = S.commit_T_into(ck, U1, W1, U2, W2, T, ABC_Z_1, ABC_Z_2, &r_T)?;

    // append `comm_T` to the transcript and obtain a challenge
    comm_T.absorb_in_ro(&mut ro);

    // compute a challenge from the RO
    let r = ro.squeeze(NUM_CHALLENGE_BITS);

    // fold the instance using `r` and `comm_T`
    U1.fold_mut(U2, &comm_T, &r);

    // fold the witness using `r` and `T`
    W1.fold_mut(W2, T, &r_T, &r)?;

    // return the commitment
    Ok((
      Self {
        comm_T: comm_T.compress(),
      },
      r,
    ))
  }

  /// Takes as input a relaxed R1CS instance `U1` and R1CS instance `U2`
  /// with the same shape and defined with respect to the same parameters,
  /// and outputs a folded instance `U` with the same shape,
  /// with the guarantee that the folded instance `U`
  /// if and only if `U1` and `U2` are satisfiable.
  pub fn verify(
    &self,
    ro_consts: &ROConstants<E>,
    pp_digest: &E::Scalar,
    U1: &RelaxedR1CSInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<RelaxedR1CSInstance<E>, NovaError> {
    // initialize a new RO
    let mut ro = E::RO::new(ro_consts.clone(), NUM_FE_FOR_RO);

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<E>(*pp_digest));

    // append U2 to transcript, U1 does not need to absorbed since U2.X[0] = Hash(params, U1, i, z0, zi)
    U2.absorb_in_ro(&mut ro);

    // append `comm_T` to the transcript and obtain a challenge
    let comm_T = Commitment::<E>::decompress(&self.comm_T)?;
    comm_T.absorb_in_ro(&mut ro);

    // compute a challenge from the RO
    let r = ro.squeeze(NUM_CHALLENGE_BITS);

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &comm_T, &r);

    // return the folded instance
    Ok(U)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    bellpepper::{
      r1cs::{NovaShape, NovaWitness},
      solver::SatisfyingAssignment,
      test_shape_cs::TestShapeCS,
    },
    provider::{PallasEngine, Secp256k1Engine},
    r1cs::commitment_key,
    traits::{snark::default_ck_hint, Engine},
  };
  use ::bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::{Field, PrimeField};
  use rand::rngs::OsRng;

  fn synthesize_tiny_r1cs_bellpepper<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    x_val: Option<Scalar>,
  ) -> Result<(), SynthesisError> {
    // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = AllocatedNum::alloc_infallible(cs.namespace(|| "x"), || x_val.unwrap());
    let _ = x.inputize(cs.namespace(|| "x is input"));

    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + Scalar::from(5u64))
    })?;
    let _ = y.inputize(cs.namespace(|| "y is output"));

    cs.enforce(
      || "y = x^3 + x + 5",
      |lc| {
        lc + x_cu.get_variable()
          + x.get_variable()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
      },
      |lc| lc + CS::one(),
      |lc| lc + y.get_variable(),
    );

    Ok(())
  }

  fn test_tiny_r1cs_bellpepper_with<E: Engine>() {
    // First create the shape
    let mut cs: TestShapeCS<E> = TestShapeCS::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, None);
    let (shape, ck) = cs.r1cs_shape_and_key(&*default_ck_hint());
    let ro_consts =
      <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants::default();

    // Now get the instance and assignment for one instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(5)));
    let (U1, W1) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the first instance is satisfiable
    shape.is_sat(&ck, &U1, &W1).unwrap();

    // Now get the instance and assignment for second instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(135)));
    let (U2, W2) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the second instance is satisfiable
    shape.is_sat(&ck, &U2, &W2).unwrap();

    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<E as Engine>::Scalar::ZERO,
      &shape,
      &U1,
      &W1,
      &U2,
      &W2,
    );
  }

  #[test]
  fn test_tiny_r1cs_bellpepper() {
    test_tiny_r1cs_bellpepper_with::<PallasEngine>();
    // test_tiny_r1cs_bellpepper_with::<Bn256EngineKZG>();
    test_tiny_r1cs_bellpepper_with::<Secp256k1Engine>();
  }

  fn execute_sequence<E: Engine>(
    ck: &CommitmentKey<E>,
    ro_consts: &<<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants,
    pp_digest: &<E as Engine>::Scalar,
    shape: &R1CSShape<E>,
    U1: &R1CSInstance<E>,
    W1: &R1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) {
    // produce a default running instance
    let mut r_W = RelaxedR1CSWitness::default(shape);
    let mut r_U = RelaxedR1CSInstance::default(ck, shape);

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let res = NIFS::prove(ck, ro_consts, pp_digest, shape, &r_U, &r_W, U1, W1);
    assert!(res.is_ok());
    let (nifs, (_U, W), _) = res.unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &r_U, U1);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;

    // produce a step SNARK with (W2, U2) as the second incoming witness-instance pair
    let res = NIFS::prove(ck, ro_consts, pp_digest, shape, &r_U, &r_W, U2, W2);
    assert!(res.is_ok());
    let (nifs, (_U, W), _) = res.unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &r_U, U2);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;

    // check if the running instance is satisfiable
    shape.is_sat_relaxed(ck, &r_U, &r_W).unwrap();
  }

  fn test_tiny_r1cs_with<E: Engine>() {
    let num_vars = 3;
    let S = crate::r1cs::tests::tiny_r1cs::<E>(num_vars);
    let one = <E::Scalar as Field>::ONE;

    // generate generators and ro constants
    let ck = commitment_key(&S, &*default_ck_hint());
    let ro_consts =
      <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants::default();

    let rand_inst_witness_generator =
      |ck: &CommitmentKey<E>, I: &E::Scalar| -> (E::Scalar, R1CSInstance<E>, R1CSWitness<E>) {
        let i0 = *I;

        // compute a satisfying (vars, X) tuple
        let (O, vars, X) = {
          let z0 = i0 * i0; // constraint 0
          let z1 = i0 * z0; // constraint 1
          let z2 = z1 + i0; // constraint 2
          let i1 = z2 + one + one + one + one + one; // constraint 3

          // store the witness and IO for the instance
          let W = vec![z0, z1, z2];
          let X = vec![i0, i1];
          (i1, W, X)
        };

        let W = {
          let res = R1CSWitness::new(&S, vars);
          assert!(res.is_ok());
          res.unwrap()
        };
        let U = {
          let comm_W = W.commit(ck);
          let res = R1CSInstance::new(&S, comm_W, X);
          assert!(res.is_ok());
          res.unwrap()
        };

        // check that generated instance is satisfiable
        S.is_sat(ck, &U, &W).unwrap();

        (O, U, W)
      };

    let mut csprng: OsRng = OsRng;
    let I = E::Scalar::random(&mut csprng); // the first input is picked randomly for the first instance
    let (O, U1, W1) = rand_inst_witness_generator(&ck, &I);
    let (_O, U2, W2) = rand_inst_witness_generator(&ck, &O);

    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<E as Engine>::Scalar::ZERO,
      &S,
      &U1,
      &W1,
      &U2,
      &W2,
    );
  }

  #[test]
  fn test_tiny_r1cs() {
    test_tiny_r1cs_with::<PallasEngine>();
    // test_tiny_r1cs_with::<Bn256EngineKZG>();
    test_tiny_r1cs_with::<Secp256k1Engine>();
  }
}
