//! This module defines the Cyclefold `RecursiveSNARK` type with its `new`, `prove_step`, and
//! `verify` methods.

use crate::{
  bellpepper::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  constants::{
    BN_LIMB_WIDTH, BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS, NUM_FE_IN_EMULATED_POINT,
    NUM_HASH_BITS,
  },
  cyclefold::circuit::CycleFoldCircuit,
  errors::NovaError,
  gadgets::scalar_as_base,
  r1cs::{
    self, CommitmentKeyHint, R1CSInstance, R1CSResult, R1CSWitness, RelaxedR1CSInstance,
    RelaxedR1CSWitness,
  },
  traits::{
    circuit::StepCircuit, commitment::CommitmentTrait, AbsorbInROTrait, CurveCycleEquipped, Dual,
    Engine, ROConstantsCircuit, ROTrait,
  },
  Commitment, CommitmentKey, DigestComputer, R1CSWithArity, ROConstants, ResourceBuffer,
  SimpleDigestible,
};

use super::{
  nifs::{CycleFoldNIFS, PrimaryNIFS},
  nova_circuit::{AugmentedCircuit, AugmentedCircuitInputs, AugmentedCircuitParams},
  util::{absorb_primary_relaxed_r1cs, FoldingData},
};

use abomonation::Abomonation;
use abomonation_derive::Abomonation;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use ff::{PrimeField, PrimeFieldBits};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};

/// The public parameters used in the CycleFold recursive SNARK proof and verification
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Abomonation)]
#[serde(bound = "")]
#[abomonation_bounds(
where
  E1: CurveCycleEquipped,
  <E1::Scalar as PrimeField>::Repr: Abomonation,
  <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
)]
pub struct PublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  F_arity_primary: usize,
  ro_consts_primary: ROConstants<Dual<E1>>,
  ro_consts_circuit_primary: ROConstantsCircuit<Dual<E1>>,
  ck_primary: CommitmentKey<E1>,
  circuit_shape_primary: R1CSWithArity<E1>,
  augmented_circuit_params: AugmentedCircuitParams,

  ro_consts_cyclefold: ROConstants<Dual<E1>>,
  ck_cyclefold: CommitmentKey<Dual<E1>>,
  circuit_shape_cyclefold: R1CSWithArity<Dual<E1>>,
  #[abomonation_skip]
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E1::Scalar>,
}

impl<E1> PublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  /// Builds the public parameters for the circuit `C1`.
  /// The same note for public parameter hints apply as in the case for Nova's public parameters:
  /// For some final compressing SNARKs the size of the commitment key must be larger, so we include
  /// `ck_hint_primary` and `ck_hint_cyclefold` parameters to accommodate this.
  pub fn setup<C1: StepCircuit<E1::Scalar>>(
    c_primary: &C1,
    ck_hint_primary: &CommitmentKeyHint<E1>,
    ck_hint_cyclefold: &CommitmentKeyHint<Dual<E1>>,
  ) -> Self {
    let F_arity_primary = c_primary.arity();
    let ro_consts_primary = ROConstants::<Dual<E1>>::default();
    let ro_consts_circuit_primary = ROConstantsCircuit::<Dual<E1>>::default();

    let augmented_circuit_params = AugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS);
    let circuit_primary: AugmentedCircuit<'_, Dual<E1>, E1, C1> = AugmentedCircuit::new(
      &augmented_circuit_params,
      ro_consts_circuit_primary.clone(),
      None,
      c_primary,
    );
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);
    let (r1cs_shape_primary, ck_primary) = cs.r1cs_shape_and_key(ck_hint_primary);
    let circuit_shape_primary = R1CSWithArity::new(r1cs_shape_primary, F_arity_primary);

    let ro_consts_cyclefold = ROConstants::<Dual<E1>>::default();
    let mut cs: ShapeCS<Dual<E1>> = ShapeCS::new();
    let circuit_cyclefold: CycleFoldCircuit<E1> = CycleFoldCircuit::default();
    let _ = circuit_cyclefold.synthesize(&mut cs);
    let (r1cs_shape_cyclefold, ck_cyclefold) = cs.r1cs_shape_and_key(ck_hint_cyclefold);
    let circuit_shape_cyclefold = R1CSWithArity::new(r1cs_shape_cyclefold, 0);

    Self {
      F_arity_primary,
      ro_consts_primary,
      ro_consts_circuit_primary,
      ck_primary,
      circuit_shape_primary,
      augmented_circuit_params,
      ro_consts_cyclefold,
      ck_cyclefold,
      circuit_shape_cyclefold,
      digest: OnceCell::new(),
    }
  }

  /// Calculate the digest of the public parameters.
  pub fn digest(&self) -> E1::Scalar {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .expect("Failure in retrieving digest")
  }

  /// Returns the number of constraints in the primary and cyclefold circuits
  pub const fn num_constraints(&self) -> (usize, usize) {
    (
      self.circuit_shape_primary.r1cs_shape.num_cons,
      self.circuit_shape_cyclefold.r1cs_shape.num_cons,
    )
  }

  /// Returns the number of variables in the primary and cyclefold circuits
  pub const fn num_variables(&self) -> (usize, usize) {
    (
      self.circuit_shape_primary.r1cs_shape.num_vars,
      self.circuit_shape_cyclefold.r1cs_shape.num_vars,
    )
  }
}

impl<E1> SimpleDigestible for PublicParams<E1> where E1: CurveCycleEquipped {}

/// A SNARK that proves the correct execution of an incremental computation in the CycleFold folding
/// scheme.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<E1>
where
  E1: CurveCycleEquipped,
{
  // Input
  z0_primary: Vec<E1::Scalar>,

  // primary circuit data
  r_W_primary: RelaxedR1CSWitness<E1>,
  r_U_primary: RelaxedR1CSInstance<E1>,
  l_w_primary: R1CSWitness<E1>,
  l_u_primary: R1CSInstance<E1>,

  // cyclefold circuit data
  r_W_cyclefold: RelaxedR1CSWitness<Dual<E1>>,
  r_U_cyclefold: RelaxedR1CSInstance<Dual<E1>>,

  // memory buffers for folding steps
  buffer_primary: ResourceBuffer<E1>,
  buffer_cyclefold: ResourceBuffer<Dual<E1>>,

  i: usize,
  zi_primary: Vec<E1::Scalar>,
}

impl<E1> RecursiveSNARK<E1>
where
  E1: CurveCycleEquipped,
{
  /// Create a new instance of a recursive SNARK
  pub fn new<C1: StepCircuit<E1::Scalar>>(
    pp: &PublicParams<E1>,
    c_primary: &C1,
    z0_primary: &[E1::Scalar],
  ) -> Result<Self, NovaError> {
    if z0_primary.len() != pp.F_arity_primary {
      return Err(NovaError::InvalidInitialInputLength);
    }

    let r1cs_primary = &pp.circuit_shape_primary.r1cs_shape;
    let r1cs_cyclefold = &pp.circuit_shape_cyclefold.r1cs_shape;

    let r_U_cyclefold = RelaxedR1CSInstance::default(&pp.ck_cyclefold, r1cs_cyclefold);
    let r_W_cyclefold = RelaxedR1CSWitness::default(r1cs_cyclefold);

    let mut cs_primary = SatisfyingAssignment::<E1>::new();
    let inputs_primary: AugmentedCircuitInputs<Dual<E1>, E1> = AugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      <Dual<E1> as Engine>::Base::from(0u64),
      z0_primary.to_vec(),
      None,
      None,
      None,
      None,
      None,
      None,
    );

    let circuit_primary = AugmentedCircuit::new(
      &pp.augmented_circuit_params,
      pp.ro_consts_circuit_primary.clone(),
      Some(inputs_primary),
      c_primary,
    );

    let zi_primary = circuit_primary.synthesize(&mut cs_primary)?;
    let (l_u_primary, l_w_primary) =
      cs_primary.r1cs_instance_and_witness(r1cs_primary, &pp.ck_primary)?;

    let r_U_primary = RelaxedR1CSInstance::default(&pp.ck_primary, r1cs_primary);
    let r_W_primary = RelaxedR1CSWitness::default(r1cs_primary);

    let zi_primary = zi_primary
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<_>, _>>()?;

    let buffer_primary = ResourceBuffer {
      l_w: None,
      l_u: None,
      ABC_Z_1: R1CSResult::default(r1cs_primary.num_cons),
      ABC_Z_2: R1CSResult::default(r1cs_primary.num_cons),
      T: r1cs::default_T::<E1>(r1cs_primary.num_cons),
    };

    let buffer_cyclefold = ResourceBuffer {
      l_w: None,
      l_u: None,
      ABC_Z_1: R1CSResult::default(r1cs_cyclefold.num_cons),
      ABC_Z_2: R1CSResult::default(r1cs_cyclefold.num_cons),
      T: r1cs::default_T::<Dual<E1>>(r1cs_cyclefold.num_cons),
    };

    Ok(Self {
      z0_primary: z0_primary.to_vec(),
      r_W_primary,
      r_U_primary,
      l_w_primary,
      l_u_primary,
      r_W_cyclefold,
      r_U_cyclefold,
      buffer_primary,
      buffer_cyclefold,
      i: 0,
      zi_primary,
    })
  }

  /// Update the `RecursiveSNARK` by proving a step of the incremental computation.
  pub fn prove_step<C1: StepCircuit<E1::Scalar>>(
    &mut self,
    pp: &PublicParams<E1>,
    c_primary: &C1,
  ) -> Result<(), NovaError> {
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    let (nifs_primary, (r_U_primary, r_W_primary), r) = PrimaryNIFS::<E1, Dual<E1>>::prove(
      &pp.ck_primary,
      &pp.ro_consts_primary,
      &pp.digest(),
      &pp.circuit_shape_primary.r1cs_shape,
      &self.r_U_primary,
      &self.r_W_primary,
      &self.l_u_primary,
      &self.l_w_primary,
    )?;

    let r_bools = r
      .to_le_bits()
      .iter()
      .map(|b| Some(*b))
      .take(NUM_CHALLENGE_BITS)
      .collect::<Option<Vec<_>>>()
      .map(|v| v.try_into().unwrap());

    let comm_T = Commitment::<E1>::decompress(&nifs_primary.comm_T)?;
    let E_new = self.r_U_primary.comm_E + comm_T * r;

    let W_new = self.r_U_primary.comm_W + self.l_u_primary.comm_W * r;

    let mut cs_cyclefold_E = SatisfyingAssignment::<Dual<E1>>::with_capacity(
      pp.circuit_shape_cyclefold.r1cs_shape.num_io + 1,
      pp.circuit_shape_cyclefold.r1cs_shape.num_vars,
    );

    let circuit_cyclefold_E: CycleFoldCircuit<E1> =
      CycleFoldCircuit::new(Some(self.r_U_primary.comm_E), Some(comm_T), r_bools);

    let _ = circuit_cyclefold_E.synthesize(&mut cs_cyclefold_E);

    let (l_u_cyclefold_E, l_w_cyclefold_E) = cs_cyclefold_E
      .r1cs_instance_and_witness(&pp.circuit_shape_cyclefold.r1cs_shape, &pp.ck_cyclefold)
      .map_err(|_| NovaError::UnSat)?;

    // TODO: check if this is better or worse than `prove_mut` with a clone of `self.r_U_cyclefold`
    let (nifs_cyclefold_E, (r_U_cyclefold_E, r_W_cyclefold_E)) = CycleFoldNIFS::prove(
      &pp.ck_cyclefold,
      &pp.ro_consts_cyclefold,
      &scalar_as_base::<E1>(pp.digest()),
      &pp.circuit_shape_cyclefold.r1cs_shape,
      &self.r_U_cyclefold,
      &self.r_W_cyclefold,
      &l_u_cyclefold_E,
      &l_w_cyclefold_E,
    )?;

    let comm_T_E = Commitment::<Dual<E1>>::decompress(&nifs_cyclefold_E.comm_T)?;

    let mut cs_cyclefold_W = SatisfyingAssignment::<Dual<E1>>::with_capacity(
      pp.circuit_shape_cyclefold.r1cs_shape.num_io + 1,
      pp.circuit_shape_cyclefold.r1cs_shape.num_vars,
    );

    let circuit_cyclefold_W: CycleFoldCircuit<E1> = CycleFoldCircuit::new(
      Some(self.r_U_primary.comm_W),
      Some(self.l_u_primary.comm_W),
      r_bools,
    );

    let _ = circuit_cyclefold_W.synthesize(&mut cs_cyclefold_W);

    let (l_u_cyclefold_W, l_w_cyclefold_W) = cs_cyclefold_W
      .r1cs_instance_and_witness(&pp.circuit_shape_cyclefold.r1cs_shape, &pp.ck_cyclefold)
      .map_err(|_| NovaError::UnSat)?;

    // TODO: check if this is better or worse than `prove_mut` with a clone of r_U_cyclefold_E
    let (nifs_cyclefold_W, (r_U_cyclefold_W, r_W_cyclefold_W)) = CycleFoldNIFS::prove(
      &pp.ck_cyclefold,
      &pp.ro_consts_cyclefold,
      &scalar_as_base::<E1>(pp.digest()),
      &pp.circuit_shape_cyclefold.r1cs_shape,
      &r_U_cyclefold_E,
      &r_W_cyclefold_E,
      &l_u_cyclefold_W,
      &l_w_cyclefold_W,
    )?;

    let comm_T_W = Commitment::<Dual<E1>>::decompress(&nifs_cyclefold_W.comm_T)?;

    let mut cs_primary = SatisfyingAssignment::<E1>::with_capacity(
      pp.circuit_shape_primary.r1cs_shape.num_io + 1,
      pp.circuit_shape_primary.r1cs_shape.num_vars,
    );

    let data_p = FoldingData::new(self.r_U_primary.clone(), self.l_u_primary.clone(), comm_T);
    let data_c_E = FoldingData::new(self.r_U_cyclefold.clone(), l_u_cyclefold_E, comm_T_E);
    let data_c_W = FoldingData::new(r_U_cyclefold_E, l_u_cyclefold_W, comm_T_W);

    let inputs_primary: AugmentedCircuitInputs<Dual<E1>, E1> = AugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      <Dual<E1> as Engine>::Base::from(self.i as u64),
      self.z0_primary.clone(),
      Some(self.zi_primary.clone()),
      Some(data_p),
      Some(data_c_E),
      Some(data_c_W),
      Some(E_new),
      Some(W_new),
    );

    let circuit_primary: AugmentedCircuit<'_, Dual<E1>, E1, C1> = AugmentedCircuit::new(
      &pp.augmented_circuit_params,
      pp.ro_consts_circuit_primary.clone(),
      Some(inputs_primary),
      c_primary,
    );

    let zi_primary = circuit_primary.synthesize(&mut cs_primary)?;

    let (l_u_primary, l_w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.circuit_shape_primary.r1cs_shape, &pp.ck_primary)
      .map_err(|_| NovaError::UnSat)?;

    self.zi_primary = zi_primary
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<_>, _>>()?;

    self.r_U_primary = r_U_primary;
    self.r_W_primary = r_W_primary;
    self.l_u_primary = l_u_primary;
    self.l_w_primary = l_w_primary;
    self.r_U_cyclefold = r_U_cyclefold_W;
    self.r_W_cyclefold = r_W_cyclefold_W;

    self.i += 1;

    Ok(())
  }

  /// Verify the correctness of the `RecursiveSNARK`
  pub fn verify(
    &self,
    pp: &PublicParams<E1>,
    num_steps: usize,
    z0_primary: &[E1::Scalar],
  ) -> Result<Vec<E1::Scalar>, NovaError> {
    // number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;

    // check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;

    // check if the initial inputs match
    let is_inputs_not_match = self.z0_primary != z0_primary;

    // check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outputs = self.r_U_primary.X.len() != 2;

    if is_num_steps_zero
      || is_num_steps_not_match
      || is_inputs_not_match
      || is_instance_has_two_outputs
    {
      return Err(NovaError::ProofVerifyError);
    }

    // Calculate the hashes of the primary running instance and cyclefold running instance
    let (hash_primary, hash_cyclefold) = {
      let mut hasher = <Dual<E1> as Engine>::RO::new(
        pp.ro_consts_primary.clone(),
        2 + 2 * pp.F_arity_primary + 2 * NUM_FE_IN_EMULATED_POINT + 3,
      );
      hasher.absorb(pp.digest());
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0_primary {
        hasher.absorb(*e);
      }
      for e in &self.zi_primary {
        hasher.absorb(*e);
      }
      absorb_primary_relaxed_r1cs::<E1, Dual<E1>>(&self.r_U_primary, &mut hasher);
      let hash_primary = hasher.squeeze(NUM_HASH_BITS);

      let mut hasher = <Dual<E1> as Engine>::RO::new(
        pp.ro_consts_cyclefold.clone(),
        1 + 1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS,
      );
      hasher.absorb(pp.digest());
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      self.r_U_cyclefold.absorb_in_ro(&mut hasher);
      let hash_cyclefold = hasher.squeeze(NUM_HASH_BITS);

      (hash_primary, hash_cyclefold)
    };

    // Verify the hashes equal the public IO for the final primary instance
    if scalar_as_base::<Dual<E1>>(hash_primary) != self.l_u_primary.X[0]
      || scalar_as_base::<Dual<E1>>(hash_cyclefold) != self.l_u_primary.X[1]
    {
      return Err(NovaError::ProofVerifyError);
    }

    // Verify the satisfiability of running relaxed instances, and the final primary instance.
    let (res_r_primary, (res_l_primary, res_r_cyclefold)) = rayon::join(
      || {
        pp.circuit_shape_primary.r1cs_shape.is_sat_relaxed(
          &pp.ck_primary,
          &self.r_U_primary,
          &self.r_W_primary,
        )
      },
      || {
        rayon::join(
          || {
            pp.circuit_shape_primary.r1cs_shape.is_sat(
              &pp.ck_primary,
              &self.l_u_primary,
              &self.l_w_primary,
            )
          },
          || {
            pp.circuit_shape_cyclefold.r1cs_shape.is_sat_relaxed(
              &pp.ck_cyclefold,
              &self.r_U_cyclefold,
              &self.r_W_cyclefold,
            )
          },
        )
      },
    );

    res_r_primary?;
    res_l_primary?;
    res_r_cyclefold?;

    Ok(self.zi_primary.clone())
  }
}

#[cfg(test)]
mod test {
  use std::marker::PhantomData;

  use bellpepper_core::num::AllocatedNum;

  use super::*;
  use crate::{
    provider::{PallasEngine, Secp256k1Engine},
    traits::snark::default_ck_hint, StepCounterType,
  };

  #[derive(Clone)]
  struct SquareCircuit<F> {
    _p: PhantomData<F>,
    counter_type: StepCounterType,
  }

  impl<F: PrimeField> StepCircuit<F> for SquareCircuit<F> {
    fn arity(&self) -> usize {
      1
    }

    fn get_counter_type(&self) -> StepCounterType {
      self.counter_type
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let x = &z[0];
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;

      Ok(vec![x_sq])
    }
  }

  fn test_trivial_cyclefold_prove_verify_with<E: CurveCycleEquipped>() {
    let primary_circuit = SquareCircuit::<E::Scalar> { _p: PhantomData, counter_type: StepCounterType::Incremental };

    let pp = PublicParams::<E>::setup(&primary_circuit, &*default_ck_hint(), &*default_ck_hint());

    let z0 = vec![E::Scalar::from(2u64)];

    let mut recursive_snark = RecursiveSNARK::new(&pp, &primary_circuit, &z0).unwrap();

    (1..5).for_each(|iter| {
      let res_proof = recursive_snark.prove_step(&pp, &primary_circuit);
      res_proof.unwrap();

      let res_verify = recursive_snark.verify(&pp, iter, &z0);
      res_verify.unwrap();
    });
  }

  #[test]
  fn test_cyclefold_prove_verify() {
    test_trivial_cyclefold_prove_verify_with::<PallasEngine>();
    // test_trivial_cyclefold_prove_verify_with::<Bn256EngineKZG>();
    test_trivial_cyclefold_prove_verify_with::<Secp256k1Engine>();
  }
}
