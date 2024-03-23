//! This library implements Nova, a high-speed recursive SNARK.
#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
// #![forbid(unsafe_code)] // Commented for development with `Abomonation`

// private modules
mod bellpepper;
mod circuit;
mod digest;
mod nifs;

// public modules
pub mod constants;
pub mod errors;
pub mod gadgets;
pub mod provider;
pub mod r1cs;
pub mod spartan;
pub mod traits;

pub mod cyclefold;
pub mod supernova;

use once_cell::sync::OnceCell;
use traits::{CurveCycleEquipped, Dual};

use crate::digest::{DigestComputer, SimpleDigestible};
use crate::{
  bellpepper::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  r1cs::R1CSResult,
};
use abomonation::Abomonation;
use abomonation_derive::Abomonation;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use circuit::{NovaAugmentedCircuit, NovaAugmentedCircuitInputs, NovaAugmentedCircuitParams};
use constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS};
use errors::NovaError;
use ff::{Field, PrimeField};
use gadgets::scalar_as_base;
use nifs::NIFS;
use r1cs::{
  CommitmentKeyHint, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use traits::{
  circuit::StepCircuit,
  commitment::{CommitmentEngineTrait, CommitmentTrait},
  snark::RelaxedR1CSSNARKTrait,
  AbsorbInROTrait, Engine, ROConstants, ROConstantsCircuit, ROTrait,
};

/// A type that holds parameters for the primary and secondary circuits of Nova and SuperNova
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Abomonation)]
#[serde(bound = "")]
#[abomonation_bounds(where <E::Scalar as PrimeField>::Repr: Abomonation)]
pub struct R1CSWithArity<E: Engine> {
  F_arity: usize,
  r1cs_shape: R1CSShape<E>,
}

impl<E: Engine> SimpleDigestible for R1CSWithArity<E> {}

impl<E: Engine> R1CSWithArity<E> {
  /// Create a new `R1CSWithArity`
  pub fn new(r1cs_shape: R1CSShape<E>, F_arity: usize) -> Self {
    Self {
      F_arity,
      r1cs_shape,
    }
  }

  /// Return the [`R1CSWithArity`]' digest.
  pub fn digest(&self) -> E::Scalar {
    let dc: DigestComputer<'_, <E as Engine>::Scalar, Self> = DigestComputer::new(self);
    dc.digest().expect("Failure in computing digest")
  }
}

/// A type that holds public parameters of Nova
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<E>
where
  E: CurveCycleEquipped,
{
  F_arity_primary: usize,
  F_arity_secondary: usize,
  ro_consts_primary: ROConstants<E>,
  ro_consts_circuit_primary: ROConstantsCircuit<Dual<E>>,
  ck_primary: Arc<CommitmentKey<E>>,
  circuit_shape_primary: R1CSWithArity<E>,
  ro_consts_secondary: ROConstants<Dual<E>>,
  ro_consts_circuit_secondary: ROConstantsCircuit<E>,
  ck_secondary: Arc<CommitmentKey<Dual<E>>>,
  circuit_shape_secondary: R1CSWithArity<Dual<E>>,
  augmented_circuit_params_primary: NovaAugmentedCircuitParams,
  augmented_circuit_params_secondary: NovaAugmentedCircuitParams,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

// Ensure to include necessary crates and features in your Cargo.toml
// e.g., abomonation, serde, etc., with the appropriate feature flags.

/// A version of [`crate::PublicParams`] that is amenable to fast ser/de using Abomonation
#[cfg(feature = "abomonate")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Abomonation)]
#[serde(bound = "")]
#[abomonation_bounds(
where
  E1: CurveCycleEquipped,
  <E1::Scalar as PrimeField>::Repr: Abomonation,
  <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
)]
pub struct FlatPublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  F_arity_primary: usize,
  F_arity_secondary: usize,
  ro_consts_primary: ROConstants<E1>,
  ro_consts_circuit_primary: ROConstantsCircuit<Dual<E1>>,
  ck_primary: CommitmentKey<E1>,
  circuit_shape_primary: R1CSWithArity<E1>,
  ro_consts_secondary: ROConstants<Dual<E1>>,
  ro_consts_circuit_secondary: ROConstantsCircuit<E1>,
  ck_secondary: CommitmentKey<Dual<E1>>,
  circuit_shape_secondary: R1CSWithArity<Dual<E1>>,
  augmented_circuit_params_primary: NovaAugmentedCircuitParams,
  augmented_circuit_params_secondary: NovaAugmentedCircuitParams,
}

#[cfg(feature = "abomonate")]
impl<E1> TryFrom<PublicParams<E1>> for FlatPublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  type Error = &'static str;

  fn try_from(value: PublicParams<E1>) -> Result<Self, Self::Error> {
    let ck_primary =
      Arc::try_unwrap(value.ck_primary).map_err(|_| "Failed to unwrap Arc for ck_primary")?;
    let ck_secondary =
      Arc::try_unwrap(value.ck_secondary).map_err(|_| "Failed to unwrap Arc for ck_secondary")?;
    Ok(Self {
      F_arity_primary: value.F_arity_primary,
      F_arity_secondary: value.F_arity_secondary,
      ro_consts_primary: value.ro_consts_primary,
      ro_consts_circuit_primary: value.ro_consts_circuit_primary,
      ck_primary,
      circuit_shape_primary: value.circuit_shape_primary,
      ro_consts_secondary: value.ro_consts_secondary,
      ro_consts_circuit_secondary: value.ro_consts_circuit_secondary,
      ck_secondary,
      circuit_shape_secondary: value.circuit_shape_secondary,
      augmented_circuit_params_primary: value.augmented_circuit_params_primary,
      augmented_circuit_params_secondary: value.augmented_circuit_params_secondary,
    })
  }
}

#[cfg(feature = "abomonate")]
impl<E1> From<FlatPublicParams<E1>> for PublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  fn from(value: FlatPublicParams<E1>) -> Self {
    Self {
      F_arity_primary: value.F_arity_primary,
      F_arity_secondary: value.F_arity_secondary,
      ro_consts_primary: value.ro_consts_primary,
      ro_consts_circuit_primary: value.ro_consts_circuit_primary,
      ck_primary: Arc::new(value.ck_primary),
      circuit_shape_primary: value.circuit_shape_primary,
      ro_consts_secondary: value.ro_consts_secondary,
      ro_consts_circuit_secondary: value.ro_consts_circuit_secondary,
      ck_secondary: Arc::new(value.ck_secondary),
      circuit_shape_secondary: value.circuit_shape_secondary,
      augmented_circuit_params_primary: value.augmented_circuit_params_primary,
      augmented_circuit_params_secondary: value.augmented_circuit_params_secondary,
      digest: OnceCell::new(),
    }
  }
}

impl<E1> SimpleDigestible for PublicParams<E1> where E1: CurveCycleEquipped {}

impl<E1> PublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  /// Set up builder to create `PublicParams` for a pair of circuits `C1` and `C2`.
  ///
  /// # Note
  ///
  /// Public parameters set up a number of bases for the homomorphic commitment scheme of Nova.
  ///
  /// Some final compressing SNARKs, like variants of Spartan, use computation commitments that require
  /// larger sizes for these parameters. These SNARKs provide a hint for these values by
  /// implementing `RelaxedR1CSSNARKTrait::ck_floor()`, which can be passed to this function.
  ///
  /// If you're not using such a SNARK, pass `arecibo::traits::snark::default_ck_hint()` instead.
  ///
  /// # Arguments
  ///
  /// * `c_primary`: The primary circuit of type `C1`.
  /// * `c_secondary`: The secondary circuit of type `C2`.
  /// * `ck_hint1`: A `CommitmentKeyHint` for `G1`, which is a function that provides a hint
  ///   for the number of generators required in the commitment scheme for the primary circuit.
  /// * `ck_hint2`: A `CommitmentKeyHint` for `G2`, similar to `ck_hint1`, but for the secondary circuit.
  ///
  /// # Example
  ///
  /// ```rust
  /// # use arecibo::spartan::ppsnark::RelaxedR1CSSNARK;
  /// # use arecibo::provider::pcs::ipa_pc::EvaluationEngine;
  /// # use arecibo::provider::{PallasEngine, VestaEngine};
  /// # use arecibo::traits::{circuit::TrivialCircuit, Engine, snark::RelaxedR1CSSNARKTrait};
  /// use arecibo::PublicParams;
  ///
  /// type E1 = PallasEngine;
  /// type E2 = VestaEngine;
  /// type EE<E> = EvaluationEngine<E>;
  /// type SPrime<E> = RelaxedR1CSSNARK<E, EE<E>>;
  ///
  /// let circuit1 = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
  /// let circuit2 = TrivialCircuit::<<E2 as Engine>::Scalar>::default();
  /// // Only relevant for a SNARK using computation commitmnets, pass &(|_| 0)
  /// // or &*nova_snark::traits::snark::default_ck_hint() otherwise.
  /// let ck_hint1 = &*SPrime::<E1>::ck_floor();
  /// let ck_hint2 = &*SPrime::<E2>::ck_floor();
  ///
  /// let pp = PublicParams::setup(&circuit1, &circuit2, ck_hint1, ck_hint2).unwrap();
  /// ```
  pub fn setup<C1: StepCircuit<E1::Scalar>, C2: StepCircuit<<Dual<E1> as Engine>::Scalar>>(
    c_primary: &C1,
    c_secondary: &C2,
    ck_hint1: &CommitmentKeyHint<E1>,
    ck_hint2: &CommitmentKeyHint<Dual<E1>>,
  ) -> Result<Self, NovaError> {
    let augmented_circuit_params_primary =
      NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let augmented_circuit_params_secondary =
      NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);

    let ro_consts_primary: ROConstants<E1> = ROConstants::<E1>::default();
    let ro_consts_secondary: ROConstants<Dual<E1>> = ROConstants::<Dual<E1>>::default();

    let F_arity_primary = c_primary.arity();
    let F_arity_secondary = c_secondary.arity();

    // ro_consts_circuit_primary are parameterized by E2 because the type alias uses E2::Base = E1::Scalar
    let ro_consts_circuit_primary: ROConstantsCircuit<Dual<E1>> =
      ROConstantsCircuit::<Dual<E1>>::default();
    let ro_consts_circuit_secondary: ROConstantsCircuit<E1> = ROConstantsCircuit::<E1>::default();

    // Initialize ck for the primary
    let circuit_primary: NovaAugmentedCircuit<'_, Dual<E1>, C1> = NovaAugmentedCircuit::new(
      &augmented_circuit_params_primary,
      None,
      c_primary,
      ro_consts_circuit_primary.clone(),
    );
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);
    let (r1cs_shape_primary, ck_primary) = cs.r1cs_shape_and_key(ck_hint1);
    let ck_primary = Arc::new(ck_primary);

    // Initialize ck for the secondary
    let circuit_secondary: NovaAugmentedCircuit<'_, E1, C2> = NovaAugmentedCircuit::new(
      &augmented_circuit_params_secondary,
      None,
      c_secondary,
      ro_consts_circuit_secondary.clone(),
    );
    let mut cs: ShapeCS<Dual<E1>> = ShapeCS::new();
    let _ = circuit_secondary.synthesize(&mut cs);
    let (r1cs_shape_secondary, ck_secondary) = cs.r1cs_shape_and_key(ck_hint2);
    let ck_secondary = Arc::new(ck_secondary);

    if r1cs_shape_primary.num_io != 2 || r1cs_shape_secondary.num_io != 2 {
      return Err(NovaError::InvalidStepCircuitIO);
    }

    let circuit_shape_primary = R1CSWithArity::new(r1cs_shape_primary, F_arity_primary);
    let circuit_shape_secondary = R1CSWithArity::new(r1cs_shape_secondary, F_arity_secondary);

    Ok(Self {
      F_arity_primary,
      F_arity_secondary,
      ro_consts_primary,
      ro_consts_circuit_primary,
      ck_primary,
      circuit_shape_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      ck_secondary,
      circuit_shape_secondary,
      augmented_circuit_params_primary,
      augmented_circuit_params_secondary,
      digest: OnceCell::new(),
    })
  }

  /// Retrieve the digest of the public parameters.
  pub fn digest(&self) -> E1::Scalar {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .expect("Failure in retrieving digest")
  }

  /// Returns the number of constraints in the primary and secondary circuits
  pub const fn num_constraints(&self) -> (usize, usize) {
    (
      self.circuit_shape_primary.r1cs_shape.num_cons,
      self.circuit_shape_secondary.r1cs_shape.num_cons,
    )
  }

  /// Returns the number of variables in the primary and secondary circuits
  pub const fn num_variables(&self) -> (usize, usize) {
    (
      self.circuit_shape_primary.r1cs_shape.num_vars,
      self.circuit_shape_secondary.r1cs_shape.num_vars,
    )
  }
}

/// A resource buffer for [`RecursiveSNARK`] for storing scratch values that are computed by `prove_step`,
/// which allows the reuse of memory allocations and avoids unnecessary new allocations in the critical section.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ResourceBuffer<E: Engine> {
  l_w: Option<R1CSWitness<E>>,
  l_u: Option<R1CSInstance<E>>,

  ABC_Z_1: R1CSResult<E>,
  ABC_Z_2: R1CSResult<E>,

  /// buffer for `commit_T`
  T: Vec<E::Scalar>,
}

/// A SNARK that proves the correct execution of an incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<E1>
where
  E1: CurveCycleEquipped,
{
  z0_primary: Vec<E1::Scalar>,
  z0_secondary: Vec<<Dual<E1> as Engine>::Scalar>,
  r_W_primary: RelaxedR1CSWitness<E1>,
  r_U_primary: RelaxedR1CSInstance<E1>,
  r_W_secondary: RelaxedR1CSWitness<Dual<E1>>,
  r_U_secondary: RelaxedR1CSInstance<Dual<E1>>,
  l_w_secondary: R1CSWitness<Dual<E1>>,
  l_u_secondary: R1CSInstance<Dual<E1>>,

  /// Buffer for memory needed by the primary fold-step
  buffer_primary: ResourceBuffer<E1>,
  /// Buffer for memory needed by the secondary fold-step
  buffer_secondary: ResourceBuffer<Dual<E1>>,

  i: usize,
  zi_primary: Vec<E1::Scalar>,
  zi_secondary: Vec<<Dual<E1> as Engine>::Scalar>,
}

impl<E1> RecursiveSNARK<E1>
where
  E1: CurveCycleEquipped,
{
  /// Create new instance of recursive SNARK
  pub fn new<C1: StepCircuit<E1::Scalar>, C2: StepCircuit<<Dual<E1> as Engine>::Scalar>>(
    pp: &PublicParams<E1>,
    c_primary: &C1,
    c_secondary: &C2,
    z0_primary: &[E1::Scalar],
    z0_secondary: &[<Dual<E1> as Engine>::Scalar],
  ) -> Result<Self, NovaError> {
    if z0_primary.len() != pp.F_arity_primary || z0_secondary.len() != pp.F_arity_secondary {
      return Err(NovaError::InvalidInitialInputLength);
    }

    let r1cs_primary = &pp.circuit_shape_primary.r1cs_shape;
    let r1cs_secondary = &pp.circuit_shape_secondary.r1cs_shape;

    // base case for the primary
    let mut cs_primary = SatisfyingAssignment::<E1>::new();
    let inputs_primary: NovaAugmentedCircuitInputs<Dual<E1>> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      E1::Scalar::ZERO,
      z0_primary.to_vec(),
      None,
      None,
      None,
      None,
    );

    let circuit_primary: NovaAugmentedCircuit<'_, Dual<E1>, C1> = NovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_primary,
      Some(inputs_primary),
      c_primary,
      pp.ro_consts_circuit_primary.clone(),
    );
    let zi_primary = circuit_primary.synthesize(&mut cs_primary)?;
    let (u_primary, w_primary) =
      cs_primary.r1cs_instance_and_witness(r1cs_primary, &pp.ck_primary)?;

    // base case for the secondary
    let mut cs_secondary = SatisfyingAssignment::<Dual<E1>>::new();
    let inputs_secondary: NovaAugmentedCircuitInputs<E1> = NovaAugmentedCircuitInputs::new(
      pp.digest(),
      <Dual<E1> as Engine>::Scalar::ZERO,
      z0_secondary.to_vec(),
      None,
      None,
      Some(u_primary.clone()),
      None,
    );
    let circuit_secondary: NovaAugmentedCircuit<'_, E1, C2> = NovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_secondary,
      Some(inputs_secondary),
      c_secondary,
      pp.ro_consts_circuit_secondary.clone(),
    );
    let zi_secondary = circuit_secondary.synthesize(&mut cs_secondary)?;
    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.circuit_shape_secondary.r1cs_shape, &pp.ck_secondary)?;

    // IVC proof for the primary circuit
    let l_w_primary = w_primary;
    let l_u_primary = u_primary;
    let r_W_primary = RelaxedR1CSWitness::from_r1cs_witness(r1cs_primary, l_w_primary);
    let r_U_primary = RelaxedR1CSInstance::from_r1cs_instance(
      &*pp.ck_primary,
      &pp.circuit_shape_primary.r1cs_shape,
      l_u_primary,
    );

    // IVC proof for the secondary circuit
    let l_w_secondary = w_secondary;
    let l_u_secondary = u_secondary;
    let r_W_secondary = RelaxedR1CSWitness::<Dual<E1>>::default(r1cs_secondary);
    let r_U_secondary = RelaxedR1CSInstance::<Dual<E1>>::default(&pp.ck_secondary, r1cs_secondary);

    assert!(
      !(zi_primary.len() != pp.F_arity_primary || zi_secondary.len() != pp.F_arity_secondary),
      "Invalid step length"
    );

    let zi_primary = zi_primary
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;

    let zi_secondary = zi_secondary
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<Dual<E1> as Engine>::Scalar>, _>>()?;

    let buffer_primary = ResourceBuffer {
      l_w: None,
      l_u: None,
      ABC_Z_1: R1CSResult::default(r1cs_primary.num_cons),
      ABC_Z_2: R1CSResult::default(r1cs_primary.num_cons),
      T: r1cs::default_T::<E1>(r1cs_primary.num_cons),
    };

    let buffer_secondary = ResourceBuffer {
      l_w: None,
      l_u: None,
      ABC_Z_1: R1CSResult::default(r1cs_secondary.num_cons),
      ABC_Z_2: R1CSResult::default(r1cs_secondary.num_cons),
      T: r1cs::default_T::<Dual<E1>>(r1cs_secondary.num_cons),
    };

    Ok(Self {
      z0_primary: z0_primary.to_vec(),
      z0_secondary: z0_secondary.to_vec(),
      r_W_primary,
      r_U_primary,
      r_W_secondary,
      r_U_secondary,
      l_w_secondary,
      l_u_secondary,

      buffer_primary,
      buffer_secondary,
      i: 0,
      zi_primary,
      zi_secondary,
    })
  }

  /// Inputs of the primary circuits
  pub fn z0_primary(&self) -> &Vec<E1::Scalar> {
    &self.z0_primary
  }

  /// Outputs of the primary circuits
  pub fn zi_primary(&self) -> &Vec<E1::Scalar> {
    &self.zi_primary
  }

  /// Create a new `RecursiveSNARK` (or updates the provided `RecursiveSNARK`)
  /// by executing a step of the incremental computation
  #[tracing::instrument(skip_all, name = "nova::RecursiveSNARK::prove_step")]
  pub fn prove_step<C1: StepCircuit<E1::Scalar>, C2: StepCircuit<<Dual<E1> as Engine>::Scalar>>(
    &mut self,
    pp: &PublicParams<E1>,
    c_primary: &C1,
    c_secondary: &C2,
  ) -> Result<(), NovaError> {
    // first step was already done in the constructor
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    // save the inputs before proceeding to the `i+1`th step
    let r_U_primary_i = self.r_U_primary.clone();
    let r_U_secondary_i = self.r_U_secondary.clone();
    let l_u_secondary_i = self.l_u_secondary.clone();

    // fold the secondary circuit's instance
    let (nifs_secondary, _) = NIFS::prove_mut(
      &*pp.ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<E1>(pp.digest()),
      &pp.circuit_shape_secondary.r1cs_shape,
      &mut self.r_U_secondary,
      &mut self.r_W_secondary,
      &self.l_u_secondary,
      &self.l_w_secondary,
      &mut self.buffer_secondary.T,
      &mut self.buffer_secondary.ABC_Z_1,
      &mut self.buffer_secondary.ABC_Z_2,
    )?;

    let mut cs_primary = SatisfyingAssignment::<E1>::with_capacity(
      pp.circuit_shape_primary.r1cs_shape.num_io + 1,
      pp.circuit_shape_primary.r1cs_shape.num_vars,
    );
    let inputs_primary: NovaAugmentedCircuitInputs<Dual<E1>> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      E1::Scalar::from(self.i as u64),
      self.z0_primary.to_vec(),
      Some(self.zi_primary.clone()),
      Some(r_U_secondary_i),
      Some(l_u_secondary_i),
      Some(Commitment::<Dual<E1>>::decompress(&nifs_secondary.comm_T)?),
    );

    let circuit_primary: NovaAugmentedCircuit<'_, Dual<E1>, C1> = NovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_primary,
      Some(inputs_primary),
      c_primary,
      pp.ro_consts_circuit_primary.clone(),
    );

    let zi_primary = circuit_primary.synthesize(&mut cs_primary)?;

    let (l_u_primary, l_w_primary) =
      cs_primary.r1cs_instance_and_witness(&pp.circuit_shape_primary.r1cs_shape, &pp.ck_primary)?;

    // fold the primary circuit's instance
    let (nifs_primary, _) = NIFS::prove_mut(
      &*pp.ck_primary,
      &pp.ro_consts_primary,
      &pp.digest(),
      &pp.circuit_shape_primary.r1cs_shape,
      &mut self.r_U_primary,
      &mut self.r_W_primary,
      &l_u_primary,
      &l_w_primary,
      &mut self.buffer_primary.T,
      &mut self.buffer_primary.ABC_Z_1,
      &mut self.buffer_primary.ABC_Z_2,
    )?;

    let mut cs_secondary = SatisfyingAssignment::<Dual<E1>>::with_capacity(
      pp.circuit_shape_secondary.r1cs_shape.num_io + 1,
      pp.circuit_shape_secondary.r1cs_shape.num_vars,
    );
    let inputs_secondary: NovaAugmentedCircuitInputs<E1> = NovaAugmentedCircuitInputs::new(
      pp.digest(),
      <Dual<E1> as Engine>::Scalar::from(self.i as u64),
      self.z0_secondary.to_vec(),
      Some(self.zi_secondary.clone()),
      Some(r_U_primary_i),
      Some(l_u_primary),
      Some(Commitment::<E1>::decompress(&nifs_primary.comm_T)?),
    );

    let circuit_secondary: NovaAugmentedCircuit<'_, E1, C2> = NovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_secondary,
      Some(inputs_secondary),
      c_secondary,
      pp.ro_consts_circuit_secondary.clone(),
    );
    let zi_secondary = circuit_secondary.synthesize(&mut cs_secondary)?;

    let (l_u_secondary, l_w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.circuit_shape_secondary.r1cs_shape, &pp.ck_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    // update the running instances and witnesses
    self.zi_primary = zi_primary
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;
    self.zi_secondary = zi_secondary
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<Dual<E1> as Engine>::Scalar>, _>>()?;

    self.l_u_secondary = l_u_secondary;
    self.l_w_secondary = l_w_secondary;

    self.i += 1;

    Ok(())
  }

  /// Verify the correctness of the `RecursiveSNARK`
  pub fn verify(
    &self,
    pp: &PublicParams<E1>,
    num_steps: usize,
    z0_primary: &[E1::Scalar],
    z0_secondary: &[<Dual<E1> as Engine>::Scalar],
  ) -> Result<(Vec<E1::Scalar>, Vec<<Dual<E1> as Engine>::Scalar>), NovaError> {
    // number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;

    // check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;

    // check if the initial inputs match
    let is_inputs_not_match = self.z0_primary != z0_primary || self.z0_secondary != z0_secondary;

    // check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outputs = self.l_u_secondary.X.len() != 2
      || self.r_U_primary.X.len() != 2
      || self.r_U_secondary.X.len() != 2;

    if is_num_steps_zero
      || is_num_steps_not_match
      || is_inputs_not_match
      || is_instance_has_two_outputs
    {
      return Err(NovaError::ProofVerifyError);
    }

    // check if the output hashes in R1CS instances point to the right running instances
    let (hash_primary, hash_secondary) = {
      let mut hasher = <Dual<E1> as Engine>::RO::new(
        pp.ro_consts_secondary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * pp.F_arity_primary,
      );
      hasher.absorb(pp.digest());
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0_primary {
        hasher.absorb(*e);
      }
      for e in &self.zi_primary {
        hasher.absorb(*e);
      }
      self.r_U_secondary.absorb_in_ro(&mut hasher);

      let mut hasher2 = <E1 as Engine>::RO::new(
        pp.ro_consts_primary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * pp.F_arity_secondary,
      );
      hasher2.absorb(scalar_as_base::<E1>(pp.digest()));
      hasher2.absorb(<Dual<E1> as Engine>::Scalar::from(num_steps as u64));
      for e in z0_secondary {
        hasher2.absorb(*e);
      }
      for e in &self.zi_secondary {
        hasher2.absorb(*e);
      }
      self.r_U_primary.absorb_in_ro(&mut hasher2);

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != self.l_u_secondary.X[0]
      || hash_secondary != scalar_as_base::<Dual<E1>>(self.l_u_secondary.X[1])
    {
      return Err(NovaError::ProofVerifyError);
    }

    // check the satisfiability of the provided instances
    let (res_r_primary, (res_r_secondary, res_l_secondary)) = rayon::join(
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
            pp.circuit_shape_secondary.r1cs_shape.is_sat_relaxed(
              &pp.ck_secondary,
              &self.r_U_secondary,
              &self.r_W_secondary,
            )
          },
          || {
            pp.circuit_shape_secondary.r1cs_shape.is_sat(
              &pp.ck_secondary,
              &self.l_u_secondary,
              &self.l_w_secondary,
            )
          },
        )
      },
    );

    // check the returned res objects
    res_r_primary?;
    res_r_secondary?;
    res_l_secondary?;

    Ok((self.zi_primary.clone(), self.zi_secondary.clone()))
  }

  /// Get the outputs after the last step of computation.
  pub fn outputs(&self) -> (&[E1::Scalar], &[<Dual<E1> as Engine>::Scalar]) {
    (&self.zi_primary, &self.zi_secondary)
  }

  /// The number of steps which have been executed thus far.
  pub fn num_steps(&self) -> usize {
    self.i
  }
}

/// A type that holds the prover key for `CompressedSNARK`
#[derive(Clone, Debug)]
pub struct ProverKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  pk_primary: S1::ProverKey,
  pk_secondary: S2::ProverKey,
}

/// A type that holds the verifier key for `CompressedSNARK`
#[derive(Debug, Clone, Serialize)]
#[serde(bound = "")]
pub struct VerifierKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  F_arity_primary: usize,
  F_arity_secondary: usize,
  ro_consts_primary: ROConstants<E1>,
  ro_consts_secondary: ROConstants<Dual<E1>>,
  pp_digest: E1::Scalar,
  vk_primary: S1::VerifierKey,
  vk_secondary: S2::VerifierKey,
}

/// A SNARK that proves the knowledge of a valid `RecursiveSNARK`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSNARK<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  r_U_primary: RelaxedR1CSInstance<E1>,
  r_W_snark_primary: S1,

  r_U_secondary: RelaxedR1CSInstance<Dual<E1>>,
  l_u_secondary: R1CSInstance<Dual<E1>>,
  nifs_secondary: NIFS<Dual<E1>>,
  f_W_snark_secondary: S2,

  zn_primary: Vec<E1::Scalar>,
  zn_secondary: Vec<<Dual<E1> as Engine>::Scalar>,
}

impl<E1, S1, S2> CompressedSNARK<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  /// Creates prover and verifier keys for `CompressedSNARK`
  pub fn setup(
    pp: &PublicParams<E1>,
  ) -> Result<(ProverKey<E1, S1, S2>, VerifierKey<E1, S1, S2>), NovaError> {
    let (pk_primary, vk_primary) =
      S1::setup(pp.ck_primary.clone(), &pp.circuit_shape_primary.r1cs_shape)?;
    let (pk_secondary, vk_secondary) = S2::setup(
      pp.ck_secondary.clone(),
      &pp.circuit_shape_secondary.r1cs_shape,
    )?;

    let pk = ProverKey {
      pk_primary,
      pk_secondary,
    };

    let vk = VerifierKey {
      F_arity_primary: pp.F_arity_primary,
      F_arity_secondary: pp.F_arity_secondary,
      ro_consts_primary: pp.ro_consts_primary.clone(),
      ro_consts_secondary: pp.ro_consts_secondary.clone(),
      pp_digest: pp.digest(),
      vk_primary,
      vk_secondary,
    };

    Ok((pk, vk))
  }

  /// Create a new `CompressedSNARK`
  pub fn prove(
    pp: &PublicParams<E1>,
    pk: &ProverKey<E1, S1, S2>,
    recursive_snark: &RecursiveSNARK<E1>,
  ) -> Result<Self, NovaError> {
    // fold the secondary circuit's instance with its running instance
    let (nifs_secondary, (f_U_secondary, f_W_secondary), _) = NIFS::prove(
      &*pp.ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<E1>(pp.digest()),
      &pp.circuit_shape_secondary.r1cs_shape,
      &recursive_snark.r_U_secondary,
      &recursive_snark.r_W_secondary,
      &recursive_snark.l_u_secondary,
      &recursive_snark.l_w_secondary,
    )?;

    // create SNARKs proving the knowledge of f_W_primary and f_W_secondary
    let (r_W_snark_primary, f_W_snark_secondary) = rayon::join(
      || {
        S1::prove(
          &pp.ck_primary,
          &pk.pk_primary,
          &pp.circuit_shape_primary.r1cs_shape,
          &recursive_snark.r_U_primary,
          &recursive_snark.r_W_primary,
        )
      },
      || {
        S2::prove(
          &pp.ck_secondary,
          &pk.pk_secondary,
          &pp.circuit_shape_secondary.r1cs_shape,
          &f_U_secondary,
          &f_W_secondary,
        )
      },
    );

    Ok(Self {
      r_U_primary: recursive_snark.r_U_primary.clone(),
      r_W_snark_primary: r_W_snark_primary?,

      r_U_secondary: recursive_snark.r_U_secondary.clone(),
      l_u_secondary: recursive_snark.l_u_secondary.clone(),
      nifs_secondary,
      f_W_snark_secondary: f_W_snark_secondary?,

      zn_primary: recursive_snark.zi_primary.clone(),
      zn_secondary: recursive_snark.zi_secondary.clone(),
    })
  }

  /// Verify the correctness of the `CompressedSNARK`
  pub fn verify(
    &self,
    vk: &VerifierKey<E1, S1, S2>,
    num_steps: usize,
    z0_primary: &[E1::Scalar],
    z0_secondary: &[<Dual<E1> as Engine>::Scalar],
  ) -> Result<(Vec<E1::Scalar>, Vec<<Dual<E1> as Engine>::Scalar>), NovaError> {
    // the number of steps cannot be zero
    if num_steps == 0 {
      return Err(NovaError::ProofVerifyError);
    }

    // check if the (relaxed) R1CS instances have two public outputs
    if self.l_u_secondary.X.len() != 2
      || self.r_U_primary.X.len() != 2
      || self.r_U_secondary.X.len() != 2
    {
      return Err(NovaError::ProofVerifyError);
    }

    // check if the output hashes in R1CS instances point to the right running instances
    let (hash_primary, hash_secondary) = {
      let mut hasher = <Dual<E1> as Engine>::RO::new(
        vk.ro_consts_secondary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * vk.F_arity_primary,
      );
      hasher.absorb(vk.pp_digest);
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0_primary {
        hasher.absorb(*e);
      }
      for e in &self.zn_primary {
        hasher.absorb(*e);
      }
      self.r_U_secondary.absorb_in_ro(&mut hasher);

      let mut hasher2 = <E1 as Engine>::RO::new(
        vk.ro_consts_primary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * vk.F_arity_secondary,
      );
      hasher2.absorb(scalar_as_base::<E1>(vk.pp_digest));
      hasher2.absorb(<Dual<E1> as Engine>::Scalar::from(num_steps as u64));
      for e in z0_secondary {
        hasher2.absorb(*e);
      }
      for e in &self.zn_secondary {
        hasher2.absorb(*e);
      }
      self.r_U_primary.absorb_in_ro(&mut hasher2);

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != self.l_u_secondary.X[0]
      || hash_secondary != scalar_as_base::<Dual<E1>>(self.l_u_secondary.X[1])
    {
      return Err(NovaError::ProofVerifyError);
    }

    // fold the secondary's running instance with the last instance to get a folded instance
    let f_U_secondary = self.nifs_secondary.verify(
      &vk.ro_consts_secondary,
      &scalar_as_base::<E1>(vk.pp_digest),
      &self.r_U_secondary,
      &self.l_u_secondary,
    )?;

    // check the satisfiability of the folded instances using
    // SNARKs proving the knowledge of their satisfying witnesses
    let (res_primary, res_secondary) = rayon::join(
      || {
        self
          .r_W_snark_primary
          .verify(&vk.vk_primary, &self.r_U_primary)
      },
      || {
        self
          .f_W_snark_secondary
          .verify(&vk.vk_secondary, &f_U_secondary)
      },
    );

    res_primary?;
    res_secondary?;

    Ok((self.zn_primary.clone(), self.zn_secondary.clone()))
  }
}

/// Compute the circuit digest of a [`StepCircuit`].
///
/// Note for callers: This function should be called with its performance characteristics in mind.
/// It will synthesize and digest the full `circuit` given.
pub fn circuit_digest<E1: CurveCycleEquipped, C: StepCircuit<E1::Scalar>>(
  circuit: &C,
) -> E1::Scalar {
  let augmented_circuit_params = NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);

  // ro_consts_circuit are parameterized by G2 because the type alias uses G2::Base = G1::Scalar
  let ro_consts_circuit: ROConstantsCircuit<Dual<E1>> = ROConstantsCircuit::<Dual<E1>>::default();

  // Initialize ck for the primary
  let augmented_circuit: NovaAugmentedCircuit<'_, Dual<E1>, C> =
    NovaAugmentedCircuit::new(&augmented_circuit_params, None, circuit, ro_consts_circuit);
  let mut cs: ShapeCS<E1> = ShapeCS::new();
  let _ = augmented_circuit.synthesize(&mut cs);
  cs.r1cs_shape().digest()
}

type CommitmentKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey;
type Commitment<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment;
type CompressedCommitment<E> = <<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as CommitmentTrait<E>>::CompressedCommitment;
type CE<E> = <E as Engine>::CE;

#[cfg(test)]
mod tests {
  use self::traits::CurveCycleEquipped;

  use super::*;
  use crate::{
    provider::{
      pcs, pcs::non_hiding_zeromorph::ZMPCS, Bn256EngineIPA, Bn256EngineKZG, Bn256EngineZM,
      PallasEngine, Secp256k1Engine,
    },
    traits::{evaluation::EvaluationEngineTrait, snark::default_ck_hint},
  };
  use ::bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
  use core::{fmt::Write, marker::PhantomData};
  use expect_test::{expect, Expect};
  use ff::PrimeField;
  use halo2curves::bn256::Bn256;
  use traits::circuit::TrivialCircuit;

  type EE<E> = pcs::ipa_pc::EvaluationEngine<E>;
  type S<E, EE> = spartan::snark::RelaxedR1CSSNARK<E, EE>;
  type SPrime<E, EE> = spartan::ppsnark::RelaxedR1CSSNARK<E, EE>;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F> {
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> StepCircuit<F> for CubicCircuit<F> {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = &z[0];
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
      })?;

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

      Ok(vec![y])
    }
  }

  impl<F: PrimeField> CubicCircuit<F> {
    fn output(&self, z: &[F]) -> Vec<F> {
      vec![z[0] * z[0] * z[0] + z[0] + F::from(5u64)]
    }
  }

  fn test_pp_digest_with<E1, T1, T2, EE1, EE2>(circuit1: &T1, circuit2: &T2, expected: &Expect)
  where
    E1: CurveCycleEquipped,
    T1: StepCircuit<E1::Scalar>,
    T2: StepCircuit<<Dual<E1> as Engine>::Scalar>,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<Dual<E1>>,
    // this is due to the reliance on Abomonation
    <E1::Scalar as PrimeField>::Repr: Abomonation,
    <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
  {
    // this tests public parameters with a size specifically intended for a spark-compressed SNARK
    let ck_hint1 = &*SPrime::<E1, EE1>::ck_floor();
    let ck_hint2 = &*SPrime::<Dual<E1>, EE2>::ck_floor();
    let pp = PublicParams::<E1>::setup(circuit1, circuit2, ck_hint1, ck_hint2).unwrap();

    let digest_str = pp
      .digest()
      .to_repr()
      .as_ref()
      .iter()
      .fold(String::new(), |mut output, b| {
        let _ = write!(output, "{b:02x}");
        output
      });

    expected.assert_eq(&digest_str);
  }

  #[test]
  fn test_pp_digest() {
    test_pp_digest_with::<PallasEngine, _, _, EE<_>, EE<_>>(
      &TrivialCircuit::default(),
      &TrivialCircuit::default(),
      &expect!["e5a6a85b77f3fb958b69722a5a21bf656fd21a6b5a012708a4b086b6be6d2b03"],
    );

    test_pp_digest_with::<PallasEngine, _, _, EE<_>, EE<_>>(
      &CubicCircuit::default(),
      &TrivialCircuit::default(),
      &expect!["ec707a8b822baebca114b6e61b238374f9ed358c542dd37ee73febb47832cd01"],
    );

    test_pp_digest_with::<Bn256EngineIPA, _, _, EE<_>, EE<_>>(
      &TrivialCircuit::default(),
      &TrivialCircuit::default(),
      &expect!["df52de22456157eb056003d4dc580a167ab8ce40a151c9944ea09a6fd0028600"],
    );

    test_pp_digest_with::<Bn256EngineIPA, _, _, EE<_>, EE<_>>(
      &CubicCircuit::default(),
      &TrivialCircuit::default(),
      &expect!["b3ad0f4b734c5bd2ab9e83be8ee0cbaaa120e5cd0270b51cb9d7778a33f0b801"],
    );

    test_pp_digest_with::<Secp256k1Engine, _, _, EE<_>, EE<_>>(
      &TrivialCircuit::default(),
      &TrivialCircuit::default(),
      &expect!["e1feca53664212ee750da857c726b2a09bb30b2964f22ea85a19b58c9eaf5701"],
    );
    test_pp_digest_with::<Secp256k1Engine, _, _, EE<_>, EE<_>>(
      &CubicCircuit::default(),
      &TrivialCircuit::default(),
      &expect!["4ad6b10b6fd24fecba49f08d35bc874a6da9c77735bc0bcf4b78b1914a97e602"],
    );
  }

  fn test_ivc_trivial_with<E1>()
  where
    E1: CurveCycleEquipped,
  {
    let test_circuit1 = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
    let test_circuit2 = TrivialCircuit::<<Dual<E1> as Engine>::Scalar>::default();

    // produce public parameters
    let pp = PublicParams::<E1>::setup(
      &test_circuit1,
      &test_circuit2,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();
    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::new(
      &pp,
      &test_circuit1,
      &test_circuit2,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )
    .unwrap();

    recursive_snark
      .prove_step(&pp, &test_circuit1, &test_circuit2)
      .unwrap();

    // verify the recursive SNARK
    recursive_snark
      .verify(
        &pp,
        num_steps,
        &[<E1 as Engine>::Scalar::ZERO],
        &[<Dual<E1> as Engine>::Scalar::ZERO],
      )
      .unwrap();
  }

  #[test]
  fn test_ivc_trivial() {
    test_ivc_trivial_with::<PallasEngine>();
    test_ivc_trivial_with::<Bn256EngineIPA>();
    test_ivc_trivial_with::<Secp256k1Engine>();
  }

  fn test_ivc_nontrivial_with<E1>()
  where
    E1: CurveCycleEquipped,
  {
    let circuit_primary = TrivialCircuit::default();
    let circuit_secondary = CubicCircuit::default();

    // produce public parameters
    let pp = PublicParams::<E1>::setup(
      &circuit_primary,
      &circuit_secondary,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1>::new(
      &pp,
      &circuit_primary,
      &circuit_secondary,
      &[<E1 as Engine>::Scalar::ONE],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )
    .unwrap();

    for i in 0..num_steps {
      recursive_snark
        .prove_step(&pp, &circuit_primary, &circuit_secondary)
        .unwrap();

      // verify the recursive snark at each step of recursion
      recursive_snark
        .verify(
          &pp,
          i + 1,
          &[<E1 as Engine>::Scalar::ONE],
          &[<Dual<E1> as Engine>::Scalar::ZERO],
        )
        .unwrap();
    }

    // verify the recursive SNARK
    let (zn_primary, zn_secondary) = recursive_snark
      .verify(
        &pp,
        num_steps,
        &[<E1 as Engine>::Scalar::ONE],
        &[<Dual<E1> as Engine>::Scalar::ZERO],
      )
      .unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
    let mut zn_secondary_direct = vec![<Dual<E1> as Engine>::Scalar::ZERO];
    for _i in 0..num_steps {
      zn_secondary_direct = circuit_secondary.clone().output(&zn_secondary_direct);
    }
    assert_eq!(zn_secondary, zn_secondary_direct);
    assert_eq!(
      zn_secondary,
      vec![<Dual<E1> as Engine>::Scalar::from(2460515u64)]
    );
  }

  #[test]
  fn test_ivc_nontrivial() {
    test_ivc_nontrivial_with::<PallasEngine>();
    test_ivc_nontrivial_with::<Bn256EngineKZG>();
    test_ivc_nontrivial_with::<Secp256k1Engine>();
  }

  fn test_ivc_nontrivial_with_some_compression_with<E1, S1, S2>()
  where
    E1: CurveCycleEquipped,
    // this is due to the reliance on Abomonation
    <E1::Scalar as PrimeField>::Repr: Abomonation,
    <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
    S1: RelaxedR1CSSNARKTrait<E1>,
    S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
  {
    let circuit_primary = TrivialCircuit::default();
    let circuit_secondary = CubicCircuit::default();

    // produce public parameters, which we'll maybe use with a preprocessing compressed SNARK
    let pp = PublicParams::<E1>::setup(
      &circuit_primary,
      &circuit_secondary,
      &*S1::ck_floor(),
      &*S2::ck_floor(),
    )
    .unwrap();

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1>::new(
      &pp,
      &circuit_primary,
      &circuit_secondary,
      &[<E1 as Engine>::Scalar::ONE],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )
    .unwrap();

    for _i in 0..num_steps {
      recursive_snark
        .prove_step(&pp, &circuit_primary, &circuit_secondary)
        .unwrap();
    }

    // verify the recursive SNARK
    let (zn_primary, zn_secondary) = recursive_snark
      .verify(
        &pp,
        num_steps,
        &[<E1 as Engine>::Scalar::ONE],
        &[<Dual<E1> as Engine>::Scalar::ZERO],
      )
      .unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
    let mut zn_secondary_direct = vec![<Dual<E1> as Engine>::Scalar::ZERO];
    for _i in 0..num_steps {
      zn_secondary_direct = circuit_secondary.clone().output(&zn_secondary_direct);
    }
    assert_eq!(zn_secondary, zn_secondary_direct);
    assert_eq!(
      zn_secondary,
      vec![<Dual<E1> as Engine>::Scalar::from(2460515u64)]
    );

    // run the compressed snark
    // produce the prover and verifier keys for compressed snark
    let (pk, vk) = CompressedSNARK::<_, S1, S2>::setup(&pp).unwrap();

    // produce a compressed SNARK
    let compressed_snark = CompressedSNARK::<_, S1, S2>::prove(&pp, &pk, &recursive_snark).unwrap();

    // verify the compressed SNARK
    compressed_snark
      .verify(
        &vk,
        num_steps,
        &[<E1 as Engine>::Scalar::ONE],
        &[<Dual<E1> as Engine>::Scalar::ZERO],
      )
      .unwrap();
  }

  fn test_ivc_nontrivial_with_compression_with<E1, EE1, EE2>()
  where
    E1: CurveCycleEquipped,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<Dual<E1>>,
    // this is due to the reliance on Abomonation
    <E1::Scalar as PrimeField>::Repr: Abomonation,
    <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
  {
    test_ivc_nontrivial_with_some_compression_with::<E1, S<_, EE1>, S<_, EE2>>()
  }

  #[test]
  fn test_ivc_nontrivial_with_compression() {
    test_ivc_nontrivial_with_compression_with::<PallasEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_compression_with::<Bn256EngineIPA, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_compression_with::<Secp256k1Engine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_compression_with::<Bn256EngineZM, ZMPCS<Bn256, _>, EE<_>>();
    test_ivc_nontrivial_with_compression_with::<
      Bn256EngineKZG,
      pcs::hyperkzg::EvaluationEngine<Bn256, _>,
      EE<_>,
    >();
  }

  fn test_ivc_nontrivial_with_spark_compression_with<E1, EE1, EE2>()
  where
    E1: CurveCycleEquipped,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<Dual<E1>>,
    // this is due to the reliance on Abomonation
    <E1::Scalar as PrimeField>::Repr: Abomonation,
    <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
  {
    test_ivc_nontrivial_with_some_compression_with::<E1, SPrime<_, EE1>, SPrime<_, EE2>>()
  }

  #[test]
  fn test_ivc_nontrivial_with_spark_compression() {
    test_ivc_nontrivial_with_spark_compression_with::<PallasEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_spark_compression_with::<Bn256EngineIPA, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_spark_compression_with::<Secp256k1Engine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_spark_compression_with::<Bn256EngineZM, ZMPCS<Bn256, _>, EE<_>>();
    test_ivc_nontrivial_with_spark_compression_with::<
      Bn256EngineKZG,
      pcs::hyperkzg::EvaluationEngine<Bn256, _>,
      EE<_>,
    >();
  }

  type BatchedS<E, EE> = spartan::batched::BatchedRelaxedR1CSSNARK<E, EE>;
  type BatchedSPrime<E, EE> = spartan::batched::BatchedRelaxedR1CSSNARK<E, EE>;

  fn test_ivc_nontrivial_with_batched_compression_with<E1, EE1, EE2>()
  where
    E1: CurveCycleEquipped,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<Dual<E1>>,
    // this is due to the reliance on Abomonation
    <E1::Scalar as PrimeField>::Repr: Abomonation,
    <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
  {
    // this tests compatibility of the batched workflow with the non-batched one
    test_ivc_nontrivial_with_some_compression_with::<E1, BatchedS<_, EE1>, BatchedS<_, EE2>>()
  }

  #[test]
  fn test_ivc_nontrivial_with_batched_compression() {
    test_ivc_nontrivial_with_batched_compression_with::<PallasEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_batched_compression_with::<Bn256EngineIPA, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_batched_compression_with::<Secp256k1Engine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_batched_compression_with::<Bn256EngineZM, ZMPCS<Bn256, _>, EE<_>>();
    test_ivc_nontrivial_with_batched_compression_with::<
      Bn256EngineKZG,
      pcs::hyperkzg::EvaluationEngine<Bn256, _>,
      EE<_>,
    >();
  }

  fn test_ivc_nontrivial_with_batched_spark_compression_with<E1, EE1, EE2>()
  where
    E1: CurveCycleEquipped,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<Dual<E1>>,
    // this is due to the reliance on Abomonation
    <E1::Scalar as PrimeField>::Repr: Abomonation,
    <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
  {
    // this tests compatibility of the batched workflow with the non-batched one
    test_ivc_nontrivial_with_some_compression_with::<E1, BatchedSPrime<_, EE1>, BatchedSPrime<_, EE2>>(
    )
  }

  #[test]
  fn test_ivc_nontrivial_with_batched_spark_compression() {
    test_ivc_nontrivial_with_batched_spark_compression_with::<PallasEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_batched_spark_compression_with::<Bn256EngineIPA, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_batched_spark_compression_with::<Secp256k1Engine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_batched_spark_compression_with::<Bn256EngineZM, ZMPCS<Bn256, _>, EE<_>>(
    );
    test_ivc_nontrivial_with_batched_spark_compression_with::<
      Bn256EngineKZG,
      pcs::hyperkzg::EvaluationEngine<Bn256, _>,
      EE<_>,
    >();
  }

  fn test_ivc_nondet_with_compression_with<E1, EE1, EE2>()
  where
    E1: CurveCycleEquipped,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<Dual<E1>>,
    // this is due to the reliance on Abomonation
    <E1::Scalar as PrimeField>::Repr: Abomonation,
    <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
  {
    // y is a non-deterministic advice representing the fifth root of the input at a step.
    #[derive(Clone, Debug)]
    struct FifthRootCheckingCircuit<F> {
      y: F,
    }

    impl<F: PrimeField> FifthRootCheckingCircuit<F> {
      fn new(num_steps: usize) -> (Vec<F>, Vec<Self>) {
        let mut powers = Vec::new();
        let rng = &mut rand::rngs::OsRng;
        let mut seed = F::random(rng);
        for _i in 0..num_steps + 1 {
          seed *= seed.clone().square().square();

          powers.push(Self { y: seed });
        }

        // reverse the powers to get roots
        let roots = powers.into_iter().rev().collect::<Vec<Self>>();
        (vec![roots[0].y], roots[1..].to_vec())
      }
    }

    impl<F> StepCircuit<F> for FifthRootCheckingCircuit<F>
    where
      F: PrimeField,
    {
      fn arity(&self) -> usize {
        1
      }

      fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
      ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        let x = &z[0];

        // we allocate a variable and set it to the provided non-deterministic advice.
        let y = AllocatedNum::alloc_infallible(cs.namespace(|| "y"), || self.y);

        // We now check if y = x^{1/5} by checking if y^5 = x
        let y_sq = y.square(cs.namespace(|| "y_sq"))?;
        let y_quad = y_sq.square(cs.namespace(|| "y_quad"))?;
        let y_pow_5 = y_quad.mul(cs.namespace(|| "y_fifth"), &y)?;

        cs.enforce(
          || "y^5 = x",
          |lc| lc + y_pow_5.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + x.get_variable(),
        );

        Ok(vec![y])
      }
    }

    let circuit_primary = FifthRootCheckingCircuit {
      y: <E1 as Engine>::Scalar::ZERO,
    };

    let circuit_secondary = TrivialCircuit::default();

    // produce public parameters
    let pp = PublicParams::<E1>::setup(
      &circuit_primary,
      &circuit_secondary,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 3;

    // produce non-deterministic advice
    let (z0_primary, roots) = FifthRootCheckingCircuit::new(num_steps);
    let z0_secondary = vec![<Dual<E1> as Engine>::Scalar::ZERO];

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1>::new(
      &pp,
      &roots[0],
      &circuit_secondary,
      &z0_primary,
      &z0_secondary,
    )
    .unwrap();

    for circuit_primary in roots.iter().take(num_steps) {
      recursive_snark
        .prove_step(&pp, circuit_primary, &circuit_secondary)
        .unwrap();
    }

    // verify the recursive SNARK
    recursive_snark
      .verify(&pp, num_steps, &z0_primary, &z0_secondary)
      .unwrap();

    // produce the prover and verifier keys for compressed snark
    let (pk, vk) = CompressedSNARK::<_, S<E1, EE1>, S<_, EE2>>::setup(&pp).unwrap();

    // produce a compressed SNARK
    let compressed_snark =
      CompressedSNARK::<_, S<E1, EE1>, S<_, EE2>>::prove(&pp, &pk, &recursive_snark).unwrap();

    // verify the compressed SNARK
    compressed_snark
      .verify(&vk, num_steps, &z0_primary, &z0_secondary)
      .unwrap();
  }

  #[test]
  fn test_ivc_nondet_with_compression() {
    test_ivc_nondet_with_compression_with::<PallasEngine, EE<_>, EE<_>>();
    test_ivc_nondet_with_compression_with::<Bn256EngineIPA, EE<_>, EE<_>>();
    test_ivc_nondet_with_compression_with::<Secp256k1Engine, EE<_>, EE<_>>();
    test_ivc_nondet_with_compression_with::<Bn256EngineZM, ZMPCS<Bn256, _>, EE<_>>();
  }

  fn test_ivc_base_with<E1>()
  where
    E1: CurveCycleEquipped,
  {
    let test_circuit1 = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
    let test_circuit2 = CubicCircuit::<<Dual<E1> as Engine>::Scalar>::default();

    // produce public parameters
    let pp = PublicParams::<E1>::setup(
      &test_circuit1,
      &test_circuit2,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1>::new(
      &pp,
      &test_circuit1,
      &test_circuit2,
      &[<E1 as Engine>::Scalar::ONE],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )
    .unwrap();

    // produce a recursive SNARK
    recursive_snark
      .prove_step(&pp, &test_circuit1, &test_circuit2)
      .unwrap();

    // verify the recursive SNARK
    let (zn_primary, zn_secondary) = recursive_snark
      .verify(
        &pp,
        num_steps,
        &[<E1 as Engine>::Scalar::ONE],
        &[<Dual<E1> as Engine>::Scalar::ZERO],
      )
      .unwrap();

    assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
    assert_eq!(zn_secondary, vec![<Dual<E1> as Engine>::Scalar::from(5u64)]);
  }

  #[test]
  fn test_ivc_base() {
    test_ivc_base_with::<PallasEngine>();
    test_ivc_base_with::<Bn256EngineKZG>();
    test_ivc_base_with::<Secp256k1Engine>();
  }

  fn test_setup_with<E1: CurveCycleEquipped>() {
    #[derive(Clone, Debug, Default)]
    struct CircuitWithInputize<F: PrimeField> {
      _p: PhantomData<F>,
    }

    impl<F: PrimeField> StepCircuit<F> for CircuitWithInputize<F> {
      fn arity(&self) -> usize {
        1
      }

      fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
      ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        let x = &z[0];
        // a simplified version of this test would only have one input
        // but beside the Nova Public parameter requirement for a num_io = 2, being
        // probed in this test, we *also* require num_io to be even, so
        // negative testing requires at least 4 inputs
        let y = x.square(cs.namespace(|| "x_sq"))?;
        y.inputize(cs.namespace(|| "y"))?; // inputize y
        let y2 = x.square(cs.namespace(|| "x_sq2"))?;
        y2.inputize(cs.namespace(|| "y2"))?; // inputize y2
        let y3 = x.square(cs.namespace(|| "x_sq3"))?;
        y3.inputize(cs.namespace(|| "y3"))?; // inputize y2
        let y4 = x.square(cs.namespace(|| "x_sq4"))?;
        y4.inputize(cs.namespace(|| "y4"))?; // inputize y2
        Ok(vec![y, y2, y3, y4])
      }
    }

    // produce public parameters with trivial secondary
    let circuit = CircuitWithInputize::<<E1 as Engine>::Scalar>::default();
    let pp = PublicParams::<E1>::setup(
      &circuit,
      &TrivialCircuit::default(),
      &*default_ck_hint(),
      &*default_ck_hint(),
    );
    assert!(pp.is_err());
    assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));

    // produce public parameters with the trivial primary
    let circuit = CircuitWithInputize::<<Dual<E1> as Engine>::Scalar>::default();
    let pp = PublicParams::<E1>::setup(
      &TrivialCircuit::default(),
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );
    assert!(pp.is_err());
    assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));
  }

  #[test]
  fn test_setup() {
    test_setup_with::<Bn256EngineKZG>();
  }
}
