//! This library implements `SuperNova`, a Non-Uniform IVC based on Nova.

use std::marker::PhantomData;
use std::ops::Index;

use crate::{
  bellpepper::shape_cs::ShapeCS,
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_HASH_BITS},
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{commitment_key_size, R1CSInstance, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  scalar_as_base,
  traits::{
    circuit_supernova::StepCircuit,
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    AbsorbInROTrait, Group, ROConstants, ROConstantsCircuit, ROTrait,
  },
  CircuitShape, Commitment, CommitmentKey,
};

use abomonation::Abomonation;
use abomonation_derive::Abomonation;
use ff::{Field, PrimeField};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::bellpepper::{
  r1cs::{NovaShape, NovaWitness},
  solver::SatisfyingAssignment,
};
use bellpepper_core::ConstraintSystem;

use crate::nifs::NIFS;

mod circuit; // declare the module first
use circuit::{
  SuperNovaAugmentedCircuit, SuperNovaAugmentedCircuitInputs, SuperNovaAugmentedCircuitParams,
};

use self::error::SuperNovaError;

pub mod error;
pub(crate) mod utils;

#[cfg(test)]
mod test;

/// A struct that manages all the digests of the primary circuits of a SuperNova instance
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircuitDigests<G: Group> {
  digests: Vec<G::Scalar>,
}

impl<G: Group> SimpleDigestible for CircuitDigests<G> {}

impl<G: Group> std::ops::Deref for CircuitDigests<G> {
  type Target = Vec<G::Scalar>;

  fn deref(&self) -> &Self::Target {
    &self.digests
  }
}

impl<G: Group> CircuitDigests<G> {
  /// Construct a new [CircuitDigests]
  pub fn new(digests: Vec<G::Scalar>) -> Self {
    CircuitDigests { digests }
  }

  /// Return the [CircuitDigests]' digest.
  pub fn digest(&self) -> G::Scalar {
    let dc: DigestComputer<'_, <G as Group>::Scalar, CircuitDigests<G>> = DigestComputer::new(self);
    dc.digest().expect("Failure in computing digest")
  }
}

/// A vector of [CircuitParams] corresponding to a set of [PublicParams]
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  /// The internal circuit shapes
  pub circuit_shapes: Vec<CircuitShape<G1>>,

  ro_consts_primary: ROConstants<G1>,
  ro_consts_circuit_primary: ROConstantsCircuit<G2>,
  ck_primary: CommitmentKey<G1>, // This is shared between all circuit params
  augmented_circuit_params_primary: SuperNovaAugmentedCircuitParams,

  ro_consts_secondary: ROConstants<G2>,
  ro_consts_circuit_secondary: ROConstantsCircuit<G1>,
  ck_secondary: CommitmentKey<G2>,
  circuit_shape_secondary: CircuitShape<G2>,
  augmented_circuit_params_secondary: SuperNovaAugmentedCircuitParams,

  /// Digest constructed from this `PublicParams`' parameters
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<G1::Scalar>,
  _p: PhantomData<(C1, C2)>,
}

/// Auxilliary [PublicParams] information about the commitment keys and
/// secondary circuit. This is used as a helper struct when reconstructing
/// [PublicParams] downstream in lurk.
#[derive(Clone, PartialEq, Serialize, Deserialize, Abomonation)]
#[serde(bound = "")]
#[abomonation_bounds(
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  <G1::Scalar as PrimeField>::Repr: Abomonation,
  <G2::Scalar as PrimeField>::Repr: Abomonation,
)]
pub struct AuxParams<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  ro_consts_primary: ROConstants<G1>,
  ro_consts_circuit_primary: ROConstantsCircuit<G2>,
  augmented_circuit_params_primary: SuperNovaAugmentedCircuitParams,

  ro_consts_secondary: ROConstants<G2>,
  ro_consts_circuit_secondary: ROConstantsCircuit<G1>,
  ck_secondary: CommitmentKey<G2>,
  circuit_shape_secondary: CircuitShape<G2>,
  augmented_circuit_params_secondary: SuperNovaAugmentedCircuitParams,

  #[abomonate_with(<G1::Scalar as PrimeField>::Repr)]
  digest: G1::Scalar,
}

impl<G1, G2, C1, C2> Index<usize> for PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  type Output = CircuitShape<G1>;

  fn index(&self, index: usize) -> &Self::Output {
    &self.circuit_shapes[index]
  }
}

impl<G1, G2, C1, C2> SimpleDigestible for PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
}

impl<G1, G2, C1, C2> PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  /// Construct a new [PublicParams]
  pub fn new<NC: NonUniformCircuit<G1, G2, C1, C2>>(non_uniform_circuit: &NC) -> Self {
    let num_circuits = non_uniform_circuit.num_circuits();

    let augmented_circuit_params_primary =
      SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let ro_consts_primary: ROConstants<G1> = ROConstants::<G1>::default();
    // ro_consts_circuit_primary are parameterized by G2 because the type alias uses G2::Base = G1::Scalar
    let ro_consts_circuit_primary: ROConstantsCircuit<G2> = ROConstantsCircuit::<G2>::default();

    let circuit_shapes = (0..num_circuits)
      .map(|i| {
        let c_primary = non_uniform_circuit.primary_circuit(i);
        let F_arity = c_primary.arity();
        // Initialize ck for the primary
        let circuit_primary: SuperNovaAugmentedCircuit<'_, G2, C1> = SuperNovaAugmentedCircuit::new(
          &augmented_circuit_params_primary,
          None,
          &c_primary,
          ro_consts_circuit_primary.clone(),
          num_circuits,
        );
        let mut cs: ShapeCS<G1> = ShapeCS::new();
        let _ = circuit_primary.synthesize(&mut cs);
        // We use the largest commitment_key for all instances
        let r1cs_shape_primary = cs.r1cs_shape();
        CircuitShape::new(r1cs_shape_primary, F_arity)
      })
      .collect::<Vec<_>>();

    let ck_primary = Self::compute_primary_ck(&circuit_shapes);

    let augmented_circuit_params_secondary =
      SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts_secondary: ROConstants<G2> = ROConstants::<G2>::default();
    let c_secondary = non_uniform_circuit.secondary_circuit();
    let F_arity_secondary = c_secondary.arity();
    let ro_consts_circuit_secondary: ROConstantsCircuit<G1> = ROConstantsCircuit::<G1>::default();

    let circuit_secondary: SuperNovaAugmentedCircuit<'_, G1, C2> = SuperNovaAugmentedCircuit::new(
      &augmented_circuit_params_secondary,
      None,
      &c_secondary,
      ro_consts_circuit_secondary.clone(),
      num_circuits,
    );
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit_secondary.synthesize(&mut cs);
    let (r1cs_shape_secondary, ck_secondary) = cs.r1cs_shape_and_key(None);
    let circuit_shape_secondary = CircuitShape::new(r1cs_shape_secondary, F_arity_secondary);

    let pp = PublicParams {
      circuit_shapes,
      ro_consts_primary,
      ro_consts_circuit_primary,
      ck_primary,
      augmented_circuit_params_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      ck_secondary,
      circuit_shape_secondary,
      augmented_circuit_params_secondary,
      digest: OnceCell::new(),
      _p: PhantomData,
    };

    // make sure to initialize the `OnceCell` and compute the digest
    // and avoid paying for unexpected performance costs later
    pp.digest();
    pp
  }

  /// Breaks down an instance of [PublicParams] into the circuit params, auxilliary params, and primary commitment key.
  pub fn into_parts(self) -> (Vec<CircuitShape<G1>>, AuxParams<G1, G2>, CommitmentKey<G1>) {
    let digest = self.digest();

    let PublicParams {
      circuit_shapes,
      ro_consts_primary,
      ro_consts_circuit_primary,
      ck_primary,
      augmented_circuit_params_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      ck_secondary,
      circuit_shape_secondary,
      augmented_circuit_params_secondary,
      digest: _digest,
      _p,
    } = self;

    let aux_params = AuxParams {
      ro_consts_primary,
      ro_consts_circuit_primary,
      augmented_circuit_params_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      ck_secondary,
      circuit_shape_secondary,
      augmented_circuit_params_secondary,
      digest,
    };

    (circuit_shapes, aux_params, ck_primary)
  }

  /// Create a [PublicParams] from a vector of raw [CircuitShape], auxilliary params, and the primary commitment key.
  pub fn from_parts(
    circuit_shapes: Vec<CircuitShape<G1>>,
    aux_params: AuxParams<G1, G2>,
    ck_primary: CommitmentKey<G1>,
  ) -> Self {
    let pp = PublicParams {
      circuit_shapes,
      ro_consts_primary: aux_params.ro_consts_primary,
      ro_consts_circuit_primary: aux_params.ro_consts_circuit_primary,
      ck_primary,
      augmented_circuit_params_primary: aux_params.augmented_circuit_params_primary,
      ro_consts_secondary: aux_params.ro_consts_secondary,
      ro_consts_circuit_secondary: aux_params.ro_consts_circuit_secondary,
      ck_secondary: aux_params.ck_secondary,
      circuit_shape_secondary: aux_params.circuit_shape_secondary,
      augmented_circuit_params_secondary: aux_params.augmented_circuit_params_secondary,
      digest: OnceCell::new(),
      _p: PhantomData,
    };
    assert_eq!(
      aux_params.digest,
      pp.digest(),
      "param data is invalid; aux_params contained the incorrect digest"
    );
    pp
  }

  /// Create a [PublicParams] from a vector of raw [CircuitShape], auxilliary params, and the primary commitment key.
  /// We don't check that the `aux_params.digest` is a valid digest for the created params.
  pub fn from_parts_unchecked(
    circuit_shapes: Vec<CircuitShape<G1>>,
    aux_params: AuxParams<G1, G2>,
    ck_primary: CommitmentKey<G1>,
  ) -> Self {
    PublicParams {
      circuit_shapes,
      ro_consts_primary: aux_params.ro_consts_primary,
      ro_consts_circuit_primary: aux_params.ro_consts_circuit_primary,
      ck_primary,
      augmented_circuit_params_primary: aux_params.augmented_circuit_params_primary,
      ro_consts_secondary: aux_params.ro_consts_secondary,
      ro_consts_circuit_secondary: aux_params.ro_consts_circuit_secondary,
      ck_secondary: aux_params.ck_secondary,
      circuit_shape_secondary: aux_params.circuit_shape_secondary,
      augmented_circuit_params_secondary: aux_params.augmented_circuit_params_secondary,
      digest: aux_params.digest.into(),
      _p: PhantomData,
    }
  }

  /// Compute primary and secondary commitment keys sized to handle the largest of the circuits in the provided
  /// `CircuitShape`.
  fn compute_primary_ck(circuit_params: &[CircuitShape<G1>]) -> CommitmentKey<G1> {
    let size_primary = circuit_params
      .iter()
      .map(|circuit| commitment_key_size(&circuit.r1cs_shape, None))
      .max()
      .unwrap();

    G1::CE::setup(b"ck", size_primary)
  }

  /// Return the [PublicParams]' digest.
  pub fn digest(&self) -> G1::Scalar {
    self
      .digest
      .get_or_try_init(|| {
        let dc: DigestComputer<'_, <G1 as Group>::Scalar, PublicParams<G1, G2, C1, C2>> =
          DigestComputer::new(self);
        dc.digest()
      })
      .cloned()
      .expect("Failure in retrieving digest")
  }

  /// All of the primary circuit digests of this [PublicParams]
  pub fn circuit_param_digests(&self) -> CircuitDigests<G1> {
    let digests = self
      .circuit_shapes
      .iter()
      .map(|cp| cp.digest())
      .collect::<Vec<_>>();
    CircuitDigests { digests }
  }
}

/// A SNARK that proves the correct execution of an non-uniform incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  r_W_primary: Vec<Option<RelaxedR1CSWitness<G1>>>,
  r_U_primary: Vec<Option<RelaxedR1CSInstance<G1>>>,
  r_W_secondary: Vec<Option<RelaxedR1CSWitness<G2>>>, // usually r_W_secondary.len() == 1
  r_U_secondary: Vec<Option<RelaxedR1CSInstance<G2>>>, // usually r_U_secondary.len() == 1
  l_w_secondary: R1CSWitness<G2>,
  l_u_secondary: R1CSInstance<G2>,
  pp_digest: G1::Scalar,
  i: usize,
  zi_primary: Vec<G1::Scalar>,
  zi_secondary: Vec<G2::Scalar>,
  program_counter: G1::Scalar,
  augmented_circuit_index: usize,
  num_augmented_circuits: usize,
}

impl<G1, G2> RecursiveSNARK<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  /// iterate base step to get new instance of recursive SNARK
  #[allow(clippy::too_many_arguments)]
  pub fn iter_base_step<C1: StepCircuit<G1::Scalar>, C2: StepCircuit<G2::Scalar>>(
    pp: &PublicParams<G1, G2, C1, C2>,
    circuit_index: usize,
    c_primary: &C1,
    c_secondary: &C2,
    initial_program_counter: Option<G1::Scalar>,
    first_augmented_circuit_index: usize,
    num_augmented_circuits: usize,
    z0_primary: &[G1::Scalar],
    z0_secondary: &[G2::Scalar],
  ) -> Result<Self, SuperNovaError> {
    if z0_primary.len() != pp[circuit_index].F_arity
      || z0_secondary.len() != pp.circuit_shape_secondary.F_arity
    {
      return Err(SuperNovaError::NovaError(
        NovaError::InvalidStepOutputLength,
      ));
    }

    // base case for the primary
    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs_primary: SuperNovaAugmentedCircuitInputs<'_, G2> =
      SuperNovaAugmentedCircuitInputs::new(
        scalar_as_base::<G1>(pp.digest()),
        G1::Scalar::ZERO,
        z0_primary,
        None,
        None,
        None,
        None,
        initial_program_counter,
        G1::Scalar::ZERO, // set augmented circuit index selector to 0 in base case
      );

    let circuit_primary: SuperNovaAugmentedCircuit<'_, G2, C1> = SuperNovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_primary,
      Some(inputs_primary),
      c_primary,
      pp.ro_consts_circuit_primary.clone(),
      num_augmented_circuits,
    );

    let (zi_primary_pc_next, zi_primary) =
      circuit_primary.synthesize(&mut cs_primary).map_err(|err| {
        debug!("err {:?}", err);
        NovaError::SynthesisError
      })?;
    if zi_primary.len() != pp[circuit_index].F_arity {
      return Err(SuperNovaError::NovaError(
        NovaError::InvalidStepOutputLength,
      ));
    }
    let (u_primary, w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp[circuit_index].r1cs_shape, &pp.ck_primary)
      .map_err(|err| {
        debug!("err {:?}", err);
        NovaError::SynthesisError
      })?;

    // base case for the secondary
    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
    let inputs_secondary: SuperNovaAugmentedCircuitInputs<'_, G1> =
      SuperNovaAugmentedCircuitInputs::new(
        pp.digest(),
        G2::Scalar::ZERO,
        z0_secondary,
        None,
        None,
        Some(&u_primary),
        None,
        None,
        G2::Scalar::from(circuit_index as u64),
      );
    let circuit_secondary: SuperNovaAugmentedCircuit<'_, G1, C2> = SuperNovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_secondary,
      Some(inputs_secondary),
      c_secondary,
      pp.ro_consts_circuit_secondary.clone(),
      num_augmented_circuits,
    );
    let (_, zi_secondary) = circuit_secondary
      .synthesize(&mut cs_secondary)
      .map_err(|_| NovaError::SynthesisError)?;
    if zi_secondary.len() != pp.circuit_shape_secondary.F_arity {
      return Err(NovaError::InvalidStepOutputLength.into());
    }
    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.circuit_shape_secondary.r1cs_shape, &pp.ck_secondary)
      .map_err(|_| SuperNovaError::NovaError(NovaError::UnSat))?;

    // IVC proof for the primary circuit
    let l_w_primary = w_primary;
    let l_u_primary = u_primary;
    let r_W_primary =
      RelaxedR1CSWitness::from_r1cs_witness(&pp[circuit_index].r1cs_shape, &l_w_primary);

    let r_U_primary = RelaxedR1CSInstance::from_r1cs_instance(
      &pp.ck_primary,
      &pp[circuit_index].r1cs_shape,
      &l_u_primary,
    );

    // IVC proof of the secondary circuit
    let l_w_secondary = w_secondary;
    let l_u_secondary = u_secondary;
    let r_W_secondary = vec![Some(RelaxedR1CSWitness::<G2>::default(
      &pp.circuit_shape_secondary.r1cs_shape,
    ))];
    let r_U_secondary = vec![Some(RelaxedR1CSInstance::default(
      &pp.ck_secondary,
      &pp.circuit_shape_secondary.r1cs_shape,
    ))];

    // Outputs of the two circuits and next program counter thus far.
    let zi_primary = zi_primary
      .iter()
      .map(|v| v.get_value().ok_or(NovaError::SynthesisError.into()))
      .collect::<Result<Vec<<G1 as Group>::Scalar>, SuperNovaError>>()?;
    let zi_primary_pc_next = zi_primary_pc_next
      .expect("zi_primary_pc_next missing")
      .get_value()
      .ok_or::<SuperNovaError>(NovaError::SynthesisError.into())?;
    let zi_secondary = zi_secondary
      .iter()
      .map(|v| v.get_value().ok_or(NovaError::SynthesisError.into()))
      .collect::<Result<Vec<<G2 as Group>::Scalar>, SuperNovaError>>()?;

    // handle the base case by initialize U_next in next round
    let r_W_primary_initial_list = (0..num_augmented_circuits)
      .map(|i| (i == first_augmented_circuit_index).then(|| r_W_primary.clone()))
      .collect::<Vec<Option<RelaxedR1CSWitness<G1>>>>();

    let r_U_primary_initial_list = (0..num_augmented_circuits)
      .map(|i| (i == first_augmented_circuit_index).then(|| r_U_primary.clone()))
      .collect::<Vec<Option<RelaxedR1CSInstance<G1>>>>();

    Ok(Self {
      r_W_primary: r_W_primary_initial_list,
      r_U_primary: r_U_primary_initial_list,
      r_W_secondary,
      r_U_secondary,
      l_w_secondary,
      l_u_secondary,
      pp_digest: pp.digest(),
      i: 0_usize, // after base case, next iteration start from 1
      zi_primary,
      zi_secondary,
      program_counter: zi_primary_pc_next,
      augmented_circuit_index: first_augmented_circuit_index,
      num_augmented_circuits,
    })
  }
  /// executing a step of the incremental computation
  #[allow(clippy::too_many_arguments)]
  #[tracing::instrument(skip_all, name = "supernova::RecursiveSNARK::prove_step")]
  pub fn prove_step<C1: StepCircuit<G1::Scalar>, C2: StepCircuit<G2::Scalar>>(
    &mut self,
    pp: &PublicParams<G1, G2, C1, C2>,
    circuit_index: usize,
    c_primary: &C1,
    c_secondary: &C2,
    z0_primary: &[G1::Scalar],
    z0_secondary: &[G2::Scalar],
  ) -> Result<(), SuperNovaError> {
    // First step was already done in the constructor
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    if self.r_U_secondary.len() != 1 || self.r_W_secondary.len() != 1 {
      return Err(NovaError::ProofVerifyError.into());
    }

    if z0_primary.len() != pp[circuit_index].F_arity
      || z0_secondary.len() != pp.circuit_shape_secondary.F_arity
    {
      return Err(SuperNovaError::NovaError(
        NovaError::InvalidInitialInputLength,
      ));
    }

    // fold the secondary circuit's instance
    let (nifs_secondary, (r_U_secondary_folded, r_W_secondary_folded)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<G1>(self.pp_digest),
      &pp.circuit_shape_secondary.r1cs_shape,
      self.r_U_secondary[0].as_ref().unwrap(),
      self.r_W_secondary[0].as_ref().unwrap(),
      &self.l_u_secondary,
      &self.l_w_secondary,
    )
    .map_err(SuperNovaError::NovaError)?;

    // clone and updated running instance on respective circuit_index
    let r_U_secondary_next = r_U_secondary_folded;
    let r_W_secondary_next = r_W_secondary_folded;

    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let T =
      Commitment::<G2>::decompress(&nifs_secondary.comm_T).map_err(SuperNovaError::NovaError)?;
    let inputs_primary: SuperNovaAugmentedCircuitInputs<'_, G2> =
      SuperNovaAugmentedCircuitInputs::new(
        scalar_as_base::<G1>(self.pp_digest),
        G1::Scalar::from(self.i as u64),
        z0_primary,
        Some(&self.zi_primary),
        Some(&self.r_U_secondary),
        Some(&self.l_u_secondary),
        Some(&T),
        Some(self.program_counter),
        G1::Scalar::ZERO,
      );

    let circuit_primary: SuperNovaAugmentedCircuit<'_, G2, C1> = SuperNovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_primary,
      Some(inputs_primary),
      c_primary,
      pp.ro_consts_circuit_primary.clone(),
      self.num_augmented_circuits,
    );

    let (zi_primary_pc_next, zi_primary) = circuit_primary
      .synthesize(&mut cs_primary)
      .map_err(|_| SuperNovaError::NovaError(NovaError::SynthesisError))?;
    if zi_primary.len() != pp[circuit_index].F_arity {
      return Err(SuperNovaError::NovaError(
        NovaError::InvalidInitialInputLength,
      ));
    }

    let (l_u_primary, l_w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp[circuit_index].r1cs_shape, &pp.ck_primary)
      .map_err(|_| SuperNovaError::NovaError(NovaError::UnSat))?;

    // Split into `if let`/`else` statement
    // to avoid `returns a value referencing data owned by closure` error on `&RelaxedR1CSInstance::default` and `RelaxedR1CSWitness::default`
    let (nifs_primary, (r_U_primary_folded, r_W_primary_folded)) = match (
      self.r_U_primary.get(circuit_index),
      self.r_W_primary.get(circuit_index),
    ) {
      (Some(Some(r_U_primary)), Some(Some(r_W_primary))) => NIFS::prove(
        &pp.ck_primary,
        &pp.ro_consts_primary,
        &self.pp_digest,
        &pp[circuit_index].r1cs_shape,
        r_U_primary,
        r_W_primary,
        &l_u_primary,
        &l_w_primary,
      )
      .map_err(SuperNovaError::NovaError)?,
      _ => NIFS::prove(
        &pp.ck_primary,
        &pp.ro_consts_primary,
        &self.pp_digest,
        &pp[circuit_index].r1cs_shape,
        &RelaxedR1CSInstance::default(&pp.ck_primary, &pp[circuit_index].r1cs_shape),
        &RelaxedR1CSWitness::default(&pp[circuit_index].r1cs_shape),
        &l_u_primary,
        &l_w_primary,
      )
      .map_err(SuperNovaError::NovaError)?,
    };

    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
    let binding =
      Commitment::<G1>::decompress(&nifs_primary.comm_T).map_err(SuperNovaError::NovaError)?;
    let inputs_secondary: SuperNovaAugmentedCircuitInputs<'_, G1> =
      SuperNovaAugmentedCircuitInputs::new(
        self.pp_digest,
        G2::Scalar::from(self.i as u64),
        z0_secondary,
        Some(&self.zi_secondary),
        Some(&self.r_U_primary),
        Some(&l_u_primary),
        Some(&binding),
        None,
        G2::Scalar::from(circuit_index as u64),
      );

    let circuit_secondary: SuperNovaAugmentedCircuit<'_, G1, C2> = SuperNovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_secondary,
      Some(inputs_secondary),
      c_secondary,
      pp.ro_consts_circuit_secondary.clone(),
      self.num_augmented_circuits,
    );
    let (_, zi_secondary) = circuit_secondary
      .synthesize(&mut cs_secondary)
      .map_err(|_| SuperNovaError::NovaError(NovaError::SynthesisError))?;
    if zi_secondary.len() != pp.circuit_shape_secondary.F_arity {
      return Err(SuperNovaError::NovaError(
        NovaError::InvalidInitialInputLength,
      ));
    }

    let (l_u_secondary_next, l_w_secondary_next) = cs_secondary
      .r1cs_instance_and_witness(&pp.circuit_shape_secondary.r1cs_shape, &pp.ck_secondary)
      .map_err(|_| SuperNovaError::NovaError(NovaError::UnSat))?;

    // update the running instances and witnesses
    let zi_primary = zi_primary
      .iter()
      .map(|v| {
        v.get_value()
          .ok_or(SuperNovaError::NovaError(NovaError::SynthesisError))
      })
      .collect::<Result<Vec<<G1 as Group>::Scalar>, SuperNovaError>>()?;
    let zi_primary_pc_next = zi_primary_pc_next
      .expect("zi_primary_pc_next missing")
      .get_value()
      .ok_or(SuperNovaError::NovaError(NovaError::SynthesisError))?;
    let zi_secondary = zi_secondary
      .iter()
      .map(|v| {
        v.get_value()
          .ok_or(SuperNovaError::NovaError(NovaError::SynthesisError))
      })
      .collect::<Result<Vec<<G2 as Group>::Scalar>, SuperNovaError>>()?;

    if zi_primary.len() != pp[circuit_index].F_arity
      || zi_secondary.len() != pp.circuit_shape_secondary.F_arity
    {
      return Err(SuperNovaError::NovaError(
        NovaError::InvalidStepOutputLength,
      ));
    }

    // clone and updated running instance on respective circuit_index
    self.r_U_primary[circuit_index] = Some(r_U_primary_folded);
    self.r_W_primary[circuit_index] = Some(r_W_primary_folded);
    self.r_W_secondary = vec![Some(r_W_secondary_next)];
    self.r_U_secondary = vec![Some(r_U_secondary_next)];
    self.l_w_secondary = l_w_secondary_next;
    self.l_u_secondary = l_u_secondary_next;
    self.i += 1;
    self.zi_primary = zi_primary;
    self.zi_secondary = zi_secondary;
    self.program_counter = zi_primary_pc_next;
    self.augmented_circuit_index = circuit_index;
    Ok(())
  }

  /// verify recursive snark
  pub fn verify<C1: StepCircuit<G1::Scalar>, C2: StepCircuit<G2::Scalar>>(
    &self,
    pp: &PublicParams<G1, G2, C1, C2>,
    circuit_index: usize,
    z0_primary: &[G1::Scalar],
    z0_secondary: &[G2::Scalar],
  ) -> Result<(Vec<G1::Scalar>, Vec<G2::Scalar>), SuperNovaError> {
    // number of steps cannot be zero
    if self.i == 0 {
      debug!("must verify on valid RecursiveSNARK where i > 0");
      return Err(SuperNovaError::NovaError(NovaError::ProofVerifyError));
    }

    // check the (relaxed) R1CS instances public outputs.
    if self.l_u_secondary.X.len() != 2 {
      return Err(SuperNovaError::NovaError(NovaError::ProofVerifyError));
    }

    if self.r_U_secondary.len() != 1 || self.r_W_secondary.len() != 1 {
      return Err(SuperNovaError::NovaError(NovaError::ProofVerifyError));
    }

    self.r_U_primary[circuit_index]
      .as_ref()
      .map_or(Ok(()), |U| {
        if U.X.len() != 2 {
          debug!("r_U_primary got instance length {:?} != {:?}", U.X.len(), 2);
          Err(SuperNovaError::NovaError(NovaError::ProofVerifyError))
        } else {
          Ok(())
        }
      })?;

    self.r_U_secondary[0].as_ref().map_or(Ok(()), |U| {
      if U.X.len() != 2 {
        debug!(
          "r_U_secondary got instance length {:?} != {:?}",
          U.X.len(),
          2
        );
        Err(SuperNovaError::NovaError(NovaError::ProofVerifyError))
      } else {
        Ok(())
      }
    })?;

    let num_field_primary_ro = 3 // params_next, i_new, program_counter_new
    + 2 * pp[circuit_index].F_arity // zo, z1
    + (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // # 1 * (7 + [X0, X1]*#num_limb)

    // secondary circuit
    let num_field_secondary_ro = 2 // params_next, i_new
    + 2 * pp.circuit_shape_secondary.F_arity // zo, z1
    + self.num_augmented_circuits * (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // #num_augmented_circuits * (7 + [X0, X1]*#num_limb)

    let (hash_primary, hash_secondary) = {
      let mut hasher = <G2 as Group>::RO::new(pp.ro_consts_secondary.clone(), num_field_primary_ro);
      hasher.absorb(self.pp_digest);
      hasher.absorb(G1::Scalar::from(self.i as u64));
      hasher.absorb(self.program_counter);

      for e in z0_primary {
        hasher.absorb(*e);
      }
      for e in &self.zi_primary {
        hasher.absorb(*e);
      }
      self.r_U_secondary[0].as_ref().map_or(
        Err(SuperNovaError::NovaError(NovaError::ProofVerifyError)),
        |U| {
          U.absorb_in_ro(&mut hasher);
          Ok(())
        },
      )?;

      let mut hasher2 =
        <G1 as Group>::RO::new(pp.ro_consts_primary.clone(), num_field_secondary_ro);
      hasher2.absorb(scalar_as_base::<G1>(self.pp_digest));
      hasher2.absorb(G2::Scalar::from(self.i as u64));

      for e in z0_secondary {
        hasher2.absorb(*e);
      }
      for e in &self.zi_secondary {
        hasher2.absorb(*e);
      }
      let default_value =
        RelaxedR1CSInstance::default(&pp.ck_primary, &pp[circuit_index].r1cs_shape);
      self.r_U_primary.iter().for_each(|U| {
        U.as_ref()
          .unwrap_or(&default_value)
          .absorb_in_ro(&mut hasher2);
      });

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != self.l_u_secondary.X[0] {
      debug!(
        "hash_primary {:?} not equal l_u_secondary.X[0] {:?}",
        hash_primary, self.l_u_secondary.X[0]
      );
      return Err(SuperNovaError::NovaError(NovaError::ProofVerifyError));
    }
    if hash_secondary != scalar_as_base::<G2>(self.l_u_secondary.X[1]) {
      debug!(
        "hash_secondary {:?} not equal l_u_secondary.X[1] {:?}",
        hash_secondary, self.l_u_secondary.X[1]
      );
      return Err(SuperNovaError::NovaError(NovaError::ProofVerifyError));
    }

    // check the satisfiability of the provided `circuit_index` instance
    let default_instance =
      RelaxedR1CSInstance::default(&pp.ck_primary, &pp[circuit_index].r1cs_shape);
    let default_witness = RelaxedR1CSWitness::default(&pp[circuit_index].r1cs_shape);
    let (res_r_primary, (res_r_secondary, res_l_secondary)) = rayon::join(
      || {
        pp[circuit_index].r1cs_shape.is_sat_relaxed(
          &pp.ck_primary,
          self.r_U_primary[circuit_index]
            .as_ref()
            .unwrap_or(&default_instance),
          self.r_W_primary[circuit_index]
            .as_ref()
            .unwrap_or(&default_witness),
        )
      },
      || {
        rayon::join(
          || {
            pp.circuit_shape_secondary.r1cs_shape.is_sat_relaxed(
              &pp.ck_secondary,
              self.r_U_secondary[0].as_ref().unwrap(),
              self.r_W_secondary[0].as_ref().unwrap(),
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

    res_r_primary.map_err(|err| match err {
      NovaError::UnSatIndex(i) => SuperNovaError::UnSatIndex("r_primary", i),
      e => SuperNovaError::NovaError(e),
    })?;
    res_r_secondary.map_err(|err| match err {
      NovaError::UnSatIndex(i) => SuperNovaError::UnSatIndex("r_secondary", i),
      e => SuperNovaError::NovaError(e),
    })?;
    res_l_secondary.map_err(|err| match err {
      NovaError::UnSatIndex(i) => SuperNovaError::UnSatIndex("l_secondary", i),
      e => SuperNovaError::NovaError(e),
    })?;

    Ok((self.zi_primary.clone(), self.zi_secondary.clone()))
  }

  /// get program counter
  pub fn get_program_counter(&self) -> G1::Scalar {
    self.program_counter
  }
}

/// SuperNova helper trait, for implementors that provide sets of sub-circuits to be proved via NIVC. `C1` must be a
/// type (likely an `Enum`) for which a potentially-distinct instance can be supplied for each `index` below
/// `self.num_circuits()`.
pub trait NonUniformCircuit<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  /// Initial program counter, defaults to zero.
  fn initial_program_counter(&self) -> G1::Scalar {
    G1::Scalar::ZERO
  }

  /// How many circuits are provided?
  fn num_circuits(&self) -> usize;

  /// Return a new instance of the primary circuit at `index`.
  fn primary_circuit(&self, circuit_index: usize) -> C1;

  /// Return a new instance of the secondary circuit.
  fn secondary_circuit(&self) -> C2;
}

/// Compute the circuit digest of a supernova [StepCircuit].
///
/// Note for callers: This function should be called with its performance characteristics in mind.
/// It will synthesize and digest the full `circuit` given.
pub fn circuit_digest<
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C: StepCircuit<G1::Scalar>,
>(
  circuit: &C,
  num_augmented_circuits: usize,
) -> G1::Scalar {
  let augmented_circuit_params =
    SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);

  // ro_consts_circuit are parameterized by G2 because the type alias uses G2::Base = G1::Scalar
  let ro_consts_circuit: ROConstantsCircuit<G2> = ROConstantsCircuit::<G2>::default();

  // Initialize ck for the primary
  let augmented_circuit: SuperNovaAugmentedCircuit<'_, G2, C> = SuperNovaAugmentedCircuit::new(
    &augmented_circuit_params,
    None,
    circuit,
    ro_consts_circuit,
    num_augmented_circuits,
  );
  let mut cs: ShapeCS<G1> = ShapeCS::new();
  let _ = augmented_circuit.synthesize(&mut cs);

  let F_arity = circuit.arity();
  let circuit_params = CircuitShape::new(cs.r1cs_shape(), F_arity);
  circuit_params.digest()
}
