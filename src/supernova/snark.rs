//! In this module we define a compressing SNARK for Supernova proofs

use std::marker::PhantomData;

use ff::PrimeField;

use crate::{
  constants::NUM_HASH_BITS,
  gadgets::utils::scalar_as_base,
  nifs::NIFS,
  r1cs::RelaxedR1CSInstance,
  traits::{
    circuit_supernova::StepCircuit, snark::RelaxedR1CSSNARKTrait, AbsorbInROTrait, Group, ROTrait,
  },
  NovaError, R1CSInstance,
};

use super::{error::SuperNovaError, PublicParams, RecursiveSNARK};

/// The prover key for the Supernova CompressedSNARK
pub struct ProverKey<G1, G2, C1, C2, S1, S2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
  S1: RelaxedR1CSSNARKTrait<G1>,
  S2: RelaxedR1CSSNARKTrait<G2>,
{
  pks_primary: Vec<S1::ProverKey>,
  pk_secondary: S2::ProverKey,
  _p: PhantomData<(C1, C2)>,
}

/// The verifier key for the Supernova CompressedSNARK
pub struct VerifierKey<G1, G2, C1, C2, S1, S2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
  S1: RelaxedR1CSSNARKTrait<G1>,
  S2: RelaxedR1CSSNARKTrait<G2>,
{
  vks_primary: Vec<S1::VerifierKey>,
  vk_secondary: S2::VerifierKey,
  _p: PhantomData<(C1, C2)>,
}

#[derive(Clone, Debug)]
/// The SNARK that proves the knowledge of a valid Supernova `RecursiveSNARK`
pub struct CompressedSNARK<G1, G2, C1, C2, S1, S2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
  S1: RelaxedR1CSSNARKTrait<G1>,
  S2: RelaxedR1CSSNARKTrait<G2>,
{
  r_U_primary: Vec<RelaxedR1CSInstance<G1>>,
  r_W_snark_primary: Vec<S1>,

  r_U_secondary: RelaxedR1CSInstance<G2>,
  l_u_secondary: R1CSInstance<G2>,
  nifs_secondary: NIFS<G2>,
  f_W_snark_secondary: S2,

  num_steps: usize,
  program_counter: G1::Scalar,

  zn_primary: Vec<G1::Scalar>,
  zn_secondary: Vec<G2::Scalar>,

  _p: PhantomData<(C1, C2)>,
}

impl<G1, G2, C1, C2, S1, S2> CompressedSNARK<G1, G2, C1, C2, S1, S2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
  S1: RelaxedR1CSSNARKTrait<G1>,
  S2: RelaxedR1CSSNARKTrait<G2>,
{
  /// Generate the prover and verifier keys for the `CompressedSNARK` prover and verifier
  pub fn setup(
    pp: &PublicParams<G1, G2, C1, C2>,
  ) -> Result<
    (
      ProverKey<G1, G2, C1, C2, S1, S2>,
      VerifierKey<G1, G2, C1, C2, S1, S2>,
    ),
    SuperNovaError,
  > {
    let (pks_primary, vks_primary) = pp
      .circuit_shapes
      .iter()
      .map(|c| S1::setup(&pp.ck_primary, &c.r1cs_shape))
      .collect::<Result<Vec<_>, _>>()?
      .into_iter()
      .unzip();

    let (pk_secondary, vk_secondary) =
      S2::setup(&pp.ck_secondary, &pp.circuit_shape_secondary.r1cs_shape)?;

    let prover_key = ProverKey {
      pks_primary,
      pk_secondary,
      _p: PhantomData,
    };
    let verifier_key = VerifierKey {
      vks_primary,
      vk_secondary,
      _p: PhantomData,
    };

    Ok((prover_key, verifier_key))
  }

  /// create a new `CompressedSNARK` proof
  pub fn prove(
    pp: &PublicParams<G1, G2, C1, C2>,
    pk: &ProverKey<G1, G2, C1, C2, S1, S2>,
    recursive_snark: &RecursiveSNARK<G1, G2>,
  ) -> Result<Self, SuperNovaError> {
    let res_secondary = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<G1>(pp.digest()),
      &pp.circuit_shape_secondary.r1cs_shape,
      recursive_snark.r_U_secondary[0].as_ref().unwrap(),
      recursive_snark.r_W_secondary[0].as_ref().unwrap(),
      &recursive_snark.l_u_secondary,
      &recursive_snark.l_w_secondary,
    );

    let (nifs_secondary, (f_U_secondary, f_W_secondary)) = res_secondary?;

    // only produce a proof when the number of witnesses and instances match the number of circuits
    if pk.pks_primary.len() != recursive_snark.r_W_primary.len()
      || pk.pks_primary.len() != recursive_snark.r_U_primary.len()
    {
      return Err(SuperNovaError::ProofCreationError);
    }

    let r_W_snark_primary = pk
      .pks_primary
      .iter()
      .zip(&pp.circuit_shapes)
      .zip(&recursive_snark.r_W_primary)
      .zip(&recursive_snark.r_U_primary)
      .map(|(((pk, shape), r_W), r_U)| {
        let r_W = r_W
          .as_ref()
          .unwrap_or_else(|| panic!("Expected circuit witness"));
        let r_U = r_U
          .as_ref()
          .unwrap_or_else(|| panic!("Expected circuit instance"));
        S1::prove(&pp.ck_primary, pk, &shape.r1cs_shape, r_U, r_W)
      })
      .collect::<Result<Vec<S1>, _>>()?;

    let f_W_snark_secondary = S2::prove(
      &pp.ck_secondary,
      &pk.pk_secondary,
      &pp.circuit_shape_secondary.r1cs_shape,
      &f_U_secondary,
      &f_W_secondary,
    )?;

    let r_U_primary = recursive_snark
      .r_U_primary
      .iter()
      .enumerate()
      .map(|(idx, r_U)| {
        r_U
          .clone()
          .unwrap_or_else(|| RelaxedR1CSInstance::default(&pp.ck_primary, &pp[idx].r1cs_shape))
      })
      .collect::<Vec<_>>();

    let compressed_snark = CompressedSNARK {
      r_U_primary,
      r_W_snark_primary,

      r_U_secondary: recursive_snark.r_U_secondary[0].clone().unwrap(),
      l_u_secondary: recursive_snark.l_u_secondary.clone(),
      nifs_secondary,
      f_W_snark_secondary,

      num_steps: recursive_snark.i,
      program_counter: recursive_snark.program_counter,

      zn_primary: recursive_snark.zi_primary.clone(),
      zn_secondary: recursive_snark.zi_secondary.clone(),

      _p: PhantomData,
    };

    Ok(compressed_snark)
  }

  /// verify a `CompressedSNARK` proof
  pub fn verify(
    &self,
    pp: &PublicParams<G1, G2, C1, C2>,
    vk: &VerifierKey<G1, G2, C1, C2, S1, S2>,
    z0_primary: Vec<G1::Scalar>,
    z0_secondary: Vec<G2::Scalar>,
  ) -> Result<(Vec<G1::Scalar>, Vec<G2::Scalar>), SuperNovaError> {
    let last_circuit_idx = field_as_usize(self.program_counter);

    let num_field_primary_ro = 3 // params_next, i_new, program_counter_new
    + 2 * pp[last_circuit_idx].F_arity // zo, z1
    + (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // # 1 * (7 + [X0, X1]*#num_limb)

    // secondary circuit
    // NOTE: This count ensure the number of witnesses sent by the prover must bd equal to the number
    // of NIVC circuits
    let num_field_secondary_ro = 2 // params_next, i_new
    + 2 * pp.circuit_shape_secondary.F_arity // zo, z1
    + pp.circuit_shapes.len() * (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // #num_augment

    let (hash_primary, hash_secondary) = {
      let mut hasher = <G2 as Group>::RO::new(pp.ro_consts_secondary.clone(), num_field_primary_ro);

      hasher.absorb(pp.digest());
      hasher.absorb(G1::Scalar::from(self.num_steps as u64));
      hasher.absorb(self.program_counter);

      for e in z0_primary {
        hasher.absorb(e);
      }

      for e in &self.zn_primary {
        hasher.absorb(*e);
      }

      self.r_U_secondary.absorb_in_ro(&mut hasher);

      let mut hasher2 =
        <G1 as Group>::RO::new(pp.ro_consts_primary.clone(), num_field_secondary_ro);

      hasher2.absorb(scalar_as_base::<G1>(pp.digest()));
      hasher2.absorb(G2::Scalar::from(self.num_steps as u64));

      for e in z0_secondary {
        hasher2.absorb(e);
      }

      for e in &self.zn_secondary {
        hasher2.absorb(*e);
      }

      self.r_U_primary.iter().for_each(|U| {
        U.absorb_in_ro(&mut hasher2);
      });

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != self.l_u_secondary.X[0] {
      return Err(NovaError::ProofVerifyError.into());
    }

    if hash_secondary != scalar_as_base::<G2>(self.l_u_secondary.X[1]) {
      return Err(NovaError::ProofVerifyError.into());
    }

    let res_primary = self
      .r_W_snark_primary
      .iter()
      .enumerate()
      .zip(&vk.vks_primary)
      .map(|((idx, proof), vk)| {
        let U = &self.r_U_primary[idx];
        proof.verify(vk, U)
      })
      .collect::<Vec<_>>();

    let f_U_secondary = self.nifs_secondary.verify(
      &pp.ro_consts_secondary,
      &scalar_as_base::<G1>(pp.digest()),
      &self.r_U_secondary,
      &self.l_u_secondary,
    )?;

    let res_secondary = self
      .f_W_snark_secondary
      .verify(&vk.vk_secondary, &f_U_secondary);

    res_primary
      .iter()
      .map(|res| res.clone().map_err(|_| NovaError::ProofVerifyError))
      .collect::<Result<Vec<_>, _>>()?;

    res_secondary?;

    Ok((self.zn_primary.clone(), self.zn_secondary.clone()))
  }
}

// TODO: This should be factored out as described in issue #64
fn field_as_usize<F: PrimeField>(x: F) -> usize {
  u32::from_le_bytes(x.to_repr().as_ref()[0..4].try_into().unwrap()) as usize
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    provider::{
      bn256_grumpkin::{bn256, grumpkin},
      ipa_pc::EvaluationEngine,
      secp_secq::{secp256k1, secq256k1},
    },
    spartan::snark::RelaxedR1CSSNARK,
    supernova::NonUniformCircuit,
    traits::{circuit_supernova::TrivialSecondaryCircuit, evaluation::EvaluationEngineTrait},
  };

  use abomonation::Abomonation;
  use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::Field;
  use pasta_curves::{pallas, vesta};

  type EE<G> = EvaluationEngine<G>;
  type S<G, EE> = RelaxedR1CSSNARK<G, EE>;

  #[derive(Clone)]
  struct SquareCircuit<G: Group> {
    _p: PhantomData<G>,
  }

  impl<G: Group> StepCircuit<G::Scalar> for SquareCircuit<G> {
    fn arity(&self) -> usize {
      1
    }

    fn circuit_index(&self) -> usize {
      0
    }

    fn synthesize<CS: ConstraintSystem<G::Scalar>>(
      &self,
      cs: &mut CS,
      _pc: Option<&AllocatedNum<G::Scalar>>,
      z: &[AllocatedNum<G::Scalar>],
    ) -> Result<
      (
        Option<AllocatedNum<G::Scalar>>,
        Vec<AllocatedNum<G::Scalar>>,
      ),
      SynthesisError,
    > {
      let z_i = &z[0];

      let z_next = z_i.square(cs.namespace(|| "z_i^2"))?;

      let next_pc = AllocatedNum::alloc(cs.namespace(|| "next_pc"), || Ok(G::Scalar::from(1u64)))?;

      cs.enforce(
        || "next_pc = 1",
        |lc| lc + CS::one(),
        |lc| lc + next_pc.get_variable(),
        |lc| lc + CS::one(),
      );

      Ok((Some(next_pc), vec![z_next]))
    }
  }

  #[derive(Clone)]
  struct CubeCircuit<G: Group> {
    _p: PhantomData<G>,
  }

  impl<G: Group> StepCircuit<G::Scalar> for CubeCircuit<G> {
    fn arity(&self) -> usize {
      1
    }

    fn circuit_index(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<G::Scalar>>(
      &self,
      cs: &mut CS,
      _pc: Option<&AllocatedNum<G::Scalar>>,
      z: &[AllocatedNum<G::Scalar>],
    ) -> Result<
      (
        Option<AllocatedNum<G::Scalar>>,
        Vec<AllocatedNum<G::Scalar>>,
      ),
      SynthesisError,
    > {
      let z_i = &z[0];

      let z_sq = z_i.square(cs.namespace(|| "z_i^2"))?;
      let z_cu = z_sq.mul(cs.namespace(|| "z_i^3"), z_i)?;

      let next_pc = AllocatedNum::alloc(cs.namespace(|| "next_pc"), || Ok(G::Scalar::from(0u64)))?;

      cs.enforce(
        || "next_pc = 0",
        |lc| lc + CS::one(),
        |lc| lc + next_pc.get_variable(),
        |lc| lc,
      );

      Ok((Some(next_pc), vec![z_cu]))
    }
  }

  #[derive(Clone)]
  enum TestCircuit<G: Group> {
    Square(SquareCircuit<G>),
    Cube(CubeCircuit<G>),
  }

  impl<G: Group> StepCircuit<G::Scalar> for TestCircuit<G> {
    fn arity(&self) -> usize {
      1
    }

    fn circuit_index(&self) -> usize {
      match self {
        TestCircuit::Square(c) => c.circuit_index(),
        TestCircuit::Cube(c) => c.circuit_index(),
      }
    }

    fn synthesize<CS: ConstraintSystem<G::Scalar>>(
      &self,
      cs: &mut CS,
      pc: Option<&AllocatedNum<G::Scalar>>,
      z: &[AllocatedNum<G::Scalar>],
    ) -> Result<
      (
        Option<AllocatedNum<G::Scalar>>,
        Vec<AllocatedNum<G::Scalar>>,
      ),
      SynthesisError,
    > {
      match self {
        TestCircuit::Square(c) => c.synthesize(cs, pc, z),
        TestCircuit::Cube(c) => c.synthesize(cs, pc, z),
      }
    }
  }

  struct TestNIVC<G1, G2> {
    _p: PhantomData<(G1, G2)>,
  }

  impl<G1, G2> TestNIVC<G1, G2> {
    fn new() -> Self {
      TestNIVC { _p: PhantomData }
    }
  }

  impl<G1, G2> NonUniformCircuit<G1, G2, TestCircuit<G1>, TrivialSecondaryCircuit<G2::Scalar>>
    for TestNIVC<G1, G2>
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
  {
    fn initial_program_counter(&self) -> <G1 as Group>::Scalar {
      G1::Scalar::from(0u64)
    }

    fn num_circuits(&self) -> usize {
      2
    }

    fn primary_circuit(&self, circuit_index: usize) -> TestCircuit<G1> {
      match circuit_index {
        0 => TestCircuit::Square(SquareCircuit { _p: PhantomData }),
        1 => TestCircuit::Cube(CubeCircuit { _p: PhantomData }),
        _ => panic!("Invalid circuit index"),
      }
    }

    fn secondary_circuit(&self) -> TrivialSecondaryCircuit<G2::Scalar> {
      Default::default()
    }
  }

  fn test_nivc_trivial_with_compression_with<G1, G2, E1, E2>()
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
    E1: EvaluationEngineTrait<G1>,
    E2: EvaluationEngineTrait<G2>,
    <G1::Scalar as PrimeField>::Repr: Abomonation,
    <G2::Scalar as PrimeField>::Repr: Abomonation,
  {
    const NUM_STEPS: usize = 6;

    let test_nivc = TestNIVC::<G1, G2>::new();

    let pp = PublicParams::new(&test_nivc);

    let initial_pc = test_nivc.initial_program_counter();
    let mut augmented_circuit_index = field_as_usize(initial_pc);

    let z0_primary = vec![G1::Scalar::from(17u64)];
    let z0_secondary = vec![<G2 as Group>::Scalar::ZERO];

    let mut recursive_snark = RecursiveSNARK::iter_base_step(
      &pp,
      augmented_circuit_index,
      &test_nivc.primary_circuit(augmented_circuit_index),
      &test_nivc.secondary_circuit(),
      Some(initial_pc),
      augmented_circuit_index,
      2,
      &z0_primary,
      &z0_secondary,
    )
    .unwrap();

    for _ in 0..NUM_STEPS {
      let prove_res = recursive_snark.prove_step(
        &pp,
        augmented_circuit_index,
        &test_nivc.primary_circuit(augmented_circuit_index),
        &test_nivc.secondary_circuit(),
        &z0_primary,
        &z0_secondary,
      );

      let verify_res =
        recursive_snark.verify(&pp, augmented_circuit_index, &z0_primary, &z0_secondary);

      assert!(prove_res.is_ok());
      assert!(verify_res.is_ok());

      let program_counter = recursive_snark.get_program_counter();
      augmented_circuit_index = field_as_usize(program_counter);
    }

    let (prover_key, verifier_key) =
      CompressedSNARK::<_, _, _, _, S<G1, E1>, S<G2, E2>>::setup(&pp).unwrap();

    let compressed_prove_res = CompressedSNARK::prove(&pp, &prover_key, &recursive_snark);

    assert!(compressed_prove_res.is_ok());

    let compressed_snark = compressed_prove_res.unwrap();

    let compressed_verify_res =
      compressed_snark.verify(&pp, &verifier_key, z0_primary, z0_secondary);

    assert!(compressed_verify_res.is_ok());
  }

  #[test]
  fn test_nivc_trivial_with_compression() {
    test_nivc_trivial_with_compression_with::<pallas::Point, vesta::Point, EE<_>, EE<_>>();
    test_nivc_trivial_with_compression_with::<bn256::Point, grumpkin::Point, EE<_>, EE<_>>();
    test_nivc_trivial_with_compression_with::<secp256k1::Point, secq256k1::Point, EE<_>, EE<_>>();
  }

  fn test_compression_detects_circuit_num_with<G1, G2, E1, E2>()
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
    E1: EvaluationEngineTrait<G1>,
    E2: EvaluationEngineTrait<G2>,
    <G1::Scalar as PrimeField>::Repr: Abomonation,
    <G2::Scalar as PrimeField>::Repr: Abomonation,
  {
    const NUM_STEPS: usize = 6;

    let test_nivc = TestNIVC::<G1, G2>::new();

    let pp = PublicParams::new(&test_nivc);

    let initial_pc = test_nivc.initial_program_counter();
    let mut augmented_circuit_index = field_as_usize(initial_pc);

    let z0_primary = vec![G1::Scalar::from(17u64)];
    let z0_secondary = vec![G2::Scalar::ZERO];

    let mut recursive_snark = RecursiveSNARK::iter_base_step(
      &pp,
      augmented_circuit_index,
      &test_nivc.primary_circuit(augmented_circuit_index),
      &test_nivc.secondary_circuit(),
      Some(initial_pc),
      augmented_circuit_index,
      2,
      &z0_primary,
      &z0_secondary,
    )
    .unwrap();

    for _ in 0..NUM_STEPS {
      let prove_res = recursive_snark.prove_step(
        &pp,
        augmented_circuit_index,
        &test_nivc.primary_circuit(augmented_circuit_index),
        &test_nivc.secondary_circuit(),
        &z0_primary,
        &z0_secondary,
      );

      let verify_res =
        recursive_snark.verify(&pp, augmented_circuit_index, &z0_primary, &z0_secondary);

      assert!(prove_res.is_ok());
      assert!(verify_res.is_ok());

      let program_counter = recursive_snark.get_program_counter();
      augmented_circuit_index = field_as_usize(program_counter);
    }

    let (prover_key, verifier_key) =
      CompressedSNARK::<_, _, _, _, S<G1, E1>, S<G2, E2>>::setup(&pp).unwrap();

    let mut recursive_snark_truncated = recursive_snark.clone();

    recursive_snark_truncated.r_U_primary.pop();
    recursive_snark_truncated.r_W_primary.pop();

    let bad_proof = CompressedSNARK::prove(&pp, &prover_key, &recursive_snark_truncated);
    assert!(bad_proof.is_err());

    let compressed_snark = CompressedSNARK::prove(&pp, &prover_key, &recursive_snark).unwrap();

    let mut bad_compressed_snark = compressed_snark.clone();

    bad_compressed_snark.r_U_primary.pop();
    bad_compressed_snark.r_W_snark_primary.pop();

    let bad_verification =
      bad_compressed_snark.verify(&pp, &verifier_key, z0_primary, z0_secondary);
    assert!(bad_verification.is_err());
  }

  #[test]
  #[should_panic]
  fn test_compression_detects_circuit_num() {
    test_compression_detects_circuit_num_with::<pallas::Point, vesta::Point, EE<_>, EE<_>>();
    test_compression_detects_circuit_num_with::<bn256::Point, grumpkin::Point, EE<_>, EE<_>>();
    test_compression_detects_circuit_num_with::<secp256k1::Point, secq256k1::Point, EE<_>, EE<_>>();
  }
}
