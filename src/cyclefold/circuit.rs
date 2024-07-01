//! This module defines Cyclefold circuit

use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  ConstraintSystem, SynthesisError,
};
use ff::Field;
use neptune::{circuit2::poseidon_hash_allocated, poseidon::PoseidonConstants};

use crate::{
  constants::NUM_CHALLENGE_BITS,
  gadgets::{alloc_zero, le_bits_to_num, AllocatedPoint},
  traits::{commitment::CommitmentTrait, Engine},
  Commitment,
};
use bellpepper::gadgets::boolean_utils::conditionally_select;

/// A structure containing the CycleFold circuit inputs and implementing the synthesize function
pub struct CycleFoldCircuit<E: Engine> {
  commit_1: Option<Commitment<E>>,
  commit_2: Option<Commitment<E>>,
  scalar: Option<[bool; NUM_CHALLENGE_BITS]>,
  poseidon_constants: PoseidonConstants<E::Base, generic_array::typenum::U2>,
}

impl<E: Engine> Default for CycleFoldCircuit<E> {
  fn default() -> Self {
    let poseidon_constants = PoseidonConstants::new();
    Self {
      commit_1: None,
      commit_2: None,
      scalar: None,
      poseidon_constants,
    }
  }
}
impl<E: Engine> CycleFoldCircuit<E> {
  /// Create a new CycleFold circuit with the given inputs
  pub fn new(
    commit_1: Option<Commitment<E>>,
    commit_2: Option<Commitment<E>>,
    scalar: Option<[bool; NUM_CHALLENGE_BITS]>,
  ) -> Self {
    let poseidon_constants = PoseidonConstants::new();
    Self {
      commit_1,
      commit_2,
      scalar,
      poseidon_constants,
    }
  }

  fn alloc_witness<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<
    (
      AllocatedPoint<E::GE>, // commit_1
      AllocatedPoint<E::GE>, // commit_2
      Vec<AllocatedBit>,     // scalar
    ),
    SynthesisError,
  > {
    let commit_1 = AllocatedPoint::alloc(
      cs.namespace(|| "allocate C_1"),
      self.commit_1.map(|C_1| C_1.to_coordinates()),
    )?;
    commit_1.check_on_curve(cs.namespace(|| "commit_1 on curve"))?;

    let commit_2 = AllocatedPoint::alloc(
      cs.namespace(|| "allocate C_2"),
      self.commit_2.map(|C_2| C_2.to_coordinates()),
    )?;
    commit_2.check_on_curve(cs.namespace(|| "commit_2 on curve"))?;

    let scalar: Vec<AllocatedBit> = self
      .scalar
      .unwrap_or([false; NUM_CHALLENGE_BITS])
      .into_iter()
      .enumerate()
      .map(|(idx, bit)| {
        AllocatedBit::alloc(cs.namespace(|| format!("scalar bit {idx}")), Some(bit))
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok((commit_1, commit_2, scalar))
  }

  /// Synthesize the CycleFold circuit
  pub fn synthesize<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<(), SynthesisError> {
    let (C_1, C_2, r) = self.alloc_witness(cs.namespace(|| "allocate circuit witness"))?;

    // Calculate C_final
    let r_C_2 = C_2.scalar_mul(cs.namespace(|| "r * C_2"), &r)?;

    let C_final = C_1.add(cs.namespace(|| "C_1 + r * C_2"), &r_C_2)?;

    self.inputize_point(&C_1, cs.namespace(|| "inputize C_1"))?;
    self.inputize_point(&C_2, cs.namespace(|| "inputize C_2"))?;
    self.inputize_point(&C_final, cs.namespace(|| "inputize C_final"))?;

    let scalar = le_bits_to_num(cs.namespace(|| "get scalar"), &r)?;

    scalar.inputize(cs.namespace(|| "scalar"))?;

    Ok(())
  }

  // Represent the point in the public IO as its 2-ary Poseidon hash
  fn inputize_point<CS>(
    &self,
    point: &AllocatedPoint<E::GE>,
    mut cs: CS,
  ) -> Result<(), SynthesisError>
  where
    E: Engine,
    CS: ConstraintSystem<E::Base>,
  {
    let (x, y, is_infinity) = point.get_coordinates();
    let preimage = vec![x.clone(), y.clone()];
    let val = poseidon_hash_allocated(
      cs.namespace(|| "hash point"),
      preimage,
      &self.poseidon_constants,
    )?;

    let zero = alloc_zero(cs.namespace(|| "zero"));

    let is_infinity_bit = AllocatedBit::alloc(
      cs.namespace(|| "is_infinity"),
      Some(is_infinity.get_value().unwrap_or(E::Base::ONE) == E::Base::ONE),
    )?;

    cs.enforce(
      || "infinity_bit matches",
      |lc| lc,
      |lc| lc,
      |lc| lc + is_infinity_bit.get_variable() - is_infinity.get_variable(),
    );

    // Output 0 when it is the point at infinity
    let output = conditionally_select(
      cs.namespace(|| "select output"),
      &zero,
      &val,
      &Boolean::from(is_infinity_bit),
    )?;

    output.inputize(cs.namespace(|| "inputize hash"))?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use expect_test::{expect, Expect};
  use ff::{Field, PrimeField, PrimeFieldBits};
  use neptune::Poseidon;
  use rand_core::OsRng;

  use crate::{
    bellpepper::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
    },
    constants::NIO_CYCLE_FOLD,
    gadgets::scalar_as_base,
    provider::{PallasEngine, Secp256k1Engine},
    traits::{commitment::CommitmentEngineTrait, snark::default_ck_hint, CurveCycleEquipped, Dual},
  };

  use super::*;

  fn test_cyclefold_circuit_size_with<E1>(expected_constraints: &Expect, expected_vars: &Expect)
  where
    E1: CurveCycleEquipped,
  {
    // Instantiate the circuit with trivial inputs
    let circuit: CycleFoldCircuit<Dual<E1>> = CycleFoldCircuit::default();

    // Synthesize the R1CS shape
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit.synthesize(cs.namespace(|| "synthesizing shape"));

    // Extract the number of constraints and variables
    let num_constraints = cs.num_constraints();
    let num_variables = cs.num_aux();
    let num_io = cs.num_inputs();

    // Check the number of constraints and variables match the expected values
    expected_constraints.assert_eq(&num_constraints.to_string());
    expected_vars.assert_eq(&num_variables.to_string());
    assert_eq!(num_io, NIO_CYCLE_FOLD + 1); // includes 1
  }

  #[test]
  fn test_cyclefold_circuit_size() {
    test_cyclefold_circuit_size_with::<PallasEngine>(&expect!("2090"), &expect!("2081"));
    // test_cyclefold_circuit_size_with::<Bn256EngineKZG>(&expect!("2090"), &expect!("2081"));
    test_cyclefold_circuit_size_with::<Secp256k1Engine>(&expect!("2090"), &expect!("2081"));
  }

  fn test_cyclefold_circuit_sat_with<E: CurveCycleEquipped>() {
    let rng = OsRng;

    let ck = <<Dual<E> as Engine>::CE as CommitmentEngineTrait<Dual<E>>>::setup(b"test", 5);

    // Generate random vectors to commit to
    let v1 = (0..5)
      .map(|_| <<Dual<E> as Engine>::Scalar as Field>::random(rng))
      .collect::<Vec<_>>();
    let v2 = (0..5)
      .map(|_| <<Dual<E> as Engine>::Scalar as Field>::random(rng))
      .collect::<Vec<_>>();

    // produce a random scalar
    let r = <Dual<E> as Engine>::Scalar::random(&mut OsRng);

    // Calculate the random commitments
    let C_1 = <<Dual<E> as Engine>::CE as CommitmentEngineTrait<Dual<E>>>::commit(&ck, &v1, &r);
    let C_2 = <<Dual<E> as Engine>::CE as CommitmentEngineTrait<Dual<E>>>::commit(&ck, &v2, &r);

    // Generate a random scalar
    let val: u128 = rand::random();
    let r = <<Dual<E> as Engine>::Scalar as PrimeField>::from_u128(val);
    let r_bits = r
      .to_le_bits()
      .into_iter()
      .take(128)
      .collect::<Vec<_>>()
      .try_into()
      .unwrap();

    let circuit: CycleFoldCircuit<Dual<E>> =
      CycleFoldCircuit::new(Some(C_1), Some(C_2), Some(r_bits));

    // Calculate the result out of circuit
    let native_result = C_1 + C_2 * r;

    // Generate the R1CS shape and commitment key
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(cs.namespace(|| "synthesizing shape"));
    let (shape, ck) = cs.r1cs_shape_and_key(&*default_ck_hint());

    // Synthesize the R1CS circuit on the random inputs
    let mut cs = SatisfyingAssignment::<E>::new();
    circuit
      .synthesize(cs.namespace(|| "synthesizing witness"))
      .unwrap();

    let (instance, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let X = &instance.X;

    // Helper functio to calculate the hash
    let compute_hash = |P: Commitment<Dual<E>>| -> E::Scalar {
      let (x, y, is_infinity) = P.to_coordinates();
      if is_infinity {
        return E::Scalar::ZERO;
      }

      let mut hasher = Poseidon::new_with_preimage(&[x, y], &circuit.poseidon_constants);

      hasher.hash()
    };

    // Check the circuit calculates the right thing
    let hash_1 = compute_hash(C_1);
    assert_eq!(hash_1, X[0]);
    let hash_2 = compute_hash(C_2);
    assert_eq!(hash_2, X[1]);
    let hash_res = compute_hash(native_result);
    assert_eq!(hash_res, X[2]);
    assert_eq!(r, scalar_as_base::<E>(X[3]));

    // Check the R1CS equation is satisfied
    shape.is_sat(&ck, &instance, &witness).unwrap();
  }

  #[test]
  fn test_cyclefold_circuit_sat() {
    test_cyclefold_circuit_sat_with::<PallasEngine>();
    // test_cyclefold_circuit_sat_with::<Bn256EngineKZG>();
    test_cyclefold_circuit_sat_with::<Secp256k1Engine>();
  }
}
