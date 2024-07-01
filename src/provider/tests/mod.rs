mod ipa_pc;

#[cfg(test)]
pub mod solidity_compatibility_utils {
  use crate::provider::traits::DlogGroup;
  use crate::spartan::polys::multilinear::MultilinearPolynomial;
  use crate::traits::{
    commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait, Engine,
  };
  use group::prime::PrimeCurve;
  use group::prime::PrimeCurveAffine;
  use group::GroupEncoding;
  use rand::rngs::StdRng;
  use serde::Serialize;
use serde_json::{Map, Value};
  use std::sync::Arc;

  // pub(crate) fn generate_pcs_solidity_unit_test_data<E: Engine + Serialize, EE: EvaluationEngineTrait<E>>(
  //   num_vars: usize,
  // ) -> (
  //   <E::CE as CommitmentEngineTrait<E>>::Commitment,
  //   Vec<E::Scalar>,
  //   E::Scalar,
  //   EE::EvaluationArgument,
  //   EE::VerifierKey,
  // ) {
  //   use rand_core::SeedableRng;

    // let mut rng = StdRng::seed_from_u64(num_vars as u64);

  //   let (poly, point, eval) =
  //     crate::provider::util::test_utils::random_poly_with_eval::<E, StdRng>(num_vars, &mut rng);

  //   // Mock commitment key.
  //   let ck = E::CE::setup(b"test", 1 << num_vars);
  //   let ck_arc = Arc::new(ck.clone());
  //   // Commits to the provided vector using the provided generators.
  //   let commitment = E::CE::commit(&ck_arc, poly.evaluations());

  //   let (proof, vk) = prove_verify_solidity::<E, EE>(ck_arc, &commitment, &poly, &point, &eval);

  //   (commitment, point, eval, proof, vk)
  // }

  // fn prove_verify_solidity<E: Engine + Serialize, EE: EvaluationEngineTrait<E>>(
  //   ck: Arc<<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey>,
  //   commitment: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
  //   poly: &MultilinearPolynomial<<E as Engine>::Scalar>,
  //   point: &[<E as Engine>::Scalar],
  //   eval: &<E as Engine>::Scalar,
  // ) -> (EE::EvaluationArgument, EE::VerifierKey) {
  //   use crate::traits::TranscriptEngineTrait;

  //   // Generate Prover and verifier key for given commitment key.
  //   let ock = ck.clone();
  //   let (prover_key, verifier_key) = EE::setup(ck);

  //   // Generate proof.
  //   let mut prover_transcript = E::TE::new(b"TestEval");
  //   let proof: EE::EvaluationArgument = EE::prove(
  //     &*ock,
  //     &prover_key,
  //     &mut prover_transcript,
  //     commitment,
  //     poly.evaluations(),
  //     point,
  //     eval,
  //   )
  //   .unwrap();
  //   let pcp = prover_transcript.squeeze(b"c").unwrap();

  //   // Verify proof.
  //   let mut verifier_transcript = E::TE::new(b"TestEval");
  //   EE::verify(
  //     &verifier_key,
  //     &mut verifier_transcript,
  //     commitment,
  //     point,
  //     eval,
  //     &proof,
  //   )
  //   .unwrap();
  //   let pcv = verifier_transcript.squeeze(b"c").unwrap();

  //   // Check if the prover transcript and verifier transcript are kept in the same state.
  //   assert_eq!(pcp, pcv);

  //   (proof, verifier_key)
  // }

  pub(crate) fn field_elements_to_json<E: Engine>(field_elements: &[E::Scalar]) -> Vec<Value> {
    let mut value_vector = vec![];
    field_elements.iter().enumerate().for_each(|(i, fe)| {
      let mut value = Map::new();
      value.insert("i".to_string(), Value::String(i.to_string()));
      value.insert("val".to_string(), Value::String(format!("{:?}", fe)));
      value_vector.push(Value::Object(value));
    });
    value_vector
  }

  pub(crate) fn ec_points_to_json<E>(ec_points: &[<E::GE as PrimeCurve>::Affine]) -> Vec<Value>
  where
    E: Engine,
    E::GE: DlogGroup<ScalarExt = E::Scalar>,
  {
    let mut value_vector = vec![];
    ec_points.iter().enumerate().for_each(|(i, ec_point)| {
      let mut value = Map::new();
      let coordinates_info = ec_point.to_curve().to_coordinates();
      let not_infinity = !coordinates_info.2;
      assert!(not_infinity);
      value.insert("i".to_string(), Value::String(i.to_string()));
      value.insert(
        "x".to_string(),
        Value::String(format!("{:?}", coordinates_info.0)),
      );
      value.insert(
        "y".to_string(),
        Value::String(format!("{:?}", coordinates_info.1)),
      );
      value_vector.push(Value::Object(value));
    });
    value_vector
  }

  pub(crate) fn compressed_commitment_to_json<E>(
    ec_points: &[<E::GE as PrimeCurve>::Affine],
  ) -> Vec<Value>
  where
    E: Engine,
    E::GE: DlogGroup<ScalarExt = E::Scalar>,
  {
    let mut value_vector = vec![];
    ec_points.iter().enumerate().for_each(|(i, ec_point)| {
      let mut value = Map::new();
      let compressed_commitment_info = ec_point.to_curve().to_bytes();
      let mut data = compressed_commitment_info.as_ref().to_vec();
      data.reverse();

      value.insert("i".to_string(), Value::String(i.to_string()));
      value.insert(
        "compressed".to_string(),
        Value::String(format!("0x{}", hex::encode(data))),
      );
      value_vector.push(Value::Object(value));
    });
    value_vector
  }
}
