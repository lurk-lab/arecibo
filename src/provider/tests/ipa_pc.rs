#[cfg(test)]
mod test {
  use crate::provider::ipa_pc::EvaluationEngine;
  use crate::provider::tests::solidity_compatibility_utils::{
    compressed_commitment_to_json, ec_points_to_json, field_elements_to_json,
    // generate_pcs_solidity_unit_test_data,
  };

  use crate::provider::GrumpkinEngine;
  use group::Curve;

  use crate::provider::pedersen::{CommitmentKey, CommitmentKeyExtTrait};
  use handlebars::Handlebars;
  use serde_json::json;
  use serde_json::{Map, Value};

  static IPA_COMPATIBILITY_UNIT_TESTING_TEMPLATE: &str = "
// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.16;
import \"@std/Test.sol\";
import \"src/blocks/grumpkin/Grumpkin.sol\";
import \"src/blocks/EqPolynomial.sol\";
import \"src/Utilities.sol\";
import \"src/blocks/IpaPcs.sol\";

contract IpaTest is Test {
function composeIpaInput() public pure returns (InnerProductArgument.IpaInputGrumpkin memory) {
Grumpkin.GrumpkinAffinePoint[] memory ck_v = new Grumpkin.GrumpkinAffinePoint[]({{ len ck_v }});
{{ #each ck_v }} ck_v[{{ i }}]=Grumpkin.GrumpkinAffinePoint({{ x }}, {{y}});\n {{ /each }}

Grumpkin.GrumpkinAffinePoint[] memory ck_s = new Grumpkin.GrumpkinAffinePoint[]({{ len ck_s }});
{{ #each ck_s }} ck_s[{{ i }}]=Grumpkin.GrumpkinAffinePoint({{ x }}, {{y}});\n {{ /each }}

uint256[] memory point = new uint256[]({{ len point }});
{{ #each point }} point[{{ i }}]={{ val }};\n {{ /each }}

uint256[] memory L_vec = new uint256[]({{ len L_vec }});
{{ #each L_vec }} L_vec[{{ i }}]={{ compressed }};\n {{ /each }}

uint256[] memory R_vec = new uint256[]({{ len R_vec }});
{{ #each R_vec }} R_vec[{{ i }}]={{ compressed }};\n {{ /each }}

uint256 a_hat = {{ a_hat }};

// InnerProductInstance
Grumpkin.GrumpkinAffinePoint memory commitment = Grumpkin.GrumpkinAffinePoint({{ commitment_x }}, {{ commitment_y }});

uint256 eval = {{ eval }};

return InnerProductArgument.IpaInputGrumpkin(ck_v, ck_s, point, L_vec, R_vec, commitment, eval, a_hat);
}

function testIpaGrumpkinVerification_{{ num_vars }}_Variables() public {
InnerProductArgument.IpaInputGrumpkin memory input = composeIpaInput();
assertTrue(InnerProductArgument.verifyGrumpkin(input, getTranscript()));
}

function getTranscript() public pure returns (KeccakTranscriptLib.KeccakTranscript memory) {
// b\"TestEval\" in Rust
uint8[] memory label = new uint8[](8);
label[0] = 0x54;
label[1] = 0x65;
label[2] = 0x73;
label[3] = 0x74;
label[4] = 0x45;
label[5] = 0x76;
label[6] = 0x61;
label[7] = 0x6c;

KeccakTranscriptLib.KeccakTranscript memory keccak_transcript = KeccakTranscriptLib.instantiate(label);
return keccak_transcript;
}
}
";

  // To generate Solidity unit-test:
  // cargo test test_solidity_compatibility_ipa --release -- --ignored --nocapture > ipa.t.sol
  // #[test]
  // #[ignore]
  // fn test_solidity_compatibility_ipa() {
  //   let num_vars = 2;

  //   // Secondary part of verification is IPA over Grumpkin
  //   let (commitment, point, eval, proof, vk) =
  //     generate_pcs_solidity_unit_test_data::<_, EvaluationEngine<GrumpkinEngine>>(num_vars);

  //   let num_vars_string = format!("{}", num_vars);
  //   let eval_string = format!("{:?}", eval);
  //   let commitment_x_string = format!("{:?}", commitment.comm.to_affine().x);
  //   let commitment_y_string = format!("{:?}", commitment.comm.to_affine().y);
  //   let proof_a_hat_string = format!("{:?}", proof.a_hat);

  //   let r_vec = CommitmentKey::<GrumpkinEngine>::reinterpret_commitments_as_ck(&proof.R_vec)
  //     .expect("can't reinterpred R_vec");
  //   let l_vec = CommitmentKey::<GrumpkinEngine>::reinterpret_commitments_as_ck(&proof.L_vec)
  //     .expect("can't reinterpred L_vec");

  //   let r_vec_array = compressed_commitment_to_json::<GrumpkinEngine>(&r_vec.ck);
  //   let l_vec_array = compressed_commitment_to_json::<GrumpkinEngine>(&l_vec.ck);
  //   let point_array = field_elements_to_json::<GrumpkinEngine>(&point);
  //   let ckv_array = ec_points_to_json::<GrumpkinEngine>(&vk.ck_v.ck);
  //   let cks_array = ec_points_to_json::<GrumpkinEngine>(&vk.ck_s.ck);

  //   let mut map = Map::new();
  //   map.insert("num_vars".to_string(), Value::String(num_vars_string));
  //   map.insert("eval".to_string(), Value::String(eval_string));
  //   map.insert(
  //     "commitment_x".to_string(),
  //     Value::String(commitment_x_string),
  //   );
  //   map.insert(
  //     "commitment_y".to_string(),
  //     Value::String(commitment_y_string),
  //   );
  //   map.insert("R_vec".to_string(), Value::Array(r_vec_array));
  //   map.insert("L_vec".to_string(), Value::Array(l_vec_array));
  //   map.insert("a_hat".to_string(), Value::String(proof_a_hat_string));
  //   map.insert("point".to_string(), Value::Array(point_array));
  //   map.insert("ck_v".to_string(), Value::Array(ckv_array));
  //   map.insert("ck_s".to_string(), Value::Array(cks_array));

  //   let mut reg = Handlebars::new();
  //   reg
  //     .register_template_string("ipa.t.sol", IPA_COMPATIBILITY_UNIT_TESTING_TEMPLATE)
  //     .expect("can't register template");

  //   let solidity_unit_test_source = reg.render("ipa.t.sol", &json!(map)).expect("can't render");
  //   println!("{}", solidity_unit_test_source);
  // }
}
