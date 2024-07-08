//! This module implements various gadgets necessary for Nova and applications built with Nova.
mod ecc;
pub mod lookup;
pub(crate) use ecc::AllocatedPoint;

mod nonnative;
pub(crate) use nonnative::{bignat::nat_to_limbs, bignat::BigNat, util::f_to_nat, util::Num};

mod r1cs;
pub(crate) use r1cs::{
  conditionally_select_alloc_relaxed_r1cs, conditionally_select_vec_allocated_relaxed_r1cs_instance,
};
pub(crate) use r1cs::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance};

mod utils;
#[cfg(test)]
pub(crate) use utils::alloc_one;
pub(crate) use utils::{
  alloc_bignat_constant, alloc_num_equals, alloc_scalar_as_base, alloc_zero,
  conditionally_select_allocated_bit, conditionally_select_bignat, le_bits_to_num, scalar_as_base,
};
