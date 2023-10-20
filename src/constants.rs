//! Global Nova constants

pub(crate) const NUM_CHALLENGE_BITS: usize = 128;
pub(crate) const BN_LIMB_WIDTH: usize = 64;
pub(crate) const BN_N_LIMBS: usize = 4;
pub(crate) const NUM_FE_WITHOUT_IO_FOR_CRHF: usize = 17;
pub(crate) const NUM_FE_FOR_RO: usize = 24;

/// Bit size of Nova field element hashes
pub const NUM_HASH_BITS: usize = 250;

/// The threshold of elements we check when extending a `CommitmentKey` to avoid regressions due to code changes.
/// There are no security guarantees here
pub(crate) const CK_CHECKING_THRESHOLD: usize = 1;
