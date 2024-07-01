//! This module defines the needed wrong-field NIFS prover

use std::marker::PhantomData;

use rand_core::OsRng;
use ff::Field;

use crate::{
  constants::{NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS, NUM_FE_IN_EMULATED_POINT},
  errors::NovaError,
  gadgets::scalar_as_base,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{commitment::CommitmentTrait, AbsorbInROTrait, Engine, ROConstants, ROTrait},
  CommitmentKey, CompressedCommitment,
};

use super::util::{absorb_cyclefold_r1cs, absorb_primary_commitment, absorb_primary_r1cs};

/// A SNARK that holds the proof of a step of an incremental computation of the primary circuit
/// in the CycleFold folding scheme.
/// The difference of this folding scheme from the Nova NIFS in `src/nifs.rs` is that this
#[derive(Debug)]
pub struct PrimaryNIFS<E1, E2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  pub(crate) comm_T: CompressedCommitment<E1>,
  _p: PhantomData<E2>,
}

impl<E1, E2> PrimaryNIFS<E1, E2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  /// Takes a relaxed R1CS instance-witness pair (U1, W1) and an R1CS instance-witness pair (U2, W2)
  /// and folds them into a new relaxed R1CS instance-witness pair (U, W) and a commitment to the
  /// cross term T. It also provides the challenge r used to fold the instances.
  pub fn prove(
    ck: &CommitmentKey<E1>,
    ro_consts: &ROConstants<E2>,
    pp_digest: &E1::Scalar,
    S: &R1CSShape<E1>,
    U1: &RelaxedR1CSInstance<E1>,
    W1: &RelaxedR1CSWitness<E1>,
    U2: &R1CSInstance<E1>,
    W2: &R1CSWitness<E1>,
  ) -> Result<
    (
      Self,
      (RelaxedR1CSInstance<E1>, RelaxedR1CSWitness<E1>),
      E1::Scalar,
    ),
    NovaError,
  > {
    let arity = U1.X.len();

    if arity != U2.X.len() {
      return Err(NovaError::InvalidInputLength);
    }

    let mut ro = E2::RO::new(
      ro_consts.clone(),
      1 + NUM_FE_IN_EMULATED_POINT + arity + NUM_FE_IN_EMULATED_POINT, // pp_digest + u.W + u.X + T
    );

    ro.absorb(*pp_digest);

    absorb_primary_r1cs::<E1, E2>(U2, &mut ro);

    let r_T = E1::Scalar::random(&mut OsRng);
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2, &r_T)?;

    absorb_primary_commitment::<E1, E2>(&comm_T, &mut ro);

    let r = scalar_as_base::<E2>(ro.squeeze(NUM_CHALLENGE_BITS));

    let U = U1.fold(U2, &comm_T, &r);

    let W = W1.fold(W2, &T, &r_T, &r)?;

    Ok((
      Self {
        comm_T: comm_T.compress(),
        _p: PhantomData,
      },
      (U, W),
      r,
    ))
  }
}

/// A SNARK that holds the proof of a step of an incremental computation of the CycleFold circuit
/// The difference of this folding scheme from the Nova NIFS in `src/nifs.rs` is that this folding
/// prover and verifier must fold in the `RelaxedR1CSInstance` accumulator because the optimization
/// in the
#[derive(Debug)]
pub struct CycleFoldNIFS<E: Engine> {
  pub(crate) comm_T: CompressedCommitment<E>,
}

impl<E: Engine> CycleFoldNIFS<E> {
  /// Folds an R1CS instance/witness pair (U2, W2) into a relaxed R1CS instance/witness (U1, W1)
  /// returning the new folded accumulator and a commitment to the cross-term.
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &ROConstants<E>,
    pp_digest: &E::Scalar,
    S: &R1CSShape<E>,
    U1: &RelaxedR1CSInstance<E>,
    W1: &RelaxedR1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) -> Result<(Self, (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>)), NovaError> {
    // Check `U1` and `U2` have the same arity
    if U2.X.len() != NIO_CYCLE_FOLD || U1.X.len() != NIO_CYCLE_FOLD {
      return Err(NovaError::InvalidInputLength);
    }

    // initialize a new RO
    let mut ro = E::RO::new(
      ro_consts.clone(),
      46, // 1 + (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // digest + (U) + (u) + T
    );

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<E>(*pp_digest));

    // append U1 to the transcript.
    // NOTE: this must be here because the IO for `U2` does not have the data of the hash of `U1`
    U1.absorb_in_ro(&mut ro);

    // append U2 to transcript
    absorb_cyclefold_r1cs(U2, &mut ro);

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
    ))
  }
}
