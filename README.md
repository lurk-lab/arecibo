# Nova: High-speed recursive arguments from folding schemes

> [!NOTE]
> This repository is a fork of the original hosted at [https://github.com/microsoft/nova](https://github.com/microsoft/nova). It's an incubator for experimenting with more advanced variants of the original software and working out the kinks in them.

> [!IMPORTANT]
> This fork is always maintained as up-to-date as the original repository. Occasionally, this repository backports its own contributions to the original, after an incubation time. Note that until back-ported, the changes in the present repository have not undergone the same level of review.


Nova is a high-speed recursive SNARK (a SNARK is type cryptographic proof system that enables a prover to prove a mathematical statement to a verifier with a short proof and succinct verification, and a recursive SNARK enables producing proofs that prove statements about prior proofs). 

More precisely, Nova achieves [incrementally verifiable computation (IVC)](https://iacr.org/archive/tcc2008/49480001/49480001.pdf), a powerful cryptographic primitive that allows a prover to produce a proof of correct execution of a "long running" sequential computations in an incremental fashion. For example, IVC enables the following: The prover takes as input a proof $\pi_i$ proving the first $i$ steps of its computation and then update it to produce a proof $\pi_{i+1}$ proving the correct execution of the first $i + 1$ steps. Crucially, the prover's work to update the proof does not depend on the number of steps executed thus far, and the verifier's work to verify a proof does not grow with the number of steps in the computation. IVC schemes including Nova have a wide variety of applications such as Rollups, verifiable delay functions (VDFs), succinct blockchains, incrementally verifiable versions of [verifiable state machines](https://eprint.iacr.org/2020/758.pdf), and, more generally, proofs of (virtual) machine executions (e.g., EVM, RISC-V). 

A distinctive aspect of Nova is that it is the simplest recursive proof system in the literature, yet it provides the fastest prover. Furthermore, it achieves the smallest verifier circuit (a key metric to minimize in this context): the circuit is constant-sized and its size is about 10,000 multiplication gates. Nova is constructed from a simple primitive called a *folding scheme*, a cryptographic primitive that reduces the task of checking two NP statements into the task of checking a single NP statement. 

## Details of the library
This repository provides `nova-snark,` a Rust library implementation of Nova over a cycle of elliptic curves. Our code supports three curve cycles: (1) Pallas/Vesta, (2) BN254/Grumpkin, and (3) secp/secq. 

At its core, Nova relies on a commitment scheme for vectors. Compressing IVC proofs using Spartan relies on interpreting commitments to vectors as commitments to multilinear polynomials and prove evaluations of committed polynomials. Our code implements three commitment schemes and evaluation arguments: 
1. Pedersen commitments with IPA-based evaluation argument (supported on all three curve cycles), and
2. HyperKZG commitments and evaluation argument (supported on curves with pairings e.g., BN254).
3. KZG commitments with a [Zeromorph](https://eprint.iacr.org/2023/917) evaluation argument (supported on curves equipped with a pairing).
    
For more details on using  HyperKZG / Zeromorph, please see the test `test_ivc_nontrivial_with_compression`. The HyperKZG instantiation requires a universal trusted setup (the so-called "powers of tau"). In the `setup` method in `src/provider/hyperkzg.rs`, one can load group elements produced in an existing KZG trusted setup (that was created for other proof systems based on univariate polynomials such as Plonk or variants), but the library does not currently do so (please see [this](https://github.com/microsoft/Nova/issues/270) issue). 

We also implement a SNARK, based on [Spartan](https://eprint.iacr.org/2019/550.pdf), to compress IVC proofs produced by Nova. There are two variants, one that does *not* use any preprocessing and another that uses preprocessing of circuits to ensure that the verifier's run time does not depend on the size of the step circuit.

> [!NOTE]
> This library features an implementation of Zeromorph, exclusively available here until the related changes are integrated with the official Nova repository via pull request [#301](https://github.com/microsoft/Nova/pull/301). 
>
> Additionally, we've enhanced Nova to support Supernova, offering a variant that is currently unique to this repository. This advanced capability will remain exclusive here until pull request [#283](https://github.com/microsoft/Nova/pull/283) is merged into the official Nova codebase.
>
> Our implementation of HyperKZG incorporates notable performance enhancements inspired by Shplonk (BDFG20), as detailed in [this paper](https://eprint.iacr.org/2020/081). These improvements are specifically designed to enhance efficiency and speed.

## Supported front-ends
A front-end is a tool to take a high-level program and turn it into an intermediate representation (e.g., a circuit) that can be used to prove executions of the program on concrete inputs. There are three supported ways to write high-level programs in a form that can be proven with Nova.

1. bellpepper: The native APIs of Nova accept circuits expressed with bellpepper, a Rust toolkit to express circuits. See [minroot.rs](https://github.com/microsoft/Nova/blob/main/examples/minroot.rs) or [sha256.rs](https://github.com/microsoft/Nova/blob/main/benches/sha256.rs) for examples.

2. Circom: A DSL and a compiler to transform high-level program expressed in its language into a circuit. There exist middleware to turn output of circom into a form suitable for proving with Nova. See [Nova Scotia](https://github.com/nalinbhardwaj/Nova-Scotia) and [Circom Scotia](https://github.com/argumentcomputer/circom-scotia). In the future, we will add examples in the Nova repository to use these tools with Nova.

3. [Lurk](https://github.com/argumentcomputer/lurk-rs): A Lisp dialect and a universal circuit to execute programs expressed in Lurk. The universal circuit can be proven with Nova.

In the future, we plan to support [Noir](https://noir-lang.org/), a Rust-like DSL and a compiler to transform those programs into an IR. See [this](https://github.com/microsoft/Nova/issues/275) GitHub issue for details.

## Tests and examples
By default, we enable the `asm` feature of an underlying library (which boosts performance by up to 50\%). If the library fails to build or run, one can pass `--no-default-features` to `cargo` commands noted below.

To run tests (we recommend the release mode to drastically shorten run times):
```text
cargo test --release
```

To run an example:
```text
cargo run --release --example minroot
```

## Specs and Documentation

- [SuperNova](./src/supernova/Readme.md)

## References
The following paper, which appeared at CRYPTO 2022, provides details of the Nova proof system and a proof of security:

[Nova: Recursive Zero-Knowledge Arguments from Folding Schemes](https://eprint.iacr.org/2021/370) \
Abhiram Kothapalli, Srinath Setty, and Ioanna Tzialla \
CRYPTO 2022

For efficiency, our implementation of the Nova proof system is instantiated over a cycle of elliptic curves. The following paper specifies that instantiation and provides a proof of security:

[Revisiting the Nova Proof System on a Cycle of Curves](https://eprint.iacr.org/2023/969) \
Wilson Nguyen, Dan Boneh, and Srinath Setty \
IACR ePrint 2023/969

This repository implements Supernova, published in :
[SuperNova: Proving universal machine executions without universal circuits](https://eprint.iacr.org/2022/1758) \
Abhiram Kothapalli, and Srinath Setty
IACR ePrint 2022/1758

## Acknowledgments
See the contributors list [here](https://github.com/argumentcomputer/arecibo/graphs/contributors)
