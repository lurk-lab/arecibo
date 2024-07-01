// // Code is considered dead unless used in all benchmark targets
// #![allow(dead_code)]
// use criterion::Criterion;

// use crate::common::supernova::{bench::run_bench, SnarkType, S1, S2, SS1, SS2};

// // Recursive Supernova SNARK benchmarks
// pub fn bench_one_augmented_circuit_recursive_snark(c: &mut Criterion) {
//   run_bench::<S1, S2>(c, "RecursiveSNARK-NIVC-1", 1, SnarkType::Recursive)
// }

// pub fn bench_two_augmented_circuit_recursive_snark(c: &mut Criterion) {
//   run_bench::<S1, S2>(c, "RecursiveSNARK-NIVC-2", 2, SnarkType::Recursive)
// }

// // Compressed Supernova SNARK benchmarks
// pub fn bench_one_augmented_circuit_compressed_snark(c: &mut Criterion) {
//   run_bench::<S1, S2>(c, "CompressedSNARK-NIVC-1", 1, SnarkType::Compressed)
// }

// pub fn bench_two_augmented_circuit_compressed_snark(c: &mut Criterion) {
//   run_bench::<S1, S2>(c, "CompressedSNARK-NIVC-2", 2, SnarkType::Compressed)
// }

// pub fn bench_two_augmented_circuit_compressed_snark_with_computational_commitments(
//   c: &mut Criterion,
// ) {
//   run_bench::<SS1, SS2>(
//     c,
//     "CompressedSNARK-NIVC-Commitments-2",
//     2,
//     SnarkType::Compressed,
//   )
// }
