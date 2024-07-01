// use criterion::*;

// use std::time::Duration;

// mod common;
// use common::supernova::targets::{
//   bench_two_augmented_circuit_compressed_snark_with_computational_commitments,
//   bench_two_augmented_circuit_recursive_snark,
// };

// // To run these benchmarks, first download `criterion` with `cargo install cargo-criterion`.
// // Then `cargo criterion --bench recursive-snark-supernova`. The results are located in `target/criterion/data/<name-of-benchmark>`.
// // For flamegraphs, run `cargo criterion --bench recursive-snark-supernova --features flamegraph -- --profile-time <secs>`.
// // The results are located in `target/criterion/profile/<name-of-benchmark>`.
// cfg_if::cfg_if! {
//   if #[cfg(feature = "flamegraph")] {
//     criterion_group! {
//       name = supernova_ci;
//       config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
//       targets = bench_two_augmented_circuit_recursive_snark, bench_two_augmented_circuit_compressed_snark_with_computational_commitments
//     }
//   } else {
//     criterion_group! {
//       name = supernova_ci;
//       config = Criterion::default().warm_up_time(Duration::from_millis(3000));
//       targets = bench_two_augmented_circuit_recursive_snark, bench_two_augmented_circuit_compressed_snark_with_computational_commitments
//     }
//   }
// }

// criterion_main!(supernova_ci);
