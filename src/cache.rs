//! This module provides caching utils

use std::{
  fs::{create_dir_all, remove_dir_all, File, OpenOptions},
  io::{self, BufWriter, Error, ErrorKind, Seek},
  ops::Deref,
  path::PathBuf,
  sync::{Arc, Mutex},
};

use abomonation::{encode, Abomonation};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use ff::PrimeField;
use memmap2::MmapMut;
use once_cell::sync::Lazy;

use crate::{
  r1cs::commitment_key_size,
  supernova::{AuxParams, CircuitDigests, NonUniformCircuit, PublicParams},
  traits::{
    circuit_supernova::StepCircuit,
    commitment::{CommitmentEngineTrait, CommitmentKeyTrait},
    Group,
  },
  CircuitShape, CommitmentKey,
};

/// Global registry for public parameters and commitment keys that are
/// cached onto the users local file system
pub static ARECIBO_CACHE: Lazy<Cache> = Lazy::new(init_cache);

/// If the env var `ARECIBO_CACHE` exists, then use that value.
/// Otherwise, set `ARECIBO_CACHE` to be `$HOME/.arecibo/`.
fn init_cache() -> Cache {
  let path = if let Ok(x) = std::env::var("ARECIBO_CACHE") {
    tracing::debug!("{:?}", &x);
    PathBuf::from(x)
  } else {
    let path = home::home_dir()
      .expect("targets without $HOME not supported")
      .join(".arecibo/");
    std::env::set_var("ARECIBO_CACHE", &path);
    path
  };

  create_dir_all(&path).unwrap();
  create_dir_all(path.join("commitment_keys")).unwrap();
  create_dir_all(path.join("public_params")).unwrap();

  Cache {
    inner: Arc::new(Mutex::new(path)),
  }
}

/// Cache structure that holds the root directory of all the cached files
pub struct Cache {
  inner: Arc<Mutex<PathBuf>>,
}

impl Cache {
  /// Sets the inner root directory of the cache
  pub fn set_inner(&self, path: &PathBuf) {
    self.inner.lock().unwrap().clone_from(path);

    create_dir_all(path).unwrap();
    create_dir_all(path.join("commitment_keys")).unwrap();
    create_dir_all(path.join("public_params")).unwrap();
  }

  /// Gets the inner root directory of the cache
  pub fn get_inner(&self) -> PathBuf {
    self.inner.lock().unwrap().to_path_buf()
  }

  /// Returns the commitment key file, and creates one if it doesn't exist
  pub fn ck_file<G: Group>(&self) -> io::Result<File> {
    let path = self.get_inner().join("commitment_keys").join(G::name());
    let exists = path.exists();

    let mut file = OpenOptions::new()
      .read(true)
      .write(true)
      .create(true)
      .open(path)?;

    if !exists {
      let ck = G::CE::setup(b"ck", 0);
      file.write_u64::<LittleEndian>(ck.length() as u64)?;
      ck.encode(&mut file)?;
    }

    file.rewind()?;
    Ok(file)
  }

  /// Returns the file of the given digest, and creates one if it doesn't exist
  pub fn digest_file<F: PrimeField>(&self, digest: F) -> io::Result<(File, bool)> {
    let path = self
      .get_inner()
      .join("public_params")
      .join(format!("{:?}", digest));
    let exists = path.exists();

    let mut file = OpenOptions::new()
      .read(true)
      .write(true)
      .create(true)
      .open(path)?;

    file.rewind()?;
    Ok((file, exists))
  }

  /// Returns a commitment key from the cache. If the cached commitment key is shorter
  /// than what we need, we extend the key and update the cache as a side effect.
  ///
  /// This is the same as calling `G::CE::setup(b"ck", n)`
  pub fn commitment_key<G: Group>(&self, n: usize) -> io::Result<CommitmentKey<G>> {
    let mut file = self.ck_file::<G>()?;

    let ck_size = file.read_u64::<LittleEndian>()? as usize;
    println!("ck_size: {}, n: {}", ck_size, n);

    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    let skip = std::mem::size_of::<usize>();
    let bytes = &mut mmap[skip..]; // skip the first `usize` bytes

    if ck_size < n {
      let mut ck = CommitmentKey::<G>::decode(bytes, ck_size)?;
      G::CE::extend(&mut ck, b"ck", n - ck_size)
        .map_err(|e| Error::new(ErrorKind::Other, format!("{:?}", e)))?;

      file.rewind()?;
      file.write_u64::<LittleEndian>(ck.length() as u64)?;
      ck.encode(&mut file)?; // update the cache
      Ok(ck)
    } else {
      CommitmentKey::<G>::decode(bytes, n)
    }
  }

  /// Updates the cached commitment key. If the cached commitment key is shorter than the given one, we extend it
  pub fn set_commitment_key<G: Group>(&self, ck: &CommitmentKey<G>) -> io::Result<()> {
    let mut file = self.ck_file::<G>()?;

    let ck_size = file.read_u64::<LittleEndian>()? as usize;

    if ck_size < ck.length() {
      ck.encode(&mut file)?; // update the cache
    }

    Ok(())
  }

  /// Returns the length of the commitment key in cache
  pub fn commitment_key_len<G: Group>(&self) -> io::Result<usize> {
    let mut file = self.ck_file::<G>()?;

    let len = file.read_u64::<LittleEndian>()? as usize;
    Ok(len)
  }

  /// Returns a set of public params from the cache.
  ///
  /// This is the same as calling `PublicParams::new(non_uniform_circuit)`
  pub fn public_params<G1, G2, C1, C2, NC>(
    &self,
    non_uniform_circuit: &NC,
  ) -> io::Result<PublicParams<G1, G2, C1, C2>>
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
    C1: StepCircuit<G1::Scalar>,
    C2: StepCircuit<G2::Scalar>,
    NC: NonUniformCircuit<G1, G2, C1, C2>,
    <G1::Scalar as PrimeField>::Repr: Abomonation,
    <G2::Scalar as PrimeField>::Repr: Abomonation,
  {
    let num_circuits = non_uniform_circuit.num_circuits();

    let mut digests = vec![];
    let maybe_circuit_shapes = (0..num_circuits)
      .map(|circuit_index| {
        let circuit_primary = non_uniform_circuit.primary_circuit(circuit_index);
        let digest = circuit_primary.circuit_digest::<G1, G2>(num_circuits);
        digests.push(digest);

        self.get::<G1::Scalar, CircuitShape<G1>>(digest)
      })
      .collect::<Result<Vec<CircuitShape<G1>>, _>>();

    let circuit_digests = CircuitDigests::<G1>::new(digests);
    let maybe_aux_params = self.get::<G1::Scalar, AuxParams<G1, G2>>(circuit_digests.digest());

    let pp = match (maybe_circuit_shapes, maybe_aux_params) {
      (Ok(circuit_shapes), Ok(aux_params)) => {
        tracing::info!("cache hit");

        let size_primary = circuit_shapes
          .iter()
          .map(|circuit| commitment_key_size(&circuit.r1cs_shape, None))
          .max()
          .unwrap();
        let ck_primary = self.commitment_key::<G1>(size_primary)?;

        PublicParams::<G1, G2, C1, C2>::from_parts_unchecked(circuit_shapes, aux_params, ck_primary)
      }
      _ => {
        tracing::info!("generating public params");
        let pp = PublicParams::new(non_uniform_circuit);

        let (circuit_shapes, aux_params, ck_primary) = pp.into_parts();

        self.set::<G1::Scalar, AuxParams<G1, G2>>(circuit_digests.digest(), &aux_params)?;
        for (digest, circuit_shape) in circuit_digests.deref().iter().zip(circuit_shapes.iter()) {
          self.set::<G1::Scalar, CircuitShape<G1>>(*digest, circuit_shape)?;
        }
        self.set_commitment_key::<G1>(&ck_primary)?;

        PublicParams::<G1, G2, C1, C2>::from_parts_unchecked(circuit_shapes, aux_params, ck_primary)
      }
    };

    Ok(pp)
  }

  fn get<F: PrimeField, T: Abomonation + Clone>(&self, digest: F) -> io::Result<T> {
    let (file, exists) = self.digest_file::<F>(digest)?;

    if exists {
      unsafe {
        let mut mmap = MmapMut::map_mut(&file)?;
        let (t, remaining) = abomonation::decode::<T>(&mut mmap).unwrap();
        assert!(remaining.is_empty());
        Ok(t.clone())
      }
    } else {
      Err(Error::new(
        ErrorKind::Other,
        "failed to retrieve from cache",
      ))
    }
  }

  fn set<F: PrimeField, T: Abomonation>(&self, digest: F, t: &T) -> io::Result<()> {
    let (file, exists) = self.digest_file::<F>(digest)?;

    if exists {
      tracing::info!("object already in cache; nothing will be written");
    } else {
      let mut writer = BufWriter::new(file);
      unsafe { encode(t, &mut writer)? };
    }

    Ok(())
  }

  /// Deletes all the commitment keys in the cache
  ///
  /// **Warning**: This will indiscriminately wipe out everything in `<cache>/commitment_keys`,
  /// so don't call this unless you're **sure** that there's nothing important in there.
  pub fn clear_commitment_keys(&self) -> io::Result<()> {
    let path = self.get_inner().join("commitment_keys");
    remove_dir_all(&path)?;
    create_dir_all(path)
  }
}

#[cfg(test)]
mod tests {
  use std::path::PathBuf;

  use pasta_curves::pallas;

  use crate::{
    provider::{bn256_grumpkin::bn256, secp_secq::secq256k1},
    traits::{commitment::CommitmentEngineTrait, Group},
  };

  use super::ARECIBO_CACHE;

  fn test_ck_cache_with<G: Group>() {
    for n in [10, 100, 1000] {
      let ck = ARECIBO_CACHE.commitment_key::<G>(n).unwrap();
      assert_eq!(ck, G::CE::setup(b"ck", n));
      assert_eq!(
        ARECIBO_CACHE.commitment_key_len::<G>().unwrap(),
        n.next_power_of_two()
      );
    }
  }

  #[test]
  fn test_ck_cache() {
    ARECIBO_CACHE.set_inner(&PathBuf::from("./arecibo_cache_test"));
    ARECIBO_CACHE.clear_commitment_keys().unwrap();

    test_ck_cache_with::<bn256::Point>();
    test_ck_cache_with::<pallas::Point>();
    test_ck_cache_with::<secq256k1::Point>();

    ARECIBO_CACHE.clear_commitment_keys().unwrap();
  }
}
