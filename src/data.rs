//! Very minimal utilities for reading/writing general arecibo data in disk.
use camino::Utf8Path;
#[cfg(not(target_arch = "wasm32"))]
use camino::Utf8PathBuf;
use once_cell::sync::OnceCell;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::{
  collections::HashMap,
  fs::{self, File, OpenOptions},
  io::{BufReader, BufWriter},
};

/// Global flag for writing config.
pub static WRITE: bool = false;

/// Path to the directory where Arecibo data will be stored.
pub static ARECIBO_DATA: &str = ".arecibo_data";

/// Global configuration for Arecibo data storage, including root directory and counters.
/// This configuration is initialized on first use.
pub static ARECIBO_CONFIG: OnceCell<Mutex<DataConfig>> = OnceCell::new();

/// Configuration for managing Arecibo data files, including the root directory,
/// witness counter, and cross-term counter for organizing files.
#[derive(Debug, Clone, Default)]
pub struct DataConfig {
  #[cfg(not(target_arch = "wasm32"))]
  root_dir: Utf8PathBuf,
  #[cfg(not(target_arch = "wasm32"))]
  section_counters: HashMap<String, usize>,
  write_data: bool,
}

#[cfg(not(target_arch = "wasm32"))]
/// Initializes the global configuration for Arecibo data storage, setting up the root directory
/// and initializing counters. We create the root directory if it does not already exist.
pub fn init_config() -> Mutex<DataConfig> {
  let root_dir = home::home_dir().unwrap().join(ARECIBO_DATA);
  let root_dir = Utf8PathBuf::from_path_buf(root_dir).unwrap();
  if !root_dir.exists() {
    fs::create_dir_all(&root_dir).expect("Failed to create arecibo data directory");
  }

  let config = DataConfig {
    root_dir,
    section_counters: HashMap::new(),
    write_data: WRITE,
  };

  Mutex::new(config)
}

#[cfg(target_arch = "wasm32")]
/// Initializes the global configuration for Arecibo data storage, setting up the root directory
/// and initializing counters. We create the root directory if it does not already exist.
pub fn init_config() -> Mutex<DataConfig> {
  Mutex::new(DataConfig::default())
}

#[cfg(not(target_arch = "wasm32"))]
/// Writes Arecibo data to disk, organizing it into sections and labeling it with a unique identifier.
/// This function serializes the given payload and writes it into the appropriate section and file.
/// For now, we just increment the relevant counter to ensure uniqueness.
pub fn write_arecibo_data<T: Serialize>(
  section: impl AsRef<Utf8Path>,
  label: impl AsRef<Utf8Path>,
  payload: &T,
) {
  let mutex = ARECIBO_CONFIG.get_or_init(init_config);
  let mut config = mutex.lock().unwrap();

  let section_path = config.root_dir.join(section.as_ref());
  if !section_path.exists() {
    fs::create_dir_all(&section_path).expect("Failed to create section directory");
  }

  let section = section.as_ref().to_string();
  let counter = config.section_counters.entry(section).or_insert(0);

  let file_path = section_path.join(format!("{}_{:?}", label.as_ref().as_str(), counter));
  *counter += 1;

  let file = OpenOptions::new()
    .read(true)
    .write(true)
    .truncate(true)
    .create(true)
    .open(file_path)
    .expect("Failed to create data file");

  let writer = BufWriter::new(&file);
  bincode::serialize_into(writer, payload).expect("Failed to write data");
}

#[cfg(target_arch = "wasm32")]
/// Writes Arecibo data to disk, organizing it into sections and labeling it with a unique identifier.
/// This function serializes the given payload and writes it into the appropriate section and file.
/// For now, we just increment the relevant counter to ensure uniqueness.
pub fn write_arecibo_data<T: Serialize>(
  _section: impl AsRef<Utf8Path>,
  _label: impl AsRef<Utf8Path>,
  _payload: &T,
) {
  // Do nothing
}

#[cfg(not(target_arch = "wasm32"))]
/// Reads and deserializes data from a specified section and label.
pub fn read_arecibo_data<T: DeserializeOwned>(
  section: impl AsRef<Utf8Path>,
  label: impl AsRef<Utf8Path>,
) -> T {
  let mutex = ARECIBO_CONFIG.get_or_init(init_config);
  let config = mutex.lock().unwrap();

  let section_path = config.root_dir.join(section.as_ref());
  assert!(section_path.exists(), "Section directory does not exist");

  // Assuming the label uniquely identifies the file, and ignoring the counter for simplicity
  let file_path = section_path.join(label.as_ref());
  assert!(file_path.exists(), "Data file does not exist");

  let file = File::open(file_path).expect("Failed to open data file");
  let reader = BufReader::new(file);

  bincode::deserialize_from(reader).expect("Failed to read data")
}

#[cfg(target_arch = "wasm32")]
/// Reads and deserializes data from a specified section and label.
pub fn read_arecibo_data<T: DeserializeOwned>(
  _section: impl AsRef<Utf8Path>,
  _label: impl AsRef<Utf8Path>,
) -> T {
  unimplemented!("not supported on wasm yet")
}

/// Are we configured to write data?
pub fn write_data() -> bool {
  let mutex = ARECIBO_CONFIG.get_or_init(init_config);
  let config = mutex.lock().unwrap();
  config.write_data
}

/// Set the configuration for writing data.
pub fn set_write_data(write_data: bool) {
  let mutex = ARECIBO_CONFIG.get_or_init(init_config);
  let mut config = mutex.lock().unwrap();
  config.write_data = write_data;
}
