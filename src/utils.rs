use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use serde::{Serialize, de::DeserializeOwned};

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, value)?;
    Ok(())
}

pub fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let value = serde_json::from_reader(reader)?;
    Ok(value)
}

pub fn write_bincode(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    bincode::serde::encode_into_std_write(value, &mut writer, bincode::config::standard())?;
    Ok(())
}

pub fn load_bincode<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let value = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())?;
    Ok(value)
}

pub fn write_u32_vec(path: &Path, data: &[u32]) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    for &value in data {
        f.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

pub fn load_u32_vec(path: &Path) -> Result<Vec<u32>> {
    let bytes = unsafe { Mmap::map(&File::open(path)?)? };
    let (head, values, tail) = unsafe { bytes.as_ref().align_to::<u32>() };
    if !head.is_empty() || !tail.is_empty() {
        return Err(anyhow!("unaligned u32 data in file: {}", path.display()));
    }
    Ok(values.to_vec())
}

pub fn write_u64_vec_from_usize(path: &Path, data: &[usize]) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    for &value in data {
        let value =
            u64::try_from(value).map_err(|e| anyhow!("failed to convert usize to u64: {}", e))?;
        f.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

pub fn load_usize_vec_from_u64(path: &Path) -> Result<Vec<usize>> {
    let bytes = unsafe { Mmap::map(&File::open(path)?)? };
    let (head, values, tail) = unsafe { bytes.as_ref().align_to::<u64>() };
    if !head.is_empty() || !tail.is_empty() {
        return Err(anyhow!("unaligned u64 data in file: {}", path.display()));
    }
    values
        .iter()
        .map(|&v| usize::try_from(v).map_err(|e| anyhow!("failed to convert u64 to usize: {}", e)))
        .collect::<Result<_>>()
}

pub fn progress_bar(message: &str, total: Option<u64>) -> Result<ProgressBar> {
    // Only show progress bars if SEARCH_RDF_PROGRESS environment variable is set
    let show_progress = std::env::var("SEARCH_RDF_PROGRESS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false);

    if !show_progress {
        // Return a completely hidden progress bar
        return Ok(ProgressBar::hidden());
    }

    // Create a visible progress bar with styling
    let pb = if let Some(total) = total {
        ProgressBar::new(total).with_style(
            ProgressStyle::with_template(
                "{msg} [{wide_bar:.cyan/blue}] ({pos:.}/{len:.}, TIME {elapsed}, ETA {eta})",
            )?
            .progress_chars("#>-"),
        )
    } else {
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("{msg} {spinner} TIME {elapsed}")?.tick_chars("/|\\- "),
        )
    };
    pb.set_message(message.to_string());
    Ok(pb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct TestData {
        name: String,
        value: i32,
    }

    #[test]
    fn test_write_json() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test.json");
        let data = TestData {
            name: "test".to_string(),
            value: 42,
        };

        write_json(&path, &data).expect("Failed to write");
        assert!(path.exists());
    }

    #[test]
    fn test_load_json() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test.json");
        let data = TestData {
            name: "test".to_string(),
            value: 42,
        };

        write_json(&path, &data).expect("Failed to write");
        let loaded: TestData = load_json(&path).expect("Failed to load");
        assert_eq!(data, loaded);
    }
}
