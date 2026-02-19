use std::{
    fs::{File, create_dir_all},
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::Result;
use criterion::{Criterion, criterion_group, criterion_main};

use search_rdf::{
    data::{
        Data, DataSource,
        item::{FieldType, Item, StringField},
        map::{FstMap, OrderedDataMap, TrieMap},
    },
    index::{
        Search,
        text::keyword::{KeywordIndex, KeywordSearchParams},
    },
};

fn read_tsv_items(tsv_file: &Path) -> Result<Vec<Item>> {
    let file = File::open(tsv_file)?;
    let reader = BufReader::new(file);

    let mut items = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split('\t');

        if let Some(identifier) = parts.next() {
            let fields: Vec<_> = parts
                .map(|s| StringField {
                    field_type: FieldType::Text,
                    value: s.to_string(),
                    tags: vec![],
                })
                .collect();
            let item = Item::from_string_fields(identifier.to_string(), fields)?;
            items.push(item);
        }
    }

    Ok(items)
}

fn bench_data(c: &mut Criterion) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let base_dir = Path::new(dir).join("benches");
    let tsv_file = base_dir.join("data.tsv");
    let data_dir = base_dir.join("data");

    // Read TSV items and build data
    let items = read_tsv_items(&tsv_file).expect("Failed to read TSV items");
    Data::build(items.iter().cloned().map(Ok), &data_dir).expect("Failed to build data");
    let data = Data::load(&data_dir).expect("Failed to load data");

    let mut g = c.benchmark_group("data");

    g.bench_function("build_data", |b| {
        b.iter(|| {
            Data::build(items.iter().cloned().map(Ok), &data_dir).expect("Failed to build data");
        })
    });

    g.bench_function("load_data", |b| {
        b.iter(|| {
            let _ = Data::load(&data_dir).expect("Failed to load data");
        })
    });

    // Benchmark load of OrderedDataMap
    g.bench_function("load_data_map", |b| {
        b.iter(|| {
            let _ = OrderedDataMap::load(&data_dir.join("data-map.bin"))
                .expect("Failed to load data map");
        })
    });

    // Benchmark load of TrieMap
    g.bench_function("load_trie_map", |b| {
        b.iter(|| {
            let _ = TrieMap::load(&data_dir.join("identifier-map.trie.bin"))
                .expect("Failed to load trie map");
        })
    });

    g.bench_function("field", |b| {
        let id = (data.len() / 2) as u32;
        b.iter(|| {
            let _ = data.field(id, 0).expect("Failed to get field");
        })
    });

    g.bench_function("fields", |b| {
        let id = (data.len() / 2) as u32;
        b.iter(|| {
            let _ = data.fields(id).expect("Failed to get fields");
        })
    });

    for (id, name) in [
        (0u32, "first"),
        ((data.len() / 2) as u32, "middle"),
        ((data.len() - 1) as u32, "last"),
    ] {
        let identifier = data.identifier(id).expect("Failed to get identifier");
        assert!(data.id_from_identifier(identifier).is_some());
        g.bench_function(format!("id_from_identifier_{}", name), |b| {
            b.iter(|| {
                let _ = data
                    .id_from_identifier(identifier)
                    .expect("Failed to find id");
            })
        });
    }

    g.finish();
}

fn bench_keyword_index(c: &mut Criterion) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let base_dir = Path::new(dir).join("benches");
    let tsv_file = base_dir.join("data.tsv");
    let data_dir = base_dir.join("data");
    let index_dir = base_dir.join("index");

    // create index dir if it doesn't exist
    create_dir_all(&index_dir).expect("Failed to create index dir");

    println!(
        "Building index at {} from {}",
        index_dir.display(),
        tsv_file.display()
    );

    let items = read_tsv_items(&tsv_file).expect("Failed to read TSV items");
    Data::build(items.into_iter().map(Ok), &data_dir).expect("Failed to build data");
    let data = Data::load(&data_dir).expect("Failed to load data");

    KeywordIndex::build(&data, &index_dir, &()).expect("Failed to build index");

    let index = KeywordIndex::load(data, &index_dir).expect("Failed to load index");

    let mut g = c.benchmark_group("keyword_index");

    let queries = vec!["the united states", "angela m"];
    let ks = vec![10, 100];

    let filter = |id: u32| id.is_multiple_of(2);

    for query in queries {
        for &k in &ks {
            for &exact in &[true, false] {
                g.bench_function(
                    format!("search: query={query}, k={k}, exact={exact}"),
                    |b| {
                        b.iter(|| {
                            let _ = index
                                .search(
                                    query,
                                    &KeywordSearchParams {
                                        k,
                                        exact,
                                        ..Default::default()
                                    },
                                )
                                .expect("Failed to find matches");
                        })
                    },
                );

                g.bench_function(
                    format!("search_with_filter: query={query}, k={k}, exact={exact}"),
                    |b| {
                        b.iter(|| {
                            let _ = index
                                .search_with_filter(
                                    query,
                                    &KeywordSearchParams {
                                        k,
                                        exact,
                                        ..Default::default()
                                    },
                                    filter,
                                )
                                .expect("Failed to find matches");
                        })
                    },
                );
            }
        }
    }

    g.finish();
}

fn bench_map_comparison(c: &mut Criterion) {
    use tempfile::tempdir;

    let mut g = c.benchmark_group("map_comparison");

    // Test with different dataset sizes
    let sizes = [1_000, 10_000, 100_000];

    for &size in &sizes {
        // Generate test data
        let identifiers: Vec<String> = (0..size).map(|i| format!("Q{}", i)).collect();

        // Benchmark TrieMap building
        g.bench_function(format!("trie_map_build_{}", size), |b| {
            b.iter(|| {
                let mut map = TrieMap::new();
                for (id, identifier) in identifiers.iter().enumerate() {
                    map.add(identifier, id as u32).unwrap();
                }
                map
            })
        });

        // Benchmark FstMap building
        g.bench_function(format!("fst_map_build_{}", size), |b| {
            b.iter(|| {
                let mut map = FstMap::new();
                for (id, identifier) in identifiers.iter().enumerate() {
                    map.add(identifier, id as u32).unwrap();
                }
                map
            })
        });

        // Build maps for save/load benchmarks
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let trie_path = temp_dir.path().join("trie.bin");
        let fst_path = temp_dir.path().join("fst.bin");

        let mut trie_map = TrieMap::new();
        let mut fst_map = FstMap::new();
        for (id, identifier) in identifiers.iter().enumerate() {
            trie_map.add(identifier, id as u32).unwrap();
            fst_map.add(identifier, id as u32).unwrap();
        }

        // Benchmark TrieMap save
        g.bench_function(format!("trie_map_save_{}", size), |b| {
            b.iter(|| {
                trie_map.save(&trie_path).unwrap();
            })
        });

        // Benchmark FstMap save
        g.bench_function(format!("fst_map_save_{}", size), |b| {
            b.iter(|| {
                fst_map.save(&fst_path).unwrap();
            })
        });

        // Save once for load benchmarks
        trie_map.save(&trie_path).unwrap();
        fst_map.save(&fst_path).unwrap();

        // Print file sizes
        let trie_size = std::fs::metadata(&trie_path).unwrap().len();
        let fst_size = std::fs::metadata(&fst_path).unwrap().len();
        println!(
            "Size {}: TrieMap = {} bytes, FstMap = {} bytes, Ratio = {:.2}x",
            size,
            trie_size,
            fst_size,
            trie_size as f64 / fst_size as f64
        );

        // Benchmark TrieMap load
        g.bench_function(format!("trie_map_load_{}", size), |b| {
            b.iter(|| TrieMap::load(&trie_path).unwrap())
        });

        // Benchmark FstMap load
        g.bench_function(format!("fst_map_load_{}", size), |b| {
            b.iter(|| FstMap::load(&fst_path).unwrap())
        });

        // Load maps for lookup benchmarks
        let loaded_trie = TrieMap::load(&trie_path).unwrap();
        let loaded_fst = FstMap::load(&fst_path).unwrap();

        // Benchmark lookups at different positions
        let test_positions = [(0, "first"), (size / 2, "middle"), (size - 1, "last")];

        for (pos, label) in test_positions {
            let identifier = &identifiers[pos];

            // Benchmark TrieMap get
            g.bench_function(format!("trie_map_get_{}_{}", label, size), |b| {
                b.iter(|| loaded_trie.get(identifier).unwrap())
            });

            // Benchmark FstMap get
            g.bench_function(format!("fst_map_get_{}_{}", label, size), |b| {
                b.iter(|| loaded_fst.get(identifier).unwrap())
            });
        }

        // Benchmark get for non-existent key
        let non_existent = format!("Q{}", size * 2);

        g.bench_function(format!("trie_map_get_miss_{}", size), |b| {
            b.iter(|| loaded_trie.get(&non_existent))
        });

        g.bench_function(format!("fst_map_get_miss_{}", size), |b| {
            b.iter(|| loaded_fst.get(&non_existent))
        });
    }

    g.finish();
}

criterion_group!(
    benches,
    bench_keyword_index,
    bench_data,
    bench_map_comparison
);
criterion_main!(benches);
