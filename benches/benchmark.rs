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
        map::{OrderedDataMap, TrieMap},
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
            let _ = TrieMap::load(&data_dir.join("identifier-map.bin"))
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

criterion_group!(benches, bench_keyword_index, bench_data);
criterion_main!(benches);
