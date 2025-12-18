use std::{
    fs::{File, create_dir_all},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use criterion::{Criterion, criterion_group, criterion_main};

use search_rdf::{
    data::{
        DataSource, TextData,
        map::{OrderedDataMap, TrieMap},
    },
    index::{SearchIndex, SearchParams, keyword::KeywordIndex},
};

fn convert_tsv_to_binary(tsv_file: &Path, data_dir: &Path) -> std::io::Result<()> {
    create_dir_all(data_dir)?;
    let data_file = data_dir.join("data");
    let mut out = BufWriter::new(File::create(&data_file)?);

    let file = File::open(tsv_file)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split('\t');

        if let Some(identifier) = parts.next() {
            let fields: Vec<_> = parts.collect();

            // Write identifier length (u16)
            let key_bytes = identifier.as_bytes();
            out.write_all(&(key_bytes.len() as u16).to_le_bytes())?;

            // Write identifier
            out.write_all(key_bytes)?;

            // Write number of fields (u16)
            out.write_all(&(fields.len() as u16).to_le_bytes())?;

            // Write each field
            for field in fields {
                // Write field length (u32)
                let value_bytes = field.as_bytes();
                out.write_all(&(value_bytes.len() as u32).to_le_bytes())?;

                // Write field value
                out.write_all(value_bytes)?;
            }
        }
    }

    Ok(())
}

fn bench_data(c: &mut Criterion) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let base_dir = Path::new(dir).join("benches");
    let tsv_file = base_dir.join("data.tsv");
    let data_dir = base_dir.join("data");

    // Convert TSV to binary format and build
    convert_tsv_to_binary(&tsv_file, &data_dir).expect("Failed to convert TSV");
    TextData::build(&data_dir).expect("Failed to build data");
    let data = TextData::load(&data_dir).expect("Failed to load data");

    let mut g = c.benchmark_group("data");

    g.bench_function("build_data", |b| {
        b.iter(|| {
            TextData::build(&data_dir).expect("Failed to build data");
        })
    });

    g.bench_function("load_data", |b| {
        b.iter(|| {
            let _ = TextData::load(&data_dir).expect("Failed to load data");
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

    convert_tsv_to_binary(&tsv_file, &data_dir).expect("Failed to convert TSV");
    TextData::build(&data_dir).expect("Failed to build data");
    let data = TextData::load(&data_dir).expect("Failed to load data");

    KeywordIndex::build(&data, &index_dir, ()).expect("Failed to build index");

    let index = KeywordIndex::load(data, &index_dir).expect("Failed to load index");

    let mut g = c.benchmark_group("keyword_index");

    let queries = vec!["the united states", "angela m"];
    let ks = vec![10, 100];

    let filter = |id: u32| id % 2 == 0;

    for query in queries {
        for &k in &ks {
            for &exact in &[true, false] {
                g.bench_function(
                    format!("search: query={query}, k={k}, exact={exact}"),
                    |b| {
                        b.iter(|| {
                            let _ = index
                                .search(query, SearchParams::default().with_k(k).with_exact(exact))
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
                                    SearchParams::default().with_k(k).with_exact(exact),
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
