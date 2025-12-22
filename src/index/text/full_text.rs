use std::{
    cmp::Reverse,
    collections::HashMap,
    fs::{File, create_dir_all, remove_dir_all},
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

use crate::{
    data::{DataSource, TextData},
    index::{Match, Search, SearchParams},
    utils::load_u32_vec,
};
use anyhow::{Result, anyhow};
use log::info;
use ordered_float::OrderedFloat;
use tantivy::Index;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{IndexReader, collector::TopDocs};

struct Inner {
    data: TextData,
    index: Index,
    reader: IndexReader,
    parser: QueryParser,
    field_to_data: Vec<u32>,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("data", &self.data)
            .field("index", &self.index)
            .field("reader", &"IndexReader { ... }")
            .field("parser", &"QueryParser { ... }")
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct FullTextIndex {
    inner: Arc<Inner>,
}

pub struct BuildParams {}

impl FullTextIndex {
    fn field_to_column(&self, field_id: u32) -> usize {
        let field_to_data = &self.inner.field_to_data;

        let mut field_id = field_id as usize;
        let data_id = field_to_data[field_id] as usize;

        let mut offset = 0;
        while field_id > 0 && field_to_data[field_id - 1] == data_id as u32 {
            field_id -= 1;
            offset += 1;
        }
        offset
    }
}

impl Search for FullTextIndex {
    type Data = TextData;

    type Query<'e> = &'e str;

    type BuildParams = ();

    fn build(data: &Self::Data, index_dir: &Path, _params: &Self::BuildParams) -> Result<()>
    where
        Self: Sized,
    {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("text", TEXT);
        schema_builder.add_u64_field("field_id", STORED | INDEXED);
        let schema = schema_builder.build();

        if index_dir.exists() {
            // need to remove here because tantivy does not support overwriting existing index
            remove_dir_all(index_dir)?;
        };

        create_dir_all(index_dir)?;
        let index = Index::create_in_dir(index_dir, schema.clone())?;
        let mut index_writer = index.writer(100_000_000)?;

        let text_field = schema.get_field("text")?;
        let field_id_field = schema.get_field("field_id")?;

        // Log every 5% or every 100,000 embeddings, whichever is smaller
        let total_fields = data.total_fields();
        let log_every = (total_fields / 20).clamp(1, 100_000);

        let mut field_to_data_file =
            BufWriter::new(File::create(index_dir.join("index.field-to-data"))?);
        let mut field_id: u32 = 0;
        for (id, texts) in data.items() {
            for text in texts {
                if field_id == u32::MAX {
                    return Err(anyhow!("too many fields, max {} supported", u32::MAX));
                }

                let mut doc = TantivyDocument::default();
                doc.add_text(text_field, text);
                doc.add_u64(field_id_field, field_id as u64);
                index_writer.add_document(doc)?;

                field_id += 1;
                field_to_data_file.write_all(&id.to_le_bytes())?;

                if field_id.is_multiple_of(log_every) {
                    let percentage = (field_id as f64 / total_fields as f64) * 100.0;
                    info!(
                        "Indexed {} / {} embeddings ({:.1}%) from {} items",
                        field_id,
                        total_fields,
                        percentage,
                        id + 1
                    );
                }
            }
        }

        index_writer.commit()?;

        Ok(())
    }

    fn load(data: Self::Data, index_dir: &Path) -> Result<Self>
    where
        Self: Sized,
    {
        let mut index = Index::open_in_dir(index_dir)?;
        index.set_default_multithread_executor()?;

        let reader = index.reader()?;

        let text_field = index.schema().get_field("text")?;
        let parser = QueryParser::for_index(&index, vec![text_field]);

        let field_to_data = load_u32_vec(&index_dir.join("index.field-to-data"))?;

        Ok(FullTextIndex {
            inner: Arc::new(Inner {
                data,
                index,
                reader,
                parser,
                field_to_data,
            }),
        })
    }

    fn search(&self, query: Self::Query<'_>, params: &SearchParams) -> Result<Vec<Match>> {
        let index = &self.inner.index;
        let schema = index.schema();
        let field_id_field = schema.get_field("field_id")?;

        let query = self.inner.parser.parse_query(query)?;

        let search_k = params.search_k(&self.inner.data);
        let top_docs = TopDocs::with_limit(search_k);

        let searcher = self.inner.reader.searcher();

        let mut matches = Vec::with_capacity(params.k);
        for (score, doc) in searcher.search(&query, &top_docs)? {
            if let Some(min_score) = params.min_score
                && score < min_score
            {
                continue;
            }

            let doc: HashMap<_, _> = searcher.doc(doc)?;
            let field_id = doc
                .get(&field_id_field)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow!("Missing 'field_id' field in document"))?
                as u32;

            let data_id = self.inner.field_to_data[field_id as usize];

            matches.push((data_id, field_id, score));
        }

        // Sort by data_id, then by score descending, then by field_id
        matches.sort_by_key(|&(data_id, field_id, score)| {
            (data_id, Reverse(OrderedFloat(score)), field_id)
        });

        // Deduplicate by data_id (keeping the best score)
        matches.dedup_by(|a, b| a.0 == b.0);

        // Sort by score descending, then by data_id
        matches.sort_by_key(|&(data_id, _field_id, score)| (Reverse(OrderedFloat(score)), data_id));

        // Take top k
        let matches: Vec<Match> = matches
            .into_iter()
            .map(|(data_id, field_id, score)| {
                Match::WithField(data_id, self.field_to_column(field_id), score)
            })
            .take(params.k)
            .collect();

        Ok(matches)
    }

    fn search_with_filter<F>(
        &self,
        _query: Self::Query<'_>,
        _params: &SearchParams,
        _filter: F,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        Err(anyhow!(
            "Filtered search is not supported for full-text index"
        ))
    }

    fn data(&self) -> &Self::Data {
        &self.inner.data
    }

    fn index_type(&self) -> &'static str {
        "full-text"
    }
}
