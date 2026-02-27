pub mod embedding;
pub mod python;
pub mod text;

use crate::data::DataSource;
use anyhow::Result;
use pyo3::IntoPyObject;
use std::collections::HashMap;
use std::path::Path;

pub use embedding::EmbeddingIndexWithData;
pub use embedding::{EmbeddingIndex, EmbeddingIndexParams, EmbeddingSearchParams, Metric};
pub use text::full_text::{FullTextIndex, FullTextSearchParams};
pub use text::fuzzy::{FuzzyIndex, FuzzySearchParams};
pub use text::keyword::{KeywordIndex, KeywordSearchParams};

/// Result from a search query
#[derive(Debug, Clone, PartialEq, IntoPyObject)]
pub enum Match {
    // Id and score
    Regular(u32, f32),
    // Data point id with field index and a score
    WithField(u32, usize, f32),
}

impl Match {
    /// Get the data point id
    pub fn id(&self) -> u32 {
        match self {
            Match::Regular(id, ..) => *id,
            Match::WithField(id, ..) => *id,
        }
    }

    /// Get the score
    pub fn score(&self) -> f32 {
        match self {
            Match::Regular(.., score) => *score,
            Match::WithField(.., score) => *score,
        }
    }
}

/// Implemented by any match type that carries an id and a score.
pub trait Scored {
    fn id(&self) -> u32;
    fn score(&self) -> f32;
}

impl Scored for Match {
    fn id(&self) -> u32 {
        self.id()
    }
    fn score(&self) -> f32 {
        self.score()
    }
}

/// Merge per-field match lists: keep best score per id, sort desc, truncate to k.
pub fn merge_neighbor_matches<T: Scored>(results: Vec<Vec<T>>, k: usize) -> Vec<T> {
    let mut best: HashMap<u32, T> = HashMap::new();
    for m in results.into_iter().flatten() {
        let id = m.id();
        let score = m.score();
        let prev = best.get(&id).map(|m| m.score()).unwrap_or(f32::NEG_INFINITY);
        if score > prev {
            best.insert(id, m);
        }
    }
    let mut merged: Vec<T> = best.into_values().collect();
    merged.sort_unstable_by(|a, b| {
        b.score()
            .partial_cmp(&a.score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged.truncate(k);
    merged
}

// for use with serde defaults
fn default_k() -> usize {
    10
}

pub trait SearchParamsExt {
    fn k(&self) -> usize;

    fn exact(&self) -> bool;

    fn search_k(&self, data: &impl DataSource) -> usize {
        if self.exact() {
            self.k() * data.max_fields().max(1) as usize
        } else {
            (self.k() as f32 * data.avg_fields()).ceil() as usize
        }
    }
}

/// Core trait for all search indices
pub trait Search: Send + Sync {
    /// The data type this index operates on
    type Data: DataSource;
    /// The query type this index expects
    type Query<'e>: Send + Sync + ?Sized;
    /// The build parameters the index accepts
    type BuildParams;
    /// The search parameters the index accepts
    type SearchParams: SearchParamsExt;

    /// Build and save an index
    fn build(data: &Self::Data, index_dir: &Path, params: &Self::BuildParams) -> Result<()>
    where
        Self: Sized;

    /// Load an index from disk
    fn load(data: Self::Data, index_dir: &Path) -> Result<Self>
    where
        Self: Sized;

    /// Search the index with a query
    fn search(&self, query: Self::Query<'_>, params: &Self::SearchParams) -> Result<Vec<Match>>;

    /// Search the index with a query and filter
    fn search_with_filter<F>(
        &self,
        query: Self::Query<'_>,
        params: &Self::SearchParams,
        filter: F,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool;

    /// Get reference to underlying data
    fn data(&self) -> &Self::Data;

    /// Get the type name of this index
    fn index_type(&self) -> &'static str;
}
