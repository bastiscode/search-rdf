pub mod embedding;
pub mod text;

use crate::data::DataSource;
use anyhow::Result;
use pyo3::IntoPyObject;
use std::path::Path;

pub use embedding::{EmbeddingIndex, EmbeddingIndexParams, Metric};
pub use text::embedding::TextEmbeddingIndex;
pub use text::keyword;
pub use text::keyword::KeywordIndex;

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

#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Number of results to return
    pub k: usize,
    /// Minimum score threshold
    pub min_score: Option<f32>,
    // Perform exact or approximate search
    pub exact: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            min_score: None,
            exact: false,
        }
    }
}

impl SearchParams {
    pub fn search_k(&self, data: &impl DataSource) -> usize {
        if self.exact {
            self.k * data.max_fields().max(1) as usize
        } else {
            (self.k as f32 * data.avg_fields()).ceil() as usize
        }
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn with_exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }

    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }
}

/// Core trait for all search indices
pub trait SearchIndex: Send + Sync {
    /// The data type this index operates on
    type Data: DataSource;
    /// The query type this index expects
    type Query<'e>: Send + Sync + ?Sized;
    /// The build parameters the index accepts
    type BuildParams;

    /// Build and save an index
    fn build(data: &Self::Data, index_dir: &Path, params: &Self::BuildParams) -> Result<()>
    where
        Self: Sized;

    /// Load an index from disk
    fn load(data: Self::Data, index_dir: &Path) -> Result<Self>
    where
        Self: Sized;

    /// Search the index with a query
    fn search(&self, query: Self::Query<'_>, params: &SearchParams) -> Result<Vec<Match>>;

    /// Search the index with a query and filter
    fn search_with_filter<F>(
        &self,
        query: Self::Query<'_>,
        params: &SearchParams,
        filter: F,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool;

    /// Get reference to underlying data
    fn data(&self) -> &Self::Data;

    /// Get the type name of this index
    fn index_type(&self) -> &'static str;
}
