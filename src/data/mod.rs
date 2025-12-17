pub mod embedding;
pub mod map;
pub mod text;

pub use embedding::{Embedding, Embeddings, Precision};
pub use text::{TextData, embedding::TextEmbeddings};

/// Core trait for data sources that can be indexed
pub trait DataSource: Send + Sync + Clone {
    /// The type of data in a single field
    type Field<'a>
    where
        Self: 'a;

    /// Number of data points
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of fields across all data points
    fn total_fields(&self) -> usize;

    /// Avg. number of fields per data point
    fn avg_fields(&self) -> f32 {
        self.total_fields() as f32 / self.len().max(1) as f32
    }

    /// Max number of fields for any single data point
    fn max_fields(&self) -> usize;

    /// Get number of searchable fields for a data point
    /// Returns None if the ID is invalid
    fn num_fields(&self, id: u32) -> Option<usize>;

    /// Get a specific field value for a data point
    fn field(&self, id: u32, field: usize) -> Option<Self::Field<'_>>;

    /// Get all searchable fields as a collection for a data point
    fn fields(&self, id: u32) -> Option<impl Iterator<Item = Self::Field<'_>>>;

    fn items(&self) -> impl Iterator<Item = (u32, Vec<Self::Field<'_>>)> + '_;

    /// Get the type name of this data source
    fn data_type(&self) -> &'static str;
}
