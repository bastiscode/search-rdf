mod search_rdf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "search-rdf")]
#[command(about = "Build and serve RDF search indices", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download and prepare text data
    Data {
        /// Path to configuration file
        config: String,
        /// Force rebuild even if output exists
        #[arg(long)]
        force: bool,
    },

    /// Generate embeddings from text data
    Embed {
        /// Path to configuration file
        config: String,
        /// Force rebuild even if output exists
        #[arg(long)]
        force: bool,
        /// Only process specific datasets (by name)
        #[arg(long)]
        only: Option<Vec<String>>,
    },

    /// Build search indices
    Index {
        /// Path to configuration file
        config: String,
        /// Force rebuild even if output exists
        #[arg(long)]
        force: bool,
        /// Only build specific indices (by name)
        #[arg(long)]
        only: Option<Vec<String>>,
    },

    /// Serve indices via HTTP
    Serve {
        /// Path to configuration file
        config: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Data { config, force } => {
            search_rdf::data::run(&config, force)?;
        }
        Commands::Embed { config, force, only } => {
            search_rdf::embed::run(&config, force, only)?;
        }
        Commands::Index { config, force, only } => {
            search_rdf::index::run(&config, force, only)?;
        }
        Commands::Serve { config } => {
            search_rdf::serve::run(&config).await?;
        }
    }

    Ok(())
}
