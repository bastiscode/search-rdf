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
    /// Download and prepare data
    Data {
        /// Path to configuration file
        config: String,
        /// Force rebuild even if output exists
        #[arg(long)]
        force: bool,
    },

    /// Generate embeddings for data
    Embed {
        /// Path to configuration file
        config: String,
        /// Force rebuild even if output exists
        #[arg(long)]
        force: bool,
    },

    /// Build search indices
    Index {
        /// Path to configuration file
        config: String,
        /// Force rebuild even if output exists
        #[arg(long)]
        force: bool,
    },

    /// Serve indices via HTTP
    Serve {
        /// Path to configuration file
        config: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Data { config, force } => search_rdf::data::run(&config, force),
        Commands::Embed { config, force } => search_rdf::embed::run(&config, force),
        Commands::Index { config, force } => search_rdf::index::run(&config, force),
        Commands::Serve { config } => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(num_cpus::get())
                .max_blocking_threads(num_cpus::get() * 4)
                .enable_all()
                .build()?;

            runtime.block_on(async { search_rdf::serve::run(&config).await })
        }
    }
}
