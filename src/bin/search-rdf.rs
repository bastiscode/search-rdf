mod search_rdf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "search-rdf")]
#[command(about = "Build and serve RDF search indices", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to configuration file (used when no subcommand is specified)
    #[arg(global = true)]
    config: Option<String>,

    /// Force rebuild even if output exists (used when no subcommand is specified)
    #[arg(long, global = true)]
    force: bool,
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
        Some(Commands::Data { config, force }) => search_rdf::data::run(&config, force),
        Some(Commands::Embed { config, force }) => search_rdf::embed::run(&config, force),
        Some(Commands::Index { config, force }) => search_rdf::index::run(&config, force),
        Some(Commands::Serve { config }) => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(num_cpus::get())
                .max_blocking_threads(num_cpus::get() * 4)
                .enable_all()
                .build()?;

            runtime.block_on(async { search_rdf::serve::run(&config).await })
        }
        None => {
            // Run all steps in sequence
            let config = cli
                .config
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Configuration file is required"))?;
            let force = cli.force;

            println!("Running all steps in sequence...\n");

            // Step 1: Data
            println!("=== Step 1/4: Data ===");
            search_rdf::data::run(config, force)?;

            // Step 2: Embed
            println!("\n=== Step 2/4: Embed ===");
            search_rdf::embed::run(config, force)?;

            // Step 3: Index
            println!("\n=== Step 3/4: Index ===");
            search_rdf::index::run(config, force)?;

            // Step 4: Serve
            println!("\n=== Step 4/4: Serve ===");
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(num_cpus::get())
                .max_blocking_threads(num_cpus::get() * 4)
                .enable_all()
                .build()?;

            runtime.block_on(async { search_rdf::serve::run(config).await })
        }
    }
}
