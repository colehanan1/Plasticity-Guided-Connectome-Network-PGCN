#!/usr/bin/env python3
"""
GRN Downstream Connectivity Analysis - Main Entry Point

This script orchestrates the complete pipeline:
1. Identify and cache GRNs from FlyWire labels
2. Fetch downstream connectivity from FlyWire API
3. Analyze connectivity patterns and statistics
4. Generate publication-ready visualizations

Usage:
    python main.py --full-analysis
    python main.py --demo-mode
    python main.py --extract-only
    python main.py --visualize-only

Author: Claude AI
Date: 2025-11-03
"""

import argparse
import logging
import time
import yaml
import pickle
import psutil
from pathlib import Path
from datetime import datetime
import sys

# Import our modules
from grn_downstream_extractor import GRNDownstreamExtractor
from grn_connectivity_analyzer import GRNConnectivityAnalyzer
from grn_visualization import GRNConnectivityVisualizer

# Setup logging
logger = logging.getLogger(__name__)


def setup_logging(config: dict, log_file: str = None) -> None:
    """
    Configure logging based on config settings.

    Args:
        config: Configuration dictionary
        log_file: Optional log file path
    """
    level_str = config['logging']['level']
    level = getattr(logging, level_str, logging.INFO)
    fmt = config['logging']['format']

    handlers = [logging.StreamHandler(sys.stdout)]

    if config['logging']['save_to_file']:
        log_path = log_file or config['logging']['log_file']
        handlers.append(logging.FileHandler(log_path, mode='w'))

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers
    )

    logger.info("="*80)
    logger.info("GRN DOWNSTREAM CONNECTIVITY ANALYSIS PIPELINE")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log level: {level_str}")
    logger.info("="*80)


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'grn_viz_config.yaml'

    logger.info(f"Loading configuration from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("✓ Configuration loaded successfully")
    return config


def run_extraction(config: dict, demo_mode: bool = False) -> tuple:
    """
    Run Phase 1 & 2: Identify GRNs and fetch connectivity.

    Args:
        config: Configuration dictionary
        demo_mode: Whether to run in demo mode

    Returns:
        Tuple of (sugar_grns, all_grns, sugar_connectivity, all_connectivity)
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: IDENTIFY & CACHE GRNs")
    logger.info("="*80)

    extractor = GRNDownstreamExtractor(config)

    # Load and filter GRNs
    csv_path = config['data']['processed_labels_path']

    logger.info(f"Loading GRNs from: {csv_path}")
    sugar_grns = extractor.load_and_filter_grns(csv_path, sugar_only=True)
    all_grns = extractor.load_and_filter_grns(csv_path, sugar_only=False)

    # Apply demo mode filtering
    if demo_mode:
        logger.info("DEMO MODE: Limiting to small subset")
        n_sugar = config['demo']['n_sugar_grns']
        n_all = config['demo']['n_all_grns']
        sugar_grns = sugar_grns.head(n_sugar)
        all_grns = all_grns.head(n_all)
        logger.info(f"  Using {len(sugar_grns)} sugar GRNs and {len(all_grns)} total GRNs")

    # Save GRN lists
    extractor.save_grn_lists(sugar_grns, all_grns)

    logger.info("\n" + "="*80)
    logger.info("PHASE 2: FETCH DOWNSTREAM CONNECTIVITY")
    logger.info("="*80)

    # Fetch connectivity
    use_cache = config['cache']['use_cache']
    all_connectivity = extractor.fetch_downstream_connectivity(all_grns, use_cache=use_cache)

    # Filter to sugar GRNs
    sugar_ids = set(sugar_grns['root_id'].astype(int))
    sugar_connectivity = {k: v for k, v in all_connectivity.items() if k in sugar_ids}

    logger.info(f"✓ Extraction complete:")
    logger.info(f"  Sugar GRNs: {len(sugar_connectivity)}")
    logger.info(f"  All GRNs: {len(all_connectivity)}")

    return sugar_grns, all_grns, sugar_connectivity, all_connectivity


def run_analysis(
    config: dict,
    sugar_grns, all_grns,
    sugar_connectivity, all_connectivity
) -> tuple:
    """
    Run Phase 3: Analyze connectivity patterns.

    Args:
        config: Configuration dictionary
        sugar_grns: Sugar GRN DataFrame
        all_grns: All GRN DataFrame
        sugar_connectivity: Sugar connectivity dict
        all_connectivity: All connectivity dict

    Returns:
        Tuple of (sugar_stats, all_stats, comparison_df, partner_df)
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: ANALYZE CONNECTIVITY")
    logger.info("="*80)

    analyzer = GRNConnectivityAnalyzer(config)

    # Compute statistics
    logger.info("Computing statistics for sugar GRNs...")
    sugar_stats = analyzer.compute_connectivity_statistics(sugar_connectivity, sugar_grns)

    logger.info("\nComputing statistics for all GRNs...")
    all_stats = analyzer.compute_connectivity_statistics(all_connectivity, all_grns)

    # Compare sugar vs all
    logger.info("\nComparing sugar GRNs vs all GRNs...")
    comparison_df = analyzer.compare_sugar_vs_all(
        sugar_connectivity, all_connectivity,
        sugar_grns, all_grns
    )

    # Generate partner table
    logger.info("\nGenerating partner tables...")
    top_n = config['analysis']['top_n_partners_display']
    partner_df = analyzer.generate_partner_table(sugar_connectivity, top_n=top_n)

    # Save results
    analyzer.save_statistics(sugar_stats, comparison_df, partner_df, prefix="sugar_")

    logger.info("✓ Analysis complete")

    return sugar_stats, all_stats, comparison_df, partner_df


def run_visualization(
    config: dict,
    sugar_grns, all_grns,
    sugar_connectivity, all_connectivity,
    sugar_stats, comparison_df
) -> list:
    """
    Run Phase 4: Generate visualizations.

    Args:
        config: Configuration dictionary
        sugar_grns: Sugar GRN DataFrame
        all_grns: All GRN DataFrame
        sugar_connectivity: Sugar connectivity dict
        all_connectivity: All connectivity dict
        sugar_stats: Sugar statistics dict
        comparison_df: Comparison DataFrame

    Returns:
        List of generated figure paths
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: GENERATE VISUALIZATIONS")
    logger.info("="*80)

    viz = GRNConnectivityVisualizer(config)

    figure_paths = []

    # Figure 1: GRN summary bar chart
    logger.info("Generating Figure 1: GRN summary...")
    fig1 = viz.plot_grn_summary(sugar_connectivity, sugar_grns)
    figure_paths.append(fig1)

    # Figure 2: Comparison plots
    logger.info("Generating Figure 2: Sugar vs All comparison...")
    fig2 = viz.plot_comparison(sugar_connectivity, all_connectivity, comparison_df)
    figure_paths.append(fig2)

    # Figure 3: Network graph
    logger.info("Generating Figure 3: Network graph...")
    fig3 = viz.plot_network_graph(sugar_connectivity, sugar_grns)
    if fig3:  # May be empty if no nodes
        figure_paths.append(fig3)

    # Figure 4: Heatmap
    logger.info("Generating Figure 4: Connectivity heatmap...")
    top_n = config['analysis']['top_n_partners_display']
    fig4 = viz.plot_heatmap_connectivity(sugar_connectivity, sugar_grns, top_n_partners=top_n)
    figure_paths.append(fig4)

    # Figure 5: Dashboard
    logger.info("Generating Figure 5: Analysis dashboard...")
    fig5 = viz.plot_analysis_dashboard(sugar_connectivity, sugar_stats, sugar_grns)
    figure_paths.append(fig5)

    logger.info(f"✓ Generated {len(figure_paths)} figures")

    return figure_paths


def run_full_analysis(config: dict, demo_mode: bool = False) -> None:
    """
    Run the complete analysis pipeline.

    Args:
        config: Configuration dictionary
        demo_mode: Whether to run in demo mode
    """
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**2  # MB

    # Phase 1 & 2: Extraction
    sugar_grns, all_grns, sugar_connectivity, all_connectivity = run_extraction(config, demo_mode)

    # Phase 3: Analysis
    sugar_stats, all_stats, comparison_df, partner_df = run_analysis(
        config, sugar_grns, all_grns, sugar_connectivity, all_connectivity
    )

    # Phase 4: Visualization
    figure_paths = run_visualization(
        config, sugar_grns, all_grns,
        sugar_connectivity, all_connectivity,
        sugar_stats, comparison_df
    )

    # Final summary
    elapsed = time.time() - start_time
    final_memory = process.memory_info().rss / 1024**2
    peak_memory = process.memory_info().rss / 1024**2

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Sugar GRNs analyzed:           {len(sugar_connectivity)}")
    logger.info(f"All GRNs analyzed:             {len(all_connectivity)}")

    # Count unique partners
    all_partners = set()
    for conn in sugar_connectivity.values():
        all_partners.update(conn['partners'])
    logger.info(f"Total downstream partners:     {len(all_partners)} (unique)")

    logger.info(f"Figures saved:                 {len(figure_paths)}")
    logger.info(f"Execution time:                {elapsed//60:.0f} min {elapsed%60:.0f} sec")
    logger.info(f"Memory peak:                   {peak_memory:.0f} MB")
    logger.info("="*80)
    logger.info(f"Output directory: {config['data']['output_dir']}")
    logger.info("\nKey files:")
    for fig_path in figure_paths:
        logger.info(f"  - {Path(fig_path).name}")
    logger.info("="*80)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='GRN Downstream Connectivity Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with default settings
  python main.py --full-analysis

  # Run in demo mode with small subset
  python main.py --demo-mode

  # Run only extraction (skip analysis and visualization)
  python main.py --extract-only

  # Run only visualization (requires cached data)
  python main.py --visualize-only

  # Use custom config file
  python main.py --full-analysis --config my_config.yaml

  # Run with custom data file
  python main.py --full-analysis --csv-path /path/to/processed_labels.csv.gz
        """
    )

    parser.add_argument(
        '--full-analysis',
        action='store_true',
        help='Run complete pipeline (extract, analyze, visualize)'
    )

    parser.add_argument(
        '--demo-mode',
        action='store_true',
        help='Run in demo mode with small subset of data'
    )

    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Run only extraction phase'
    )

    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Run only visualization phase (requires cached data)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Path to processed_labels.csv.gz (overrides config)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (overrides config)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI args if provided
    if args.csv_path:
        config['data']['processed_labels_path'] = args.csv_path
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    if args.demo_mode:
        config['demo']['enabled'] = True

    # Setup logging
    setup_logging(config, args.log_file)

    # Determine which phases to run
    if args.full_analysis or args.demo_mode:
        run_full_analysis(config, demo_mode=args.demo_mode)

    elif args.extract_only:
        run_extraction(config, demo_mode=args.demo_mode)

    elif args.visualize_only:
        # Load cached data
        cache_dir = Path(config['data']['cache_dir'])
        conn_path = cache_dir / 'downstream_connectivity.pkl'
        sugar_path = cache_dir / 'sugar_grns_metadata.csv'
        all_path = cache_dir / 'all_grns_metadata.csv'

        if not conn_path.exists() or not sugar_path.exists() or not all_path.exists():
            logger.error("Required cache files not found. Run extraction first.")
            sys.exit(1)

        # Load cached data
        logger.info("Loading cached data...")
        with open(conn_path, 'rb') as f:
            all_connectivity = pickle.load(f)

        import pandas as pd
        sugar_grns = pd.read_csv(sugar_path)
        all_grns = pd.read_csv(all_path)

        # Filter connectivity
        sugar_ids = set(sugar_grns['root_id'].astype(int))
        sugar_connectivity = {k: v for k, v in all_connectivity.items() if k in sugar_ids}

        # Run analysis and visualization
        sugar_stats, all_stats, comparison_df, partner_df = run_analysis(
            config, sugar_grns, all_grns, sugar_connectivity, all_connectivity
        )

        run_visualization(
            config, sugar_grns, all_grns,
            sugar_connectivity, all_connectivity,
            sugar_stats, comparison_df
        )

    else:
        parser.print_help()
        logger.error("\nError: Please specify an operation (--full-analysis, --demo-mode, etc.)")
        sys.exit(1)


if __name__ == "__main__":
    main()
