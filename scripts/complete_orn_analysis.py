#!/usr/bin/env python3
"""Complete DoOR-FlyWire ORN Pathway Analysis Script.

This script performs comprehensive olfactory receptor neuron (ORN) identification
and pathway analysis using DoOR database integration with FlyWire connectome labels.

Usage
-----
Basic analysis:
    python scripts/complete_orn_analysis.py

Specify custom FlyWire labels file:
    python scripts/complete_orn_analysis.py --labels data/processed_labels.csv.gz

Full analysis with connectivity:
    python scripts/complete_orn_analysis.py --labels data/processed_labels.csv.gz \\
        --connectivity data/connectivity.csv --output results/orn_analysis/

Features
--------
- DoOR database integration (693 odorants × 78 receptors)
- FlyWire ORN identification (>500 expected cells)
- Or42b and Or47b pathway analysis
- Behavioral predictions
- Comprehensive reporting

Example
-------
>>> python scripts/complete_orn_analysis.py --labels processed_labels.csv.gz
Initializing DoOR database...
Loading FlyWire labels...
Identifying olfactory cells...
Found 523 olfactory cells
DoOR validated: 487 cells
Analyzing Or42b pathway...
Generating comprehensive report...
Results saved to: results/orn_analysis/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pgcn.door import DoORDataManager, FlyWireORNIdentifier, ORNBehaviorPathwayAnalyzer


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for analysis script.

    Parameters
    ----------
    verbose : bool
        Enable DEBUG level logging

    Returns
    -------
    logging.Logger
        Configured logger
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Complete DoOR-FlyWire ORN pathway analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input files
    parser.add_argument(
        '--labels',
        type=str,
        default='data/processed_labels.csv.gz',
        help='Path to FlyWire processed_labels.csv.gz file'
    )

    parser.add_argument(
        '--connectivity',
        type=str,
        default=None,
        help='Path to FlyWire connectivity CSV file (optional)'
    )

    # DoOR configuration
    parser.add_argument(
        '--door-method',
        type=str,
        choices=['rpy2', 'csv', 'zenodo'],
        default='rpy2',
        help='DoOR data access method (default: rpy2)'
    )

    parser.add_argument(
        '--door-csv',
        type=str,
        default=None,
        help='Path to backup DoOR CSV file (for csv method)'
    )

    # Output configuration
    parser.add_argument(
        '--output',
        type=str,
        default='results/orn_analysis',
        help='Output directory for results'
    )

    # Analysis options
    parser.add_argument(
        '--max-cells',
        type=int,
        default=None,
        help='Maximum cells to process (for testing)'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Minimum confidence for ORN identification (default: 0.5)'
    )

    # Execution options
    parser.add_argument(
        '--install-door',
        action='store_true',
        help='Install DoOR R packages before analysis'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main analysis pipeline."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("Complete DoOR-FlyWire ORN Pathway Analysis")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Initialize DoOR database
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Initializing DoOR Database")
    logger.info("=" * 80)

    try:
        door_manager = DoORDataManager(
            method=args.door_method,
            backup_csv_path=args.door_csv
        )

        # Install DoOR packages if requested
        if args.install_door:
            logger.info("Installing DoOR R packages...")
            success = door_manager.install_door_packages()
            if not success:
                logger.error("Failed to install DoOR packages")
                return 1

        # Load DoOR data
        logger.info("Loading DoOR database...")
        door_data = door_manager.load_door_data()

        response_matrix = door_data['response_matrix']
        logger.info(f"DoOR data loaded: {response_matrix.shape}")
        logger.info(f"Odorants: {len(response_matrix.index)}")
        logger.info(f"Receptors: {len(response_matrix.columns)}")

        # Get receptor list
        receptors = door_manager.get_complete_receptor_list()
        logger.info(f"Receptor breakdown:")
        logger.info(f"  OR receptors: {len(receptors['or_receptors'])}")
        logger.info(f"  IR receptors: {len(receptors['ir_receptors'])}")
        logger.info(f"  GR receptors: {len(receptors['gr_receptors'])}")
        logger.info(f"  Sensilla: {len(receptors['sensillum_types'])}")

    except Exception as e:
        logger.error(f"Failed to initialize DoOR database: {e}")
        return 1

    # Step 2: Identify olfactory cells from FlyWire labels
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Identifying Olfactory Cells from FlyWire Labels")
    logger.info("=" * 80)

    try:
        # Check labels file exists
        labels_path = Path(args.labels)
        if not labels_path.exists():
            logger.error(f"Labels file not found: {labels_path}")
            logger.info("Please download FlyWire community labels:")
            logger.info("  wget https://codex.flywire.ai/api/download?dataset=fafb -O processed_labels.csv.gz")
            return 1

        # Create identifier
        identifier = FlyWireORNIdentifier(
            door_manager=door_manager,
            confidence_threshold=args.confidence_threshold
        )

        # Identify olfactory cells
        logger.info(f"Processing FlyWire labels: {labels_path}")
        logger.info(f"Confidence threshold: {args.confidence_threshold}")

        identified_cells = identifier.identify_olfactory_cells(
            str(labels_path),
            max_cells=args.max_cells
        )

        logger.info(f"\nIdentification complete:")
        logger.info(f"  Total olfactory cells: {len(identified_cells)}")
        logger.info(f"  DoOR validated: {identified_cells['door_match'].notna().sum()}")
        logger.info(f"  Mean confidence: {identified_cells['confidence_score'].mean():.3f}")

        # Save identified cells
        cells_output = output_dir / 'identified_olfactory_cells.csv'
        identified_cells.to_csv(cells_output, index=False)
        logger.info(f"  Saved to: {cells_output}")

        # Print receptor breakdown
        logger.info("\nReceptor type breakdown:")
        receptor_counts = identified_cells['receptor_type'].value_counts()
        for receptor_type, count in receptor_counts.items():
            logger.info(f"  {receptor_type}: {count}")

    except Exception as e:
        logger.error(f"Failed to identify olfactory cells: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Load connectivity data (if available)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Loading Connectivity Data")
    logger.info("=" * 80)

    connectivity_data = None
    if args.connectivity:
        try:
            connectivity_path = Path(args.connectivity)
            if connectivity_path.exists():
                logger.info(f"Loading connectivity: {connectivity_path}")
                connectivity_data = pd.read_csv(connectivity_path)
                logger.info(f"  Connections: {len(connectivity_data)}")
            else:
                logger.warning(f"Connectivity file not found: {connectivity_path}")
        except Exception as e:
            logger.warning(f"Failed to load connectivity: {e}")
    else:
        logger.info("No connectivity data provided (use --connectivity)")

    # Step 4: Analyze ORN pathways
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Analyzing ORN→Behavior Pathways")
    logger.info("=" * 80)

    try:
        analyzer = ORNBehaviorPathwayAnalyzer(
            door_manager=door_manager,
            identified_cells=identified_cells,
            connectivity_data=connectivity_data
        )

        # Analyze Or42b pathway
        logger.info("\nAnalyzing Or42b pathway...")
        or42b_pathway = analyzer.analyze_or42b_pathway()

        if 'error' not in or42b_pathway:
            logger.info(f"  Or42b cells found: {or42b_pathway['n_cells']}")
            logger.info("  Top ligands:")
            for ligand, response in list(or42b_pathway['best_ligands'].items())[:5]:
                logger.info(f"    {ligand}: {response:.3f}")

        # Analyze Or47b-hexanol pathway
        logger.info("\nAnalyzing Or47b-hexanol pathway...")
        or47b_pathway = analyzer.analyze_or47b_hexanol_pathway()

        if 'error' not in or47b_pathway:
            logger.info(f"  Or47b cells found: {or47b_pathway['n_cells']}")
            if 'feeding_prediction' in or47b_pathway:
                feeding_prob = or47b_pathway['feeding_prediction']['feeding_probability']
                logger.info(f"  Feeding probability: {feeding_prob:.2%}")

        # Generate comprehensive report
        logger.info("\nGenerating comprehensive report...")
        report = analyzer.generate_comprehensive_report()

        # Save report
        report_output = output_dir / 'pathway_analysis_report.json'
        with open(report_output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"  Report saved to: {report_output}")

    except Exception as e:
        logger.error(f"Failed to analyze pathways: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 5: Generate summary visualizations
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Generating Summary")
    logger.info("=" * 80)

    # Create summary statistics
    summary = {
        'analysis_parameters': {
            'labels_file': str(args.labels),
            'door_method': args.door_method,
            'confidence_threshold': args.confidence_threshold,
            'max_cells': args.max_cells
        },
        'door_database': {
            'n_odorants': len(response_matrix.index),
            'n_receptors': len(response_matrix.columns),
            'or_receptors': len(receptors['or_receptors']),
            'ir_receptors': len(receptors['ir_receptors']),
            'gr_receptors': len(receptors['gr_receptors'])
        },
        'identification_results': {
            'total_cells': len(identified_cells),
            'door_validated': int(identified_cells['door_match'].notna().sum()),
            'mean_confidence': float(identified_cells['confidence_score'].mean()),
            'receptor_types': identified_cells['receptor_type'].value_counts().to_dict()
        },
        'pathway_analyses': {
            'or42b_cells': or42b_pathway.get('n_cells', 0),
            'or47b_cells': or47b_pathway.get('n_cells', 0)
        }
    }

    # Save summary
    summary_output = output_dir / 'analysis_summary.json'
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Analysis summary saved to: {summary_output}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total olfactory cells identified: {len(identified_cells)}")
    logger.info(f"DoOR-validated cells: {identified_cells['door_match'].notna().sum()}")
    logger.info(f"Or42b cells: {or42b_pathway.get('n_cells', 0)}")
    logger.info(f"Or47b cells: {or47b_pathway.get('n_cells', 0)}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
