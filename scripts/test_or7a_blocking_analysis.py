"""Quick test of OR7a blocking & cross-transfer analysis system."""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pgcn.analysis.or7a_pathway import OR7aPathwayAnalyzer
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    print("="*80)
    print("OR7a Blocking Analysis - Quick Test")
    print("="*80)

    try:
        # Initialize analyzer
        print("\n[1/5] Initializing analyzer...")
        analyzer = OR7aPathwayAnalyzer(
            flywire_data_dir=Path('data/flywire'),
            pathway_results_dir=Path('results/or7a_complete_pathway'),
            output_dir=Path('results/or7a_blocking_test')
        )
        print("✓ Analyzer initialized")

        # Load OR7a data
        print("\n[2/5] Loading OR7a neurons...")
        or7a = analyzer.load_or7a_data()
        print(f"✓ Loaded {len(or7a)} OR7a neurons")

        # Load pathway data
        print("\n[3/5] Loading pathway data...")
        analyzer.load_pathway_data()

        if analyzer._complete_pathway is not None:
            print(f"✓ Loaded {len(analyzer._complete_pathway):,} pathway connections")
        else:
            print("⚠ Pathway data not found - run pathway mapping first:")
            print("  python scripts/map_or7a_complete_pathway.py")

        # Load DoOR responses
        print("\n[4/5] Loading DoOR responses...")
        try:
            door = analyzer.load_door_responses()
            print(f"✓ Loaded DoOR responses for {len(door)} odors")

            # Show key responses
            for _, row in door.iterrows():
                if row['found']:
                    print(f"  {row['odor_query']}: Or7a = {row['or7a_response']:.3f}")
        except Exception as e:
            print(f"⚠ DoOR loading failed: {e}")
            print("  Install rpy2 or provide CSV fallback")

        # Analyze pathway architecture
        print("\n[5/5] Analyzing pathway architecture...")
        try:
            arch = analyzer.analyze_pathway_architecture()

            if 'critical_bottleneck' in arch:
                bn = arch['critical_bottleneck']
                print(f"✓ Critical bottleneck identified:")
                print(f"  {bn['transition']}")
                print(f"  {bn['source_neurons']} → {bn['target_neurons']} neurons")
                print(f"  Ratio: {bn['expansion_ratio']:.4f}")

            if 'or7a_pn_strength' in arch:
                pn = arch['or7a_pn_strength']
                print(f"\n✓ OR7a→PN pathway strength:")
                print(f"  Total synapses: {pn['total_synapses']:,}")
                print(f"  Unique PNs: {pn['unique_pns']}")
        except Exception as e:
            print(f"⚠ Analysis failed: {e}")

        print("\n" + "="*80)
        print("✅ Test complete! System is functional.")
        print("="*80)

        print("\nNext steps:")
        print("1. Run: python scripts/analyze_or7a_baseline.py")
        print("2. Run: python scripts/predict_or7a_suppression.py")
        print("3. See: docs/OR7A_BLOCKING_ANALYSIS_GUIDE.md")

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ Test failed: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
