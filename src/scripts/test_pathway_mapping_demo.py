"""Quick demo to test OR7a complete pathway mapping."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.map_or7a_complete_pathway import CompletePathwayMapper

def main():
    print("=" * 80)
    print("OR7a Complete Pathway Mapping Demo")
    print("=" * 80)

    # Create mapper for first 2 levels (OR7a → PN → KC)
    mapper = CompletePathwayMapper(
        or7a_data_path=Path('data/flywire/search_results_or7a.csv'),
        data_source='local',
        output_dir=Path('results/or7a_pathway_demo'),
        min_synapses=5,
        max_levels=3  # OR7a → PN → KC → MBON
    )

    print("\nRunning pathway analysis...")
    print("This will trace: OR7a → DL5_PN → KC → MBON")
    print("(May take 1-2 minutes to load connections)")

    try:
        results = mapper.run_complete_analysis()

        print("\n" + "=" * 80)
        print("DEMO RESULTS")
        print("=" * 80)

        if 'pathway' in results and len(results['pathway']) > 0:
            pathway = results['pathway']
            print(f"\n✓ Traced {len(pathway):,} connections across circuit levels")

            # Show sample
            print("\nSample pathway connections:")
            sample_cols = ['source_level_name', 'target_level_name', 'target_category', 'syn_count']
            print(pathway[sample_cols].head(10).to_string(index=False))

        if 'summaries' in results and 'by_level' in results['summaries']:
            print("\n✓ Circuit level summary:")
            print(results['summaries']['by_level'].to_string(index=False))

        if 'targets' in results and len(results['targets']) > 0:
            print(f"\n✓ Identified {len(results['targets'])} potential targets")
            print("\nTop 5 critical targets:")
            top5 = results['targets'].head(5)[['cell_type', 'category', 'importance_score']]
            print(top5.to_string(index=False))

        print("\n" + "=" * 80)
        print("Demo complete! Check results/or7a_pathway_demo/ for outputs")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
