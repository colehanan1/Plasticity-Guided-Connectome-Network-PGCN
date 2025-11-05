"""Quick demo to test OR7a output mapping with a small subset."""

import sys
from pathlib import Path
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the mapper
from scripts.map_or7a_outputs import OR7aOutputMapper

def main():
    print("=" * 80)
    print("OR7a Output Mapping Demo (Small Subset)")
    print("=" * 80)

    # Create a small test dataset (first 5 OR7a neurons)
    or7a_full = pd.read_csv("data/flywire/search_results_or7a.csv")
    or7a_subset = or7a_full.head(5)

    # Save subset
    test_data_path = Path("data/flywire/or7a_test_subset.csv")
    or7a_subset.to_csv(test_data_path, index=False)
    print(f"\n✓ Created test subset with {len(or7a_subset)} neurons")

    # Create mapper with test subset
    mapper = OR7aOutputMapper(
        or7a_data_path=test_data_path,
        data_source='local',
        output_dir=Path('results/or7a_outputs_demo'),
        min_synapses=5,
        top_n_targets=10
    )

    print("\n[1/3] Loading data...")
    or7a_df = mapper.load_or7a_data()
    print(f"  ✓ Loaded {len(or7a_df)} OR7a neurons")
    print(f"  ✓ Root IDs: {or7a_df['root_id'].tolist()}")

    print("\n[2/3] Mapping outputs (this may take a minute)...")
    try:
        outputs_df = mapper.map_all_outputs()
        print(f"  ✓ Found {len(outputs_df)} output connections")
        print(f"  ✓ Unique targets: {outputs_df['target_root_id'].nunique()}")

        # Show sample
        print("\n  Sample outputs:")
        sample = outputs_df.head(10)[['or7a_name', 'target_cell_type', 'target_neuropil', 'synapse_count']]
        print(sample.to_string(index=False))

    except Exception as e:
        print(f"  ✗ Error mapping outputs: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[3/3] Creating summary...")
    try:
        summaries = mapper.generate_summary_statistics()
        print(f"  ✓ Generated {len(summaries)} summary tables")

        # Print overall summary
        print("\n  Overall Statistics:")
        print(summaries['overall'].to_string(index=False))

        print("\n  Top Target Cell Types:")
        print(summaries['target_cell_types'].head(10).to_string(index=False))

    except Exception as e:
        print(f"  ✗ Error generating summaries: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Demo complete! Check results/or7a_outputs_demo/ for outputs")
    print("=" * 80)

if __name__ == "__main__":
    main()
