#!/usr/bin/env python3
"""
Patch script to fix DoOR import and add extended analyses to analyze_or7a_dual_veto.py

This script modifies the main analysis script to:
1. Fix DoOR toolkit import with better error handling
2. Add analyses 5, 6, 7 (serotonin, weighted KC, DP1m)
3. Add supplementary figure generation
"""

import re
from pathlib import Path

def fix_door_import():
    """Fix the DoOR import section with better error handling"""

    new_import = '''# Try to import DoOR toolkit with better error handling
DOOR_AVAILABLE = False
DOOR_ERROR = None

# Try multiple import variations
import_attempts = [
    ("from door import DoOREncoder", "door"),
    ("from door_toolkit import DoOREncoder", "door_toolkit"),
    ("from door.encoder import DoOREncoder", "door"),
]

for import_statement, module_name in import_attempts:
    try:
        exec(import_statement, globals())
        DOOR_AVAILABLE = True
        print(f"✅ DoOR toolkit loaded successfully using: {import_statement}")

        # Verify DoOREncoder is accessible
        if 'DoOREncoder' in globals():
            # Try to instantiate to verify it works
            test_encoder = DoOREncoder()
            print(f"   DoOR module location: {sys.modules[module_name].__file__}")
            del test_encoder
        break
    except ImportError as e:
        DOOR_ERROR = str(e)
        continue
    except Exception as e:
        DOOR_ERROR = f"{type(e).__name__}: {e}"
        continue

if not DOOR_AVAILABLE:
    print("⚠️  DoOR toolkit not available - Analysis 4 will be skipped")
    if DOOR_ERROR:
        print(f"   Last error: {DOOR_ERROR}")
    print("   Install with: pip install door-python-toolkit")
    print("   Or run: python scripts/debug_door_import.py for diagnostics")
'''

    return new_import


def add_extended_analyses_import():
    """Add import for extended analyses module"""

    return '''# Import extended analyses
try:
    from or7a_extended_analyses import (
        analysis_5_serotonin_pathways,
        analysis_6_kc_overlap_weighted,
        analysis_7_dp1m_hub,
        generate_supplementary_figures
    )
    EXTENDED_ANALYSES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Extended analyses module not found: {e}")
    print("   Make sure or7a_extended_analyses.py is in scripts/ directory")
    EXTENDED_ANALYSES_AVAILABLE = False
'''


def add_extended_analyses_to_class():
    """Add methods to call extended analyses"""

    return '''
    def run_extended_analyses(self):
        """Run extended follow-up analyses (5, 6, 7)."""

        if not EXTENDED_ANALYSES_AVAILABLE:
            print("\\n⚠️  Extended analyses not available - skipping")
            return

        print("\\n" + "="*80)
        print("EXTENDED ANALYSES (Serotonin + Refinements)")
        print("="*80)

        # Get cross-glomerular LNs and PNs if not already done
        if 'cross_ln_ids' not in self.results:
            cross_ln_ids, dl5_pns, dm_pns, all_lns = self.get_cross_glomerular_lns()
            self.results['cross_ln_ids'] = cross_ln_ids
            self.results['dl5_pns'] = dl5_pns
            self.results['dm_pns'] = dm_pns
            self.results['all_lns'] = all_lns
        else:
            cross_ln_ids = self.results['cross_ln_ids']
            dl5_pns = self.results['dl5_pns']
            dm_pns = self.results['dm_pns']

        # Analysis 5: Serotonin pathways
        self.results['analysis5'] = analysis_5_serotonin_pathways(
            self.neurons,
            self.connections,
            cross_ln_ids,
            self.output_dir
        )

        # Analysis 6: Weighted KC overlap
        self.results['analysis6'] = analysis_6_kc_overlap_weighted(
            self.connections,
            self.labels,
            dl5_pns,
            dm_pns,
            self.output_dir
        )

        # Analysis 7: DP1m hub
        if self.ln_cross is not None:
            self.results['analysis7'] = analysis_7_dp1m_hub(
                self.ln_cross,
                self.output_dir
            )
        else:
            print("\\n⚠️  Analysis 7 skipped - ln_cross not available")
            self.results['analysis7'] = None
'''


def modify_run_all_analyses():
    """Modify run_all_analyses to include extended analyses"""

    return '''
        # Get cross-glomerular LNs
        cross_ln_ids, dl5_pns, dm_pns, all_lns = self.get_cross_glomerular_lns()

        # Store for extended analyses
        self.results['cross_ln_ids'] = cross_ln_ids
        self.results['dl5_pns'] = dl5_pns
        self.results['dm_pns'] = dm_pns
        self.results['all_lns'] = all_lns

        # Run analyses
        print("\\n" + "="*80)
        print("RUNNING ANALYSES")
        print("="*80)

        self.results['analysis1'] = self.analysis_1_neurotransmitter(cross_ln_ids, all_lns)
        self.results['analysis2'] = self.analysis_2_multihop()
        self.results['analysis3'] = self.analysis_3_kc_overlap(dl5_pns, dm_pns)
        self.results['analysis4'] = self.analysis_4_dose_response()

        # NEW: Extended analyses
        self.run_extended_analyses()

        # Generate figures
        print("\\n" + "="*80)
        print("GENERATING PUBLICATION FIGURES")
        print("="*80)

        self.generate_figure_1(self.results['analysis1'])
        self.generate_figure_2(self.results['analysis2'])
        self.generate_figure_3(self.results['analysis3'])
        self.generate_figure_4(self.results['analysis4'])

        # NEW: Supplementary figures
        if EXTENDED_ANALYSES_AVAILABLE:
            generate_supplementary_figures(self.results, self.output_dir)

        # Write summary
        self.write_summary_report()

        print("\\n" + "="*80)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE (including extended analyses)")
        print("="*80)
        print(f"\\nAll results saved to: {self.output_dir}")
        print("\\nGenerated files:")
        print("  CSV files:")
        print("    - analysis1_neurotransmitter_stats.csv")
        print("    - analysis2_multihop_pathways.csv")
        print("    - analysis3_kc_overlap_stats.csv")
        print("    - analysis3_shared_kcs.csv")
        print("    - analysis4_dose_response_predictions.csv (if DoOR available)")
        print("    - analysis5_serotonin_pathways.csv")
        print("    - analysis6_kc_overlap_weighted.csv")
        print("    - analysis7_dp1m_inputs.csv")
        print("    - analysis7_dp1m_outputs.csv")
        print("  Main Figures:")
        print("    - fig1_neurotransmitter_analysis.png/.pdf")
        print("    - fig2_multihop_pathways.png/.pdf")
        print("    - fig3_kc_overlap_analysis.png/.pdf")
        print("    - fig4_dose_response_model.png/.pdf (if DoOR available)")
        print("  Supplementary Figures:")
        print("    - suppfig1_nt_pathway_targeting.png/.pdf")
        print("    - suppfig2_kc_overlap_threshold.png/.pdf")
        print("    - suppfig3_dp1m_hub_network.png/.pdf")
        print("  Report:")
        print("    - comprehensive_analysis_summary.txt")
'''


def main():
    """Apply all patches"""

    print("="*80)
    print("DOOR FIX AND EXTENDED ANALYSIS INTEGRATION")
    print("="*80)

    script_path = Path(__file__).parent / "analyze_or7a_dual_veto.py"

    if not script_path.exists():
        print(f"❌ Error: {script_path} not found")
        return

    print(f"\\nReading: {script_path}")

    with open(script_path, 'r') as f:
        content = f.read()

    # Backup original
    backup_path = script_path.with_suffix('.py.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backup created: {backup_path}")

    # Apply patches
    print("\\nApplying patches:")

    # 1. Fix DoOR import
    print("  [1/4] Fixing DoOR import...")
    door_pattern = r'# Try to import DoOR toolkit.*?print\("⚠️  DoOR toolkit not available - Analysis 4 will be skipped"\)'
    content = re.sub(door_pattern, fix_door_import(), content, flags=re.DOTALL)

    # 2. Add extended analyses import (after other imports)
    print("  [2/4] Adding extended analyses import...")
    import_marker = 'from data_loaders.neuron_classification import'
    if import_marker in content:
        # Find end of that import block
        marker_pos = content.find(import_marker)
        next_empty_line = content.find('\\n\\n', marker_pos)
        content = content[:next_empty_line] + '\\n' + add_extended_analyses_import() + content[next_empty_line:]

    # 3. Add extended analyses methods to class
    print("  [3/4] Adding extended analysis methods...")
    # Find the run_all_analyses method
    class_pattern = r'(class Or7aDualVetoAnalyzer:.*?def run_all_analyses\(self\):)'

    # Insert before run_all_analyses
    insertion_point = content.find('    def run_all_analyses(self):')
    if insertion_point > 0:
        content = content[:insertion_point] + add_extended_analyses_to_class() + '\\n' + content[insertion_point:]

    # 4. Modify run_all_analyses method
    print("  [4/4] Modifying run_all_analyses method...")

    # Find and replace the main analysis execution section
    old_pattern = r'(# Get cross-glomerular LNs\s+cross_ln_ids.*?self\.write_summary_report\(\))'

    if re.search(old_pattern, content, re.DOTALL):
        content = re.sub(old_pattern, modify_run_all_analyses().strip(), content, flags=re.DOTALL)

    # Write modified content
    print(f"\\nWriting modified script to: {script_path}")
    with open(script_path, 'w') as f:
        f.write(content)

    print("\\n" + "="*80)
    print("✅ PATCHING COMPLETE")
    print("="*80)
    print("\\nChanges made:")
    print("  1. Enhanced DoOR import with better error handling")
    print("  2. Added import for extended analyses module")
    print("  3. Added run_extended_analyses() method to class")
    print("  4. Modified run_all_analyses() to call extended analyses")
    print("\\nNext steps:")
    print("  1. Run: python scripts/debug_door_import.py")
    print("  2. Fix any DoOR import issues identified")
    print("  3. Run: python scripts/analyze_or7a_dual_veto.py")
    print("\\nOriginal script backed up to:")
    print(f"  {backup_path}")


if __name__ == '__main__':
    main()
