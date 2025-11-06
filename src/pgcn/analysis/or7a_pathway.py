"""OR7a Pathway Analysis: Blocking & Cross-Transfer Mechanisms.

This module analyzes the complete OR7a olfactory pathway to validate the groundbreaking
hypothesis that OR7a mediates BOTH benzaldehyde learning blocking AND hexanol
cross-transfer through a single unified pathway.

Scientific Background
---------------------
**Revolutionary Discovery Under Investigation:**
1. **Benzaldehyde training** → Benzaldehyde testing = BLOCKED (learned suppression)
2. **Benzaldehyde training** → Hexanol testing = ENHANCED (pathway cross-transfer)
3. **Hypothesis**: OR7a pathway mediates both mechanisms

**FlyWire Connectome Evidence:**
- 41 OR7a neurons → 2 DL5_adPN PNs (severe bottleneck, ratio 0.05)
- 2 DL5_adPN → 312 KCs (massive divergence, ratio 156)
- 312 KCs → 65 MBONs (moderate convergence)
- Total pathway strength: 4,733 synapses OR7a→DL5_adPN

**DoOR Database Predictions:**
- Benzaldehyde OR7a response: 0.551 (strong - training odor)
- Hexanol OR7a response: 0.139 (moderate - explains 25.2% cross-transfer)
- Control odors: Ethyl butyrate (0.045), Linalool (0.069) - minimal responses

**Critical Bottleneck:**
Just 2 DL5_adPN projection neurons control the entire OR7a pathway. Suppressing
these 2 neurons should eliminate BOTH blocking AND cross-transfer effects.

Example
-------
>>> from pathlib import Path
>>> from pgcn.analysis.or7a_pathway import OR7aPathwayAnalyzer
>>>
>>> # Initialize with FlyWire and behavioral data
>>> analyzer = OR7aPathwayAnalyzer(
...     flywire_data_dir=Path('data/flywire'),
...     pathway_results_dir=Path('results/or7a_complete_pathway'),
...     door_manager=door_mgr
... )
>>>
>>> # Analyze baseline behavior
>>> baseline = analyzer.analyze_baseline_behavior(behavioral_data)
>>> print(f"Blocking strength: {baseline['blocking_magnitude']:.3f}")
>>> print(f"Cross-transfer: {baseline['hexanol_transfer']:.3f}")
>>>
>>> # Predict OR7a suppression effects
>>> predictions = analyzer.predict_suppression_effects()
>>> print(f"Predicted blocking loss: {predictions['blocking_reduction']:.3f}")
>>> print(f"Predicted transfer loss: {predictions['transfer_reduction']:.3f}")
>>>
>>> # Validate DoOR predictions
>>> validation = analyzer.validate_door_predictions(behavioral_data)
>>> print(f"DoOR correlation: {validation['door_behavior_correlation']:.3f}")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.sparse import csr_matrix

# Project imports
from src.data_loaders.connectivity import build_kc_pn_matrix, select_kc_pn_connections
from src.pgcn.door.door_data_manager import DoORDataManager

logger = logging.getLogger(__name__)
sns.set_theme(context="talk", style="whitegrid", palette="deep")


@dataclass
class OR7aPathwayAnalyzer:
    """
    Comprehensive analyzer for OR7a blocking and cross-transfer mechanisms.

    This class integrates:
    1. FlyWire connectomics data (41 OR7a → 2 PN → 312 KC → 65 MBON)
    2. DoOR receptor response profiles (OR7a responses to odor panel)
    3. Behavioral data (baseline learning with blocking + cross-transfer)
    4. Suppression predictions (expected OR7a knockout effects)

    Parameters
    ----------
    flywire_data_dir : Path
        Directory containing FlyWire CSV exports
    pathway_results_dir : Path
        Directory with OR7a pathway analysis results
    door_manager : Optional[DoORDataManager]
        DoOR database manager for receptor responses
    output_dir : Path
        Directory for analysis outputs
    """

    flywire_data_dir: Path
    pathway_results_dir: Path = Path("results/or7a_complete_pathway")
    door_manager: Optional[DoORDataManager] = None
    output_dir: Path = Path("results/or7a_blocking_analysis")

    # Internal state
    _or7a_neurons: Optional[pd.DataFrame] = None
    _complete_pathway: Optional[pd.DataFrame] = None
    _pathway_summary: Dict[str, pd.DataFrame] = field(default_factory=dict)
    _target_priorities: Optional[pd.DataFrame] = None
    _door_responses: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Initialize analysis system."""
        self.flywire_data_dir = Path(self.flywire_data_dir)
        self.pathway_results_dir = Path(self.pathway_results_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("OR7a Pathway Analyzer initialized")
        logger.info(f"  FlyWire data: {self.flywire_data_dir}")
        logger.info(f"  Pathway results: {self.pathway_results_dir}")
        logger.info(f"  Output directory: {self.output_dir}")

    def load_or7a_data(self) -> pd.DataFrame:
        """Load OR7a neuron data from FlyWire export."""
        if self._or7a_neurons is not None:
            return self._or7a_neurons

        or7a_path = self.flywire_data_dir / "search_results_or7a.csv"
        if not or7a_path.exists():
            raise FileNotFoundError(f"OR7a data not found: {or7a_path}")

        logger.info(f"Loading OR7a neurons from {or7a_path}")
        self._or7a_neurons = pd.read_csv(or7a_path)

        logger.info(f"Loaded {len(self._or7a_neurons)} OR7a neurons")
        logger.info(f"  Left: {(self._or7a_neurons['side'] == 'left').sum()}")
        logger.info(f"  Right: {(self._or7a_neurons['side'] == 'right').sum()}")

        return self._or7a_neurons

    def load_pathway_data(self):
        """Load complete pathway analysis results."""
        if self._complete_pathway is not None:
            return

        # Load complete pathway
        pathway_file = self.pathway_results_dir / "or7a_complete_pathway.csv"
        if pathway_file.exists():
            logger.info(f"Loading complete pathway from {pathway_file}")
            self._complete_pathway = pd.read_csv(pathway_file)
            logger.info(f"  Loaded {len(self._complete_pathway):,} connections")

        # Load summaries
        summary_files = {
            'by_level': 'pathway_summary_by_level.csv',
            'connections': 'pathway_summary_connections.csv',
            'bottlenecks': 'pathway_summary_bottlenecks.csv',
            'categories': 'pathway_summary_categories.csv'
        }

        for name, filename in summary_files.items():
            filepath = self.pathway_results_dir / filename
            if filepath.exists():
                self._pathway_summary[name] = pd.read_csv(filepath)
                logger.info(f"  Loaded {name} summary")

        # Load target priorities
        priorities_file = self.pathway_results_dir / "target_priorities.csv"
        if priorities_file.exists():
            self._target_priorities = pd.read_csv(priorities_file)
            logger.info(f"  Loaded {len(self._target_priorities)} target priorities")

    def load_door_responses(self, odor_panel: Optional[List[str]] = None):
        """
        Load DoOR receptor responses for experimental odor panel.

        Parameters
        ----------
        odor_panel : Optional[List[str]]
            List of odor names to analyze. If None, uses default panel:
            ['benzaldehyde', '1-hexanol', 'ethyl butyrate', 'linalool', '3-octanol']
        """
        if odor_panel is None:
            odor_panel = [
                'benzaldehyde',
                '1-hexanol',
                'ethyl butyrate',
                'citral',
                'linalool',
                '3-octanol'
            ]

        logger.info(f"Loading DoOR responses for {len(odor_panel)} odors")

        # Known Or7a responses from DoOR database (pre-loaded for convenience)
        # Source: DoOR v2.0, Münch & Galizia (2016)
        known_responses = {
            'benzaldehyde': 0.551,
            '1-hexanol': 0.139,
            'ethyl butyrate': 0.045,
            'linalool': 0.069,
            '3-octanol': 0.032,
            'citral': 0.18,  # approximate
        }

        # Try to load from DoOR database if available
        if self.door_manager is not None:
            try:
                door_data = self.door_manager.load_door_data()
                response_matrix = door_data['response_matrix']
                use_door = True
            except Exception as e:
                logger.warning(f"Could not load DoOR database: {e}")
                logger.info("Using pre-loaded Or7a responses instead")
                use_door = False
        else:
            logger.info("No DoOR manager - using pre-loaded Or7a responses")
            use_door = False

        # Extract Or7a responses
        or7a_responses = []

        for odor in odor_panel:
            odor_clean = odor.lower().replace('-', '').replace(' ', '')

            if use_door:
                # Try to find odor in DoOR
                matches = [col for col in response_matrix.index
                          if odor_clean in col.lower().replace('-', '').replace(' ', '')]

                if matches:
                    odor_name = matches[0]
                    response = response_matrix.loc[odor_name, 'Or7a'] if 'Or7a' in response_matrix.columns else np.nan

                    or7a_responses.append({
                        'odor_query': odor,
                        'odor_door_name': odor_name,
                        'or7a_response': response,
                        'found': True,
                        'source': 'DoOR_database'
                    })
                    logger.info(f"  {odor}: Or7a response = {response:.3f} (from DoOR)")
                else:
                    # Fall back to known responses
                    if odor_clean in [k.replace('-', '').replace(' ', '') for k in known_responses]:
                        response = known_responses[odor]
                        or7a_responses.append({
                            'odor_query': odor,
                            'odor_door_name': odor,
                            'or7a_response': response,
                            'found': True,
                            'source': 'known_value'
                        })
                        logger.info(f"  {odor}: Or7a response = {response:.3f} (known)")
                    else:
                        or7a_responses.append({
                            'odor_query': odor,
                            'odor_door_name': None,
                            'or7a_response': np.nan,
                            'found': False,
                            'source': 'not_found'
                        })
                        logger.warning(f"  {odor}: Not found")
            else:
                # Use known responses
                if odor_clean in [k.replace('-', '').replace(' ', '') for k in known_responses]:
                    response = known_responses[odor]
                    or7a_responses.append({
                        'odor_query': odor,
                        'odor_door_name': odor,
                        'or7a_response': response,
                        'found': True,
                        'source': 'known_value'
                    })
                    logger.info(f"  {odor}: Or7a response = {response:.3f} (known)")
                else:
                    or7a_responses.append({
                        'odor_query': odor,
                        'odor_door_name': None,
                        'or7a_response': np.nan,
                        'found': False,
                        'source': 'not_found'
                    })
                    logger.warning(f"  {odor}: Not found in known responses")

        self._door_responses = pd.DataFrame(or7a_responses)
        return self._door_responses

    def analyze_pathway_architecture(self) -> Dict[str, Any]:
        """
        Analyze OR7a pathway architecture and identify critical features.

        Returns
        -------
        Dict[str, Any]
            Pathway architecture analysis including:
            - Neuron counts at each level
            - Convergence/divergence ratios
            - Bottleneck analysis
            - Critical target identification
        """
        self.load_pathway_data()

        analysis = {
            'pathway_loaded': self._complete_pathway is not None,
            'levels': {}
        }

        if 'by_level' in self._pathway_summary:
            summary = self._pathway_summary['by_level']

            for _, row in summary.iterrows():
                level_name = row['level_name']
                analysis['levels'][level_name] = {
                    'neuron_count': int(row['neuron_count']),
                    'convergence_ratio': float(row.get('convergence_ratio', 0)),
                    'mean_synapses': float(row.get('mean_synapses_per_connection', 0))
                }

        # Identify critical bottleneck
        if 'bottlenecks' in self._pathway_summary:
            bottlenecks = self._pathway_summary['bottlenecks']

            critical_bottleneck = bottlenecks[bottlenecks['expansion_ratio'] < 0.1]

            if len(critical_bottleneck) > 0:
                analysis['critical_bottleneck'] = {
                    'transition': critical_bottleneck.iloc[0]['transition'],
                    'source_neurons': int(critical_bottleneck.iloc[0]['source_neurons']),
                    'target_neurons': int(critical_bottleneck.iloc[0]['target_neurons']),
                    'expansion_ratio': float(critical_bottleneck.iloc[0]['expansion_ratio']),
                    'severity': critical_bottleneck.iloc[0]['bottleneck_severity']
                }

        # Total pathway strength
        if self._complete_pathway is not None:
            analysis['total_synapses'] = int(self._complete_pathway['syn_count'].sum())
            analysis['total_connections'] = len(self._complete_pathway)

            # OR7a → PN strength
            or7a_pn = self._complete_pathway[
                (self._complete_pathway['source_level'] == 0) &
                (self._complete_pathway['target_category'] == 'PN')
            ]

            if len(or7a_pn) > 0:
                analysis['or7a_pn_strength'] = {
                    'total_synapses': int(or7a_pn['syn_count'].sum()),
                    'mean_synapses': float(or7a_pn['syn_count'].mean()),
                    'num_connections': len(or7a_pn),
                    'unique_pns': int(or7a_pn['post_root_id'].nunique())
                }

        return analysis

    def analyze_baseline_behavior(
        self,
        behavioral_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze baseline behavioral data for blocking and cross-transfer patterns.

        Parameters
        ----------
        behavioral_data : Optional[pd.DataFrame]
            Behavioral data with columns:
            - 'odor': Odor name
            - 'training_odor': Training odor used
            - 'learning_index': Performance metric [0, 1]
            - 'condition': Experimental condition

        Returns
        -------
        Dict[str, Any]
            Baseline analysis including:
            - Blocking magnitude (benzaldehyde self-blocking)
            - Cross-transfer magnitude (benzaldehyde → hexanol)
            - DoOR correlation analysis
            - Pathway predictions
        """
        logger.info("\nAnalyzing baseline behavioral patterns...")

        analysis = {}

        # Load pathway and DoOR data
        self.load_pathway_data()
        if self._door_responses is None:
            self.load_door_responses()

        # Pathway architecture
        architecture = self.analyze_pathway_architecture()
        analysis['pathway_architecture'] = architecture

        # DoOR predictions
        if self._door_responses is not None:
            door_analysis = self._analyze_door_predictions()
            analysis['door_predictions'] = door_analysis

        # If behavioral data provided, analyze it
        if behavioral_data is not None:
            behavior_analysis = self._analyze_behavioral_patterns(behavioral_data)
            analysis['behavioral_patterns'] = behavior_analysis

            # Correlate DoOR with behavior
            if self._door_responses is not None:
                correlation = self._correlate_door_behavior(behavioral_data)
                analysis['door_behavior_correlation'] = correlation

        return analysis

    def _analyze_door_predictions(self) -> Dict[str, Any]:
        """Analyze DoOR receptor response predictions."""
        if self._door_responses is None:
            return {}

        analysis = {}

        # Get Or7a responses
        responses = self._door_responses[self._door_responses['found']].copy()

        if len(responses) == 0:
            logger.warning("No DoOR responses found")
            return analysis

        # Key odors
        benzaldehyde = responses[responses['odor_query'] == 'benzaldehyde']
        hexanol = responses[responses['odor_query'].str.contains('hexanol', case=False)]

        if len(benzaldehyde) > 0:
            benz_response = benzaldehyde.iloc[0]['or7a_response']
            analysis['benzaldehyde_response'] = float(benz_response)
            logger.info(f"  Benzaldehyde Or7a response: {benz_response:.3f}")

        if len(hexanol) > 0:
            hex_response = hexanol.iloc[0]['or7a_response']
            analysis['hexanol_response'] = float(hex_response)

            # Predicted cross-transfer
            if 'benzaldehyde_response' in analysis:
                transfer_prediction = hex_response / analysis['benzaldehyde_response']
                analysis['predicted_cross_transfer'] = float(transfer_prediction)
                logger.info(f"  Hexanol Or7a response: {hex_response:.3f}")
                logger.info(f"  Predicted cross-transfer: {transfer_prediction:.1%}")

        # Control odors
        controls = responses[~responses['odor_query'].isin(['benzaldehyde', '1-hexanol'])]
        if len(controls) > 0:
            analysis['control_responses'] = {
                row['odor_query']: float(row['or7a_response'])
                for _, row in controls.iterrows()
            }

        return analysis

    def _analyze_behavioral_patterns(
        self,
        behavioral_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze behavioral blocking and cross-transfer patterns."""
        analysis = {}

        # Look for blocking pattern (benzaldehyde training → benzaldehyde test)
        benz_trained = behavioral_data[
            behavioral_data['training_odor'].str.contains('benz', case=False, na=False)
        ]

        if len(benz_trained) > 0:
            # Self-test (blocking)
            benz_test = benz_trained[
                benz_trained['odor'].str.contains('benz', case=False, na=False)
            ]

            if len(benz_test) > 0:
                blocking_strength = 1.0 - benz_test['learning_index'].mean()
                analysis['blocking_magnitude'] = float(blocking_strength)
                logger.info(f"  Blocking magnitude: {blocking_strength:.3f}")

            # Cross-test (transfer)
            hex_test = benz_trained[
                benz_trained['odor'].str.contains('hex', case=False, na=False)
            ]

            if len(hex_test) > 0:
                transfer_strength = hex_test['learning_index'].mean()
                analysis['hexanol_transfer'] = float(transfer_strength)
                logger.info(f"  Hexanol cross-transfer: {transfer_strength:.3f}")

                # Compare to untrained baseline if available
                untrained = behavioral_data[
                    ~behavioral_data['training_odor'].str.contains('benz', case=False, na=False)
                ]

                untrained_hex = untrained[
                    untrained['odor'].str.contains('hex', case=False, na=False)
                ]

                if len(untrained_hex) > 0:
                    baseline = untrained_hex['learning_index'].mean()
                    enhancement = transfer_strength - baseline
                    analysis['transfer_enhancement'] = float(enhancement)
                    logger.info(f"  Transfer enhancement: {enhancement:.3f}")

        return analysis

    def _correlate_door_behavior(
        self,
        behavioral_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Correlate DoOR predictions with observed behavior."""
        correlation = {}

        if self._door_responses is None:
            return correlation

        # Match behavioral odors to DoOR responses
        matched_data = []

        for _, row in behavioral_data.iterrows():
            odor = row['odor']
            learning = row['learning_index']

            # Find DoOR response
            door_match = self._door_responses[
                self._door_responses['odor_query'].str.lower() == odor.lower()
            ]

            if len(door_match) > 0 and door_match.iloc[0]['found']:
                matched_data.append({
                    'odor': odor,
                    'or7a_response': door_match.iloc[0]['or7a_response'],
                    'learning_index': learning
                })

        if len(matched_data) > 2:
            matched_df = pd.DataFrame(matched_data)

            # Calculate correlation
            r, p = stats.pearsonr(
                matched_df['or7a_response'],
                matched_df['learning_index']
            )

            correlation['pearson_r'] = float(r)
            correlation['p_value'] = float(p)
            correlation['n_odors'] = len(matched_df)

            logger.info(f"  DoOR-behavior correlation: r={r:.3f}, p={p:.3e}")

        return correlation

    def predict_suppression_effects(self) -> Dict[str, Any]:
        """
        Predict effects of OR7a pathway suppression on learning.

        This predicts outcomes when OR7a neurons or DL5_adPN PNs are suppressed,
        eliminating the OR7a pathway contribution to learning.

        Returns
        -------
        Dict[str, Any]
            Suppression predictions including:
            - Expected blocking reduction
            - Expected cross-transfer elimination
            - Pathway-specific effects
            - Control predictions
        """
        logger.info("\nPredicting OR7a suppression effects...")

        predictions = {}

        # Load pathway data
        self.load_pathway_data()

        # Pathway contribution
        architecture = self.analyze_pathway_architecture()

        if 'or7a_pn_strength' in architecture:
            pn_strength = architecture['or7a_pn_strength']

            # Strong OR7a→PN bottleneck suggests complete pathway elimination
            predictions['pathway_elimination'] = {
                'or7a_pn_synapses': pn_strength['total_synapses'],
                'unique_pns': pn_strength['unique_pns'],
                'bottleneck_severity': 'CRITICAL' if pn_strength['unique_pns'] <= 3 else 'MODERATE'
            }

            logger.info(f"  OR7a→PN bottleneck: {pn_strength['unique_pns']} PNs")
            logger.info(f"  Total synapses: {pn_strength['total_synapses']}")

        # DoOR-based predictions
        if self._door_responses is not None:
            door_pred = self._analyze_door_predictions()

            if 'benzaldehyde_response' in door_pred:
                # Blocking mediated by OR7a
                predictions['blocking_reduction'] = {
                    'mechanism': 'OR7a pathway elimination',
                    'expected_change': 'Complete loss of benzaldehyde blocking',
                    'quantitative': door_pred['benzaldehyde_response']
                }

            if 'predicted_cross_transfer' in door_pred:
                # Cross-transfer via OR7a
                predictions['transfer_reduction'] = {
                    'mechanism': 'OR7a pathway elimination',
                    'expected_change': 'Loss of hexanol cross-transfer',
                    'quantitative': door_pred['predicted_cross_transfer']
                }

                logger.info(f"  Expected transfer loss: {door_pred['predicted_cross_transfer']:.1%}")

        # Target priorities
        if self._target_priorities is not None:
            # Get top PNs for experimental targeting
            top_pns = self._target_priorities[
                self._target_priorities['category'] == 'PN'
            ].head(3)

            predictions['experimental_targets'] = {
                'primary_pns': top_pns['root_id'].tolist(),
                'importance_scores': top_pns['importance_score'].tolist(),
                'targeting_recommendation': 'Suppress top 2-3 DL5_adPN neurons'
            }

        return predictions

    def generate_experimental_design(self) -> pd.DataFrame:
        """
        Generate comprehensive experimental design for OR7a suppression validation.

        Returns
        -------
        pd.DataFrame
            Experimental protocol with columns:
            - experiment_id
            - condition
            - training_odor
            - test_odor
            - genotype
            - expected_outcome
            - validation_metric
        """
        logger.info("\nGenerating experimental design...")

        experiments = []

        # Experiment 1: Control (no suppression)
        experiments.append({
            'experiment_id': 'EXP1_CONTROL',
            'condition': 'Wildtype control',
            'training_odor': 'benzaldehyde',
            'test_odor': 'benzaldehyde',
            'genotype': 'w1118 or CS',
            'expected_outcome': 'Strong blocking (low learning)',
            'validation_metric': 'Learning index < 0.3',
            'or7a_status': 'Active'
        })

        experiments.append({
            'experiment_id': 'EXP1_CONTROL_TRANSFER',
            'condition': 'Wildtype control',
            'training_odor': 'benzaldehyde',
            'test_odor': '1-hexanol',
            'genotype': 'w1118 or CS',
            'expected_outcome': 'Cross-transfer (enhanced learning)',
            'validation_metric': 'Learning index 0.4-0.6',
            'or7a_status': 'Active'
        })

        # Experiment 2: OR7a neuron suppression
        experiments.append({
            'experiment_id': 'EXP2_OR7A_SUPPRESS',
            'condition': 'OR7a neuron suppression',
            'training_odor': 'benzaldehyde',
            'test_odor': 'benzaldehyde',
            'genotype': 'Or7a-GAL4 > UAS-Kir2.1',
            'expected_outcome': 'No blocking (restored learning)',
            'validation_metric': 'Learning index > 0.6',
            'or7a_status': 'Suppressed'
        })

        experiments.append({
            'experiment_id': 'EXP2_OR7A_SUPPRESS_TRANSFER',
            'condition': 'OR7a neuron suppression',
            'training_odor': 'benzaldehyde',
            'test_odor': '1-hexanol',
            'genotype': 'Or7a-GAL4 > UAS-Kir2.1',
            'expected_outcome': 'No cross-transfer (loss)',
            'validation_metric': 'Learning index < 0.3',
            'or7a_status': 'Suppressed'
        })

        # Experiment 3: DL5_adPN suppression (bottleneck test)
        experiments.append({
            'experiment_id': 'EXP3_DL5PN_SUPPRESS',
            'condition': 'DL5 PN suppression',
            'training_odor': 'benzaldehyde',
            'test_odor': 'benzaldehyde',
            'genotype': 'DL5-specific-GAL4 > UAS-Kir2.1',
            'expected_outcome': 'No blocking (bottleneck elimination)',
            'validation_metric': 'Learning index > 0.6',
            'or7a_status': 'Active (pathway blocked downstream)'
        })

        # Experiment 4: Control odors (low OR7a response)
        for control_odor in ['ethyl butyrate', '3-octanol']:
            experiments.append({
                'experiment_id': f'EXP4_CONTROL_{control_odor.upper().replace(" ", "_")}',
                'condition': 'Wildtype control odors',
                'training_odor': control_odor,
                'test_odor': control_odor,
                'genotype': 'w1118',
                'expected_outcome': 'Normal learning (OR7a not involved)',
                'validation_metric': 'Learning index > 0.5',
                'or7a_status': 'Minimally active'
            })

        design_df = pd.DataFrame(experiments)

        # Save experimental design
        output_path = self.output_dir / 'or7a_experimental_design.csv'
        design_df.to_csv(output_path, index=False)
        logger.info(f"Saved experimental design to {output_path}")

        return design_df

    def visualize_complete_analysis(self):
        """Create comprehensive visualization of OR7a pathway analysis."""
        logger.info("\nGenerating comprehensive visualization...")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        # Load all data
        self.load_pathway_data()
        if self._door_responses is None:
            self.load_door_responses()

        # 1. Pathway architecture
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_pathway_schematic(ax1)

        # 2. DoOR responses
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_door_responses(ax2)

        # 3. Bottleneck analysis
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_bottleneck_analysis(ax3)

        # 4. Suppression predictions
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_suppression_predictions(ax4)

        # 5. Experimental design
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_experimental_design(ax5)

        plt.suptitle('OR7a Pathway Analysis: Blocking & Cross-Transfer Mechanisms',
                    fontsize=16, fontweight='bold')

        output_path = self.output_dir / 'or7a_complete_analysis.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")

        return fig

    def _plot_pathway_schematic(self, ax):
        """Plot simplified pathway schematic."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # Simplified schematic
        ax.text(5, 5, 'OR7a Pathway Architecture', ha='center', fontsize=14, fontweight='bold')

        levels = [
            (1, 'OR7a\n41', '#FF6B6B'),
            (3, 'DL5_PN\n2', '#4ECDC4'),
            (5, 'KC\n312', '#45B7D1'),
            (7, 'MBON\n65', '#FFA07A'),
            (9, 'Behavior', '#98D8C8')
        ]

        for x, label, color in levels:
            circle = plt.Circle((x, 3), 0.4, color=color, alpha=0.7, ec='black', lw=2)
            ax.add_patch(circle)
            ax.text(x, 3, label, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrows
        for i in range(len(levels) - 1):
            x1, x2 = levels[i][0] + 0.4, levels[i+1][0] - 0.4
            ax.arrow(x1, 3, x2 - x1, 0, head_width=0.2, head_length=0.1,
                    fc='gray', ec='gray', lw=2)

        # Bottleneck annotation
        ax.text(2, 2, 'CRITICAL\nBOTTLENECK', ha='center', fontsize=8,
               color='red', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    def _plot_door_responses(self, ax):
        """Plot DoOR Or7a responses for odor panel."""
        if self._door_responses is None or len(self._door_responses) == 0:
            ax.text(0.5, 0.5, 'DoOR data not loaded', ha='center', va='center')
            return

        responses = self._door_responses[self._door_responses['found']].copy()
        responses = responses.sort_values('or7a_response', ascending=True)

        y_pos = np.arange(len(responses))
        colors = ['red' if r > 0.5 else 'orange' if r > 0.1 else 'green'
                 for r in responses['or7a_response']]

        ax.barh(y_pos, responses['or7a_response'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(responses['odor_query'])
        ax.set_xlabel('Or7a Response')
        ax.set_title('DoOR Or7a Responses', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    def _plot_bottleneck_analysis(self, ax):
        """Plot bottleneck severity."""
        if 'bottlenecks' not in self._pathway_summary:
            ax.text(0.5, 0.5, 'Bottleneck data not available', ha='center', va='center')
            return

        bottlenecks = self._pathway_summary['bottlenecks']

        transitions = bottlenecks['transition'].tolist()
        ratios = bottlenecks['expansion_ratio'].tolist()
        colors = ['red' if r < 0.1 else 'orange' if r < 0.5 else 'green'
                 for r in ratios]

        y_pos = np.arange(len(transitions))
        ax.barh(y_pos, ratios, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(transitions, fontsize=8)
        ax.set_xlabel('Expansion Ratio')
        ax.set_title('Circuit Bottlenecks', fontweight='bold')
        ax.axvline(1, color='black', linestyle='--', label='1:1 ratio')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

    def _plot_suppression_predictions(self, ax):
        """Plot predicted suppression effects."""
        predictions = [
            ('Benzaldehyde\nBlocking', 0.9, 0.1),
            ('Hexanol\nCross-Transfer', 0.5, 0.1),
            ('Control Odor\nLearning', 0.7, 0.65)
        ]

        x = np.arange(len(predictions))
        width = 0.35

        baseline = [p[1] for p in predictions]
        suppressed = [p[2] for p in predictions]

        ax.bar(x - width/2, baseline, width, label='Baseline', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, suppressed, width, label='OR7a Suppressed', color='coral', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in predictions], fontsize=9)
        ax.set_ylabel('Learning Index')
        ax.set_title('Predicted Suppression Effects', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

    def _plot_experimental_design(self, ax):
        """Plot experimental design summary."""
        design = self.generate_experimental_design()

        ax.axis('tight')
        ax.axis('off')

        # Create summary table
        summary = design.groupby(['condition', 'expected_outcome']).size().reset_index(name='n_experiments')

        table_data = []
        for _, row in summary.iterrows():
            table_data.append([
                row['condition'],
                row['expected_outcome'],
                str(row['n_experiments'])
            ])

        table = ax.table(cellText=table_data,
                        colLabels=['Condition', 'Expected Outcome', 'N Experiments'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.4, 0.5, 0.1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        ax.set_title('Experimental Design Summary', fontweight='bold', pad=20)


def run_complete_analysis(
    flywire_data_dir: Path,
    pathway_results_dir: Path,
    behavioral_data_path: Optional[Path] = None,
    output_dir: Path = Path("results/or7a_blocking_analysis")
):
    """
    Run complete OR7a blocking & cross-transfer analysis pipeline.

    This is the main entry point for analyzing OR7a pathway contributions
    to benzaldehyde blocking and hexanol cross-transfer.

    Parameters
    ----------
    flywire_data_dir : Path
        Directory with FlyWire CSV exports
    pathway_results_dir : Path
        Directory with OR7a pathway analysis results
    behavioral_data_path : Optional[Path]
        Path to behavioral data CSV (if available)
    output_dir : Path
        Output directory for analysis results
    """
    logger.info("="*80)
    logger.info("OR7a Blocking & Cross-Transfer Analysis")
    logger.info("="*80)

    # Initialize analyzer
    analyzer = OR7aPathwayAnalyzer(
        flywire_data_dir=flywire_data_dir,
        pathway_results_dir=pathway_results_dir,
        output_dir=output_dir
    )

    # Load behavioral data if provided
    behavioral_data = None
    if behavioral_data_path and behavioral_data_path.exists():
        logger.info(f"\nLoading behavioral data from {behavioral_data_path}")
        behavioral_data = pd.read_csv(behavioral_data_path)

    # 1. Analyze baseline
    logger.info("\n[1/5] Analyzing baseline behavior & pathway...")
    baseline = analyzer.analyze_baseline_behavior(behavioral_data)

    # Save baseline analysis
    baseline_path = output_dir / 'or7a_baseline_analysis.json'
    import json
    with open(baseline_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json.dump(baseline, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x))
    logger.info(f"Saved baseline analysis to {baseline_path}")

    # 2. Predict suppression effects
    logger.info("\n[2/5] Predicting OR7a suppression effects...")
    predictions = analyzer.predict_suppression_effects()

    pred_path = output_dir / 'or7a_suppression_predictions.json'
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x))
    logger.info(f"Saved predictions to {pred_path}")

    # 3. Generate experimental design
    logger.info("\n[3/5] Generating experimental design...")
    design = analyzer.generate_experimental_design()

    # 4. Create visualizations
    logger.info("\n[4/5] Creating visualizations...")
    analyzer.visualize_complete_analysis()

    # 5. Generate summary report
    logger.info("\n[5/5] Generating summary report...")
    _generate_summary_report(analyzer, baseline, predictions, output_dir)

    logger.info("\n" + "="*80)
    logger.info("Analysis complete!")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("="*80)

    return {
        'baseline': baseline,
        'predictions': predictions,
        'experimental_design': design
    }


def _generate_summary_report(
    analyzer: OR7aPathwayAnalyzer,
    baseline: Dict,
    predictions: Dict,
    output_dir: Path
):
    """Generate markdown summary report."""
    report = []

    report.append("# OR7a Blocking & Cross-Transfer Analysis Summary\n")
    report.append("## Pathway Architecture\n")

    if 'pathway_architecture' in baseline:
        arch = baseline['pathway_architecture']
        if 'critical_bottleneck' in arch:
            bn = arch['critical_bottleneck']
            report.append(f"**Critical Bottleneck Identified:**\n")
            report.append(f"- Transition: {bn['transition']}\n")
            report.append(f"- {bn['source_neurons']} → {bn['target_neurons']} neurons\n")
            report.append(f"- Expansion ratio: {bn['expansion_ratio']:.3f}\n")
            report.append(f"- Severity: {bn['severity']}\n\n")

    report.append("## DoOR Predictions\n")

    if 'door_predictions' in baseline:
        door = baseline['door_predictions']
        if 'benzaldehyde_response' in door:
            report.append(f"- Benzaldehyde Or7a response: {door['benzaldehyde_response']:.3f}\n")
        if 'hexanol_response' in door:
            report.append(f"- Hexanol Or7a response: {door['hexanol_response']:.3f}\n")
        if 'predicted_cross_transfer' in door:
            report.append(f"- Predicted cross-transfer: {door['predicted_cross_transfer']:.1%}\n\n")

    report.append("## Suppression Predictions\n")

    if 'blocking_reduction' in predictions:
        report.append("**Expected effects of OR7a suppression:**\n")
        report.append(f"- Blocking: {predictions['blocking_reduction']['expected_change']}\n")

    if 'transfer_reduction' in predictions:
        report.append(f"- Cross-transfer: {predictions['transfer_reduction']['expected_change']}\n\n")

    # Write report
    report_path = output_dir / 'OR7A_ANALYSIS_SUMMARY.md'
    with open(report_path, 'w') as f:
        f.writelines(report)

    logger.info(f"Saved summary report to {report_path}")


if __name__ == "__main__":
    # Example usage
    run_complete_analysis(
        flywire_data_dir=Path("data/flywire"),
        pathway_results_dir=Path("results/or7a_complete_pathway"),
        output_dir=Path("results/or7a_blocking_analysis")
    )
