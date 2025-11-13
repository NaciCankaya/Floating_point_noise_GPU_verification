#!/usr/bin/env python3
"""
JSON reader utilities for ablation experiments

Reads and queries experiment results from JSON files.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class ExperimentReader:
    """
    Reads and queries experiment data from JSON files.
    """

    def __init__(self, filepath: str):
        """
        Initialize reader with a JSON file.

        Args:
            filepath: Path to experiment JSON file
        """
        self.filepath = Path(filepath)
        with open(self.filepath, 'r') as f:
            self.data = json.load(f)

    def get_metadata(self) -> Dict:
        """Get experiment metadata."""
        return self.data["experiment_metadata"]

    def get_configurations(self) -> List[Dict]:
        """Get all configurations."""
        return self.data["configurations"]

    def get_config_by_id(self, config_id: str) -> Optional[Dict]:
        """Get a specific configuration by ID."""
        for config in self.data["configurations"]:
            if config["config_id"] == config_id:
                return config
        return None

    def get_configs_by_hardware(self, hardware: str) -> List[Dict]:
        """Get all configurations for a specific hardware type."""
        return [
            config for config in self.data["configurations"]
            if config["hardware"] == hardware
        ]

    def get_configs_by_value(self, variable_value: Any) -> List[Dict]:
        """Get all configurations with a specific variable value."""
        return [
            config for config in self.data["configurations"]
            if config["variable_value"] == variable_value
        ]

    def get_runs(
        self,
        config_id: Optional[str] = None,
        rep_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get runs matching the specified filters.

        Args:
            config_id: Filter by configuration ID
            rep_id: Filter by repetition ID

        Returns:
            List of matching runs
        """
        runs = self.data["runs"]

        if config_id is not None:
            runs = [run for run in runs if run["config_id"] == config_id]

        if rep_id is not None:
            runs = [run for run in runs if run["rep_id"] == rep_id]

        return runs

    def get_run(self, config_id: str, rep_id: int) -> Optional[Dict]:
        """Get a specific run."""
        for run in self.data["runs"]:
            if run["config_id"] == config_id and run["rep_id"] == rep_id:
                return run
        return None

    def extract_signal(
        self,
        config_id: str,
        rep_id: int,
        signal_type: str = "hidden_states",
        layer: str = "layer_39",
        position: str = "pos_-1",
        decode_step: int = -1,
    ) -> Optional[List[float]]:
        """
        Extract a specific signal from a run.

        Args:
            config_id: Configuration ID
            rep_id: Repetition ID
            signal_type: "hidden_states", "key_vectors", or "logprobs"
            layer: Layer identifier (e.g., "layer_39")
            position: Position identifier (e.g., "pos_-1")
            decode_step: Decode step index (-1 for last)

        Returns:
            Signal data as list of floats, or None if not found
        """
        run = self.get_run(config_id, rep_id)
        if run is None:
            return None

        step_data = run["decode_steps"][decode_step]

        if signal_type == "logprobs":
            return step_data["logprobs"][position]["log_probs"]
        else:
            return step_data[signal_type][layer][position]

    def compare_reproducibility(self, config_id: str) -> Dict:
        """
        Check reproducibility for all runs of a configuration.

        Returns:
            dict: {
                "is_reproducible": bool,
                "num_reps": int,
                "token_match": bool,
                "signal_match": bool,
                "max_l2_distance": float,
            }
        """
        import numpy as np

        runs = self.get_runs(config_id=config_id)
        if len(runs) < 2:
            return {
                "is_reproducible": True,
                "num_reps": len(runs),
                "token_match": True,
                "signal_match": True,
                "max_l2_distance": 0.0,
            }

        # Check token sequence reproducibility
        first_run = runs[0]
        first_step = first_run["decode_steps"][0]

        token_match = True
        for run in runs[1:]:
            if run["decode_steps"][0]["token_id"] != first_step["token_id"]:
                token_match = False
                break

        # Check signal reproducibility (sample last layer, last position, last step)
        signal_match = True
        max_l2 = 0.0

        # Get a signal to compare
        layer_keys = list(first_step["hidden_states"].keys())
        if layer_keys:
            layer = layer_keys[-1]  # Last layer
            pos_keys = list(first_step["hidden_states"][layer].keys())
            if pos_keys:
                pos = pos_keys[-1]  # Last position

                first_signal = np.array(first_run["decode_steps"][-1]["hidden_states"][layer][pos])

                for run in runs[1:]:
                    run_signal = np.array(run["decode_steps"][-1]["hidden_states"][layer][pos])
                    l2_dist = np.linalg.norm(run_signal - first_signal)
                    max_l2 = max(max_l2, l2_dist)

                    if l2_dist > 0:
                        signal_match = False

        is_reproducible = token_match and signal_match

        return {
            "is_reproducible": is_reproducible,
            "num_reps": len(runs),
            "token_match": token_match,
            "signal_match": signal_match,
            "max_l2_distance": float(max_l2),
        }

    def extract_all_data(self) -> Dict:
        """Return the complete experiment data."""
        return self.data

    def summary(self) -> str:
        """Generate a summary string of the experiment."""
        metadata = self.get_metadata()
        configs = self.get_configurations()
        runs = self.data["runs"]

        summary_lines = [
            "="*80,
            f"Experiment: {metadata['experiment_type']}",
            f"Variable Tested: {metadata['variable_tested']}",
            f"Model: {metadata['model']}",
            f"Date Created: {metadata['date_created']}",
            "="*80,
            f"Configurations: {len(configs)}",
            f"Total Runs: {len(runs)}",
            "",
        ]

        # List configurations
        summary_lines.append("Configurations:")
        for config in configs:
            summary_lines.append(
                f"  {config['config_id']}: {config['hardware']} "
                f"(value={config['variable_value']})"
            )

        summary_lines.append("")

        # Count runs per configuration
        summary_lines.append("Runs per configuration:")
        for config in configs:
            config_runs = self.get_runs(config_id=config["config_id"])
            summary_lines.append(f"  {config['config_id']}: {len(config_runs)} runs")

        summary_lines.append("="*80)

        return "\n".join(summary_lines)
