#!/usr/bin/env python3
"""
JSON writer utilities for ablation experiments

Writes experiment results in the standardized JSON schema.
"""

import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class ExperimentWriter:
    """
    Writes experiment data to JSON following the standardized schema.
    """

    def __init__(
        self,
        experiment_type: str,
        variable_tested: str,
        model: str = "Qwen3-30B-A3B-GPTQ-Int4",
        model_size: str = "30B",
        architecture: str = "MoE",
        sequence_length: int = 8192,
        decode_steps: int = 30,
        layer_indices: List[int] = [1, 2, 4, 12, 39],
        positions: List[int] = [-3, -2, -1],
        hidden_dim: int = 3584,
        key_dim: int = 512,
        top_k_logprobs: int = 10,
    ):
        """
        Initialize the experiment writer.

        Args:
            experiment_type: Type of experiment (e.g., "reference", "batch_size")
            variable_tested: Variable being tested (e.g., "batch_size", "compile")
            model: Model name
            model_size: Model size descriptor
            architecture: Model architecture type
            sequence_length: Maximum sequence length
            decode_steps: Number of decode steps
            layer_indices: Layers to extract from
            positions: Token positions to extract
            hidden_dim: Hidden state dimension
            key_dim: Key vector dimension
            top_k_logprobs: Number of top logprobs
        """
        self.data = {
            "experiment_metadata": {
                "experiment_type": experiment_type,
                "variable_tested": variable_tested,
                "model": model,
                "model_size": model_size,
                "architecture": architecture,
                "sequence_length": sequence_length,
                "decode_steps": decode_steps,
                "extraction_config": {
                    "layers": layer_indices,
                    "positions": positions,
                    "hidden_dim": hidden_dim,
                    "key_dim": key_dim,
                    "top_k_logprobs": top_k_logprobs,
                },
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "configurations": [],
            "runs": [],
        }

    def add_configuration(
        self,
        config_id: str,
        hardware: str,
        provider: str,
        variable_value: Any,
        cuda_version: str,
        torch_version: str,
        transformers_version: str,
        flash_attn_version: str,
        python_version: str,
        fixed_params: Dict[str, Any],
    ):
        """
        Add a configuration to the experiment.

        Args:
            config_id: Unique identifier for this configuration
            hardware: Hardware type (e.g., "A100-80GB", "H100")
            provider: Provider name (e.g., "RunPod", "vast.ai")
            variable_value: Value of the variable being tested
            cuda_version: CUDA version string
            torch_version: PyTorch version string
            transformers_version: Transformers version string
            flash_attn_version: Flash Attention version string
            python_version: Python version string
            fixed_params: Dictionary of fixed parameters
        """
        config = {
            "config_id": config_id,
            "hardware": hardware,
            "provider": provider,
            "variable_value": variable_value,
            "cuda_version": cuda_version,
            "torch_version": torch_version,
            "transformers_version": transformers_version,
            "flash_attn_version": flash_attn_version,
            "python_version": python_version,
            "fixed_params": fixed_params,
        }
        self.data["configurations"].append(config)

    def add_run(
        self,
        config_id: str,
        rep_id: int,
        run_data: Dict,
    ):
        """
        Add a run result to the experiment.

        Args:
            config_id: Configuration ID this run belongs to
            rep_id: Repetition ID (0, 1, 2, ...)
            run_data: Run result from runner.run_inference()
        """
        run = {
            "config_id": config_id,
            "rep_id": rep_id,
            "timestamp": run_data["timestamp"],
            "runtime_seconds": run_data["runtime_seconds"],
            "prompt_text": run_data["prompt_text"],
            "decode_steps": run_data["decode_steps"],
        }
        self.data["runs"].append(run)

    def save(self, filepath: str, indent: int = 2):
        """
        Save experiment data to JSON file.

        Args:
            filepath: Output file path
            indent: JSON indentation level
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=indent)

        print(f"\n✓ Experiment data saved to: {filepath}")
        print(f"  Configurations: {len(self.data['configurations'])}")
        print(f"  Runs: {len(self.data['runs'])}")

        # Calculate file size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")

    def get_data(self) -> Dict:
        """Return the experiment data dictionary."""
        return self.data


def merge_experiment_files(
    input_files: List[str],
    output_file: str,
    experiment_type: str,
    variable_tested: str,
):
    """
    Merge multiple experiment JSON files into a single file.

    Useful for combining baseline data with variant data.

    Args:
        input_files: List of JSON file paths to merge
        output_file: Output file path
        experiment_type: Experiment type for merged file
        variable_tested: Variable tested for merged file
    """
    print(f"\nMerging {len(input_files)} experiment files...")

    # Load all files
    all_data = []
    for filepath in input_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            all_data.append(data)
            print(f"  Loaded: {filepath}")

    # Use metadata from first file as base
    merged = all_data[0].copy()
    merged["experiment_metadata"]["experiment_type"] = experiment_type
    merged["experiment_metadata"]["variable_tested"] = variable_tested
    merged["experiment_metadata"]["date_created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Merge configurations and runs
    merged["configurations"] = []
    merged["runs"] = []

    for data in all_data:
        merged["configurations"].extend(data["configurations"])
        merged["runs"].extend(data["runs"])

    # Save merged file
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\n✓ Merged experiment saved to: {output_file}")
    print(f"  Total configurations: {len(merged['configurations'])}")
    print(f"  Total runs: {len(merged['runs'])}")
