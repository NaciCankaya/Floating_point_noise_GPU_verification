"""JSON writer for experiment results."""

import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentWriter:
    """Write experiment results in standardized JSON format."""
    
    def __init__(
        self,
        experiment_type: str,
        variable_tested: str,
        model_name: str = "Qwen3-30B-A3B-AWQ-Int4",
        sequence_length: int = 6000,
        decode_steps: int = 30
    ):
        """
        Initialize experiment writer.
        
        Args:
            experiment_type: e.g., "batch_size", "compile"
            variable_tested: Variable name being tested
            model_name: Model identifier
            sequence_length: Target sequence length
            decode_steps: Number of decode steps
        """
        self.experiment_type = experiment_type
        self.variable_tested = variable_tested
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.decode_steps = decode_steps
        
        # Initialize data structure
        self.data = {
            "experiment_metadata": {
                "experiment_type": experiment_type,
                "variable_tested": variable_tested,
                "model": model_name,
                "model_size": "30B",
                "architecture": "MoE",
                "sequence_length": sequence_length,
                "decode_steps": decode_steps,
                "extraction_config": {
                    "layers": [1, 2, 4, 12, 39],
                    "positions": [-3, -2, -1],
                    "hidden_dim": 3584,
                    "key_dim": 512,
                    "top_k_logprobs": 10
                },
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "configurations": [],
            "runs": []
        }
    
    def add_configuration(
        self,
        config_id: str,
        hardware: str,
        variable_value,
        provider: str = "RunPod",
        cuda_version: Optional[str] = None,
        fixed_params: Optional[Dict] = None
    ):
        """
        Add a configuration to the experiment.
        
        Args:
            config_id: Unique identifier (e.g., "A100_bs1")
            hardware: GPU type (e.g., "A100-80GB")
            variable_value: Value of the tested variable
            provider: Cloud provider
            cuda_version: CUDA version
            fixed_params: Other parameters held constant
        """
        # Get version info
        torch_version = torch.__version__
        
        try:
            import transformers
            transformers_version = transformers.__version__
        except:
            transformers_version = "unknown"
        
        try:
            import flash_attn
            flash_attn_version = flash_attn.__version__
        except:
            flash_attn_version = "unknown"
        
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Detect CUDA version if not provided
        if cuda_version is None:
            cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
        
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
            "fixed_params": fixed_params or {}
        }
        
        self.data["configurations"].append(config)
        logger.info(f"Added configuration: {config_id}")
    
    def add_run(
        self,
        config_id: str,
        rep_id: int,
        extraction_result: Dict,
        runtime_seconds: float,
        prompt_text: str
    ):
        """
        Add a run result to the experiment.
        
        Args:
            config_id: Configuration identifier
            rep_id: Repetition number (0, 1, 2)
            extraction_result: Output from InferenceRunner
            runtime_seconds: Runtime in seconds
            prompt_text: Input prompt
        """
        run = {
            "config_id": config_id,
            "rep_id": rep_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": runtime_seconds,
            "prompt_text": prompt_text,
            "decode_steps": extraction_result["decode_steps"]
        }
        
        self.data["runs"].append(run)
        logger.info(f"Added run: {config_id} rep {rep_id}")
    
    def save(self, filepath: str):
        """
        Save experiment data to JSON file.
        
        Args:
            filepath: Output path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Experiment saved to {filepath} ({file_size_mb:.1f} MB)")
    
    def merge_with_reference(self, reference_json_path: str):
        """
        Merge this experiment with reference baseline data.
        
        This is used when an experiment reuses baseline measurements
        from experiment 0.
        
        Args:
            reference_json_path: Path to reference_baseline.json
        """
        logger.info(f"Merging with reference baseline: {reference_json_path}")
        
        with open(reference_json_path, 'r') as f:
            ref_data = json.load(f)
        
        # Find baseline configurations to import
        for ref_config in ref_data["configurations"]:
            # Check if we already have this hardware
            if not any(c["hardware"] == ref_config["hardware"] 
                      for c in self.data["configurations"]):
                # Add baseline config
                self.data["configurations"].append(ref_config)
                logger.info(f"  Imported config: {ref_config['config_id']}")
        
        # Import baseline runs
        for ref_run in ref_data["runs"]:
            # Check if we already have this run
            if not any(
                r["config_id"] == ref_run["config_id"] and 
                r["rep_id"] == ref_run["rep_id"]
                for r in self.data["runs"]
            ):
                self.data["runs"].append(ref_run)
                logger.info(f"  Imported run: {ref_run['config_id']} rep {ref_run['rep_id']}")
        
        logger.info("Merge complete")
