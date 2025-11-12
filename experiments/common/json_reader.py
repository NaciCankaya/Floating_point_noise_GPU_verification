"""JSON reader for experiment data."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentReader:
    """Read experiment JSON files."""
    
    def __init__(self, json_path: str):
        """
        Initialize reader with experiment JSON.
        
        Args:
            json_path: Path to experiment JSON file
        """
        self.json_path = Path(json_path)
        
        if not self.json_path.exists():
            raise FileNotFoundError(f"Experiment file not found: {json_path}")
        
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded experiment: {self.data['experiment_metadata']['experiment_type']}")
    
    def get_config(self, config_id: str) -> Optional[Dict]:
        """Get configuration by ID."""
        for config in self.data["configurations"]:
            if config["config_id"] == config_id:
                return config
        return None
    
    def get_configs_by_hardware(self, hardware: str) -> List[Dict]:
        """Get all configurations for specific hardware."""
        return [c for c in self.data["configurations"] 
                if c["hardware"] == hardware]
    
    def get_runs(
        self,
        config_id: Optional[str] = None,
        rep_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Get runs matching criteria.
        
        Args:
            config_id: Filter by configuration
            rep_id: Filter by repetition
            
        Returns:
            List of matching runs
        """
        runs = self.data["runs"]
        
        if config_id is not None:
            runs = [r for r in runs if r["config_id"] == config_id]
        
        if rep_id is not None:
            runs = [r for r in runs if r["rep_id"] == rep_id]
        
        return runs
    
    def get_baseline_run(self, hardware: str, rep_id: int = 0) -> Optional[Dict]:
        """
        Get baseline run for specified hardware.
        
        Baseline is typically the first configuration for that hardware.
        
        Args:
            hardware: Hardware type (e.g., "A100-80GB")
            rep_id: Which repetition to fetch
            
        Returns:
            Run data or None if not found
        """
        # Find baseline config (typically has smallest variable_value or first listed)
        hw_configs = self.get_configs_by_hardware(hardware)
        
        if not hw_configs:
            logger.warning(f"No configurations found for {hardware}")
            return None
        
        # Sort by variable_value to get baseline
        baseline_config = min(hw_configs, key=lambda c: c.get("variable_value", 0))
        
        # Get run for this config and rep
        runs = self.get_runs(
            config_id=baseline_config["config_id"],
            rep_id=rep_id
        )
        
        if not runs:
            logger.warning(
                f"No run found for {baseline_config['config_id']} rep {rep_id}"
            )
            return None
        
        return runs[0]
    
    def extract_all_data(self) -> Dict:
        """Return complete experiment data."""
        return self.data
