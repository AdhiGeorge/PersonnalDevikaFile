import tiktoken
from typing import Dict, Optional, Any
import json
import logging
from datetime import datetime
from Agentres.config import Config
from pathlib import Path

logger = logging.getLogger(__name__)

class TokenTracker:
    def __init__(self, config: Config):
        """Initialize the token tracker with configuration."""
        if not isinstance(config, Config):
            raise ValueError("config must be an instance of Config")
            
        self.config = config
        self.log_file = "token_usage.json"
        self._ensure_log_file()
        self._load_usage_data()
        self.encoders = {}
        
        logger.info("Token tracker initialized")

    def _ensure_log_file(self):
        """Ensure the log file exists."""
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        if not Path(self.log_file).exists():
            with open(self.log_file, 'w') as f:
                json.dump([], f)

    def _load_usage_data(self):
        """Load existing usage data."""
        try:
            with open(self.log_file, 'r') as f:
                self.usage_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Token usage file corrupted, starting fresh")
            self.usage_data = []

    def _save_usage_data(self):
        """Save usage data to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save token usage data: {str(e)}")

    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get or create encoder for a model."""
        if model not in self.encoders:
            try:
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self.encoders[model]

    def _get_pricing(self, model: str, token_type: str) -> float:
        """Get pricing for a model and token type."""
        try:
            return self.config.get(f"llm_providers.{model}.pricing.{token_type}", 0.0)
        except Exception:
            return 0.0

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for a specific model."""
        try:
            encoder = self._get_encoder(model)
            return len(encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return 0

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for input and output tokens."""
        try:
            input_price = self._get_pricing(model, "input")
            output_price = self._get_pricing(model, "output")
            
            input_cost = (input_tokens / 1000) * input_price
            output_cost = (output_tokens / 1000) * output_price
            
            return input_cost + output_cost
        except Exception as e:
            logger.error(f"Error calculating cost: {str(e)}")
            return 0.0

    def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        project_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track token usage with enhanced metadata."""
        timestamp = datetime.now().isoformat()
        usage_entry = {
            "timestamp": timestamp,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": self._calculate_cost(model, input_tokens, output_tokens),
            "metadata": {
                "project_name": project_name,
                **(metadata or {})
            }
        }
        
        self.usage_data.append(usage_entry)
        self._save_usage_data()
        
        # Log usage for monitoring
        logger.info(f"Token usage tracked: {json.dumps(usage_entry)}")

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model and token usage."""
        # Default costs per 1K tokens (can be moved to config)
        costs = {
            "gpt-4o": {"input": 0.03, "output": 0.06},
            "gpt-35-turbo": {"input": 0.0015, "output": 0.002},
            "text-embedding-ada-002": {"input": 0.0001, "output": 0.0}
        }
        
        model_costs = costs.get(model, {"input": 0.0, "output": 0.0})
        return (input_tokens * model_costs["input"] + output_tokens * model_costs["output"]) / 1000

    def get_usage_summary(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage summary with analytics."""
        relevant_entries = [
            entry for entry in self.usage_data
            if not project_name or entry["metadata"].get("project_name") == project_name
        ]
        
        if not relevant_entries:
            return {
                "total_tokens": 0,
                "total_cost": 0.0,
                "model_breakdown": {},
                "time_series": []
            }
        
        # Calculate totals
        total_tokens = sum(entry["total_tokens"] for entry in relevant_entries)
        total_cost = sum(entry["cost"] for entry in relevant_entries)
        
        # Model breakdown
        model_breakdown = {}
        for entry in relevant_entries:
            model = entry["model"]
            if model not in model_breakdown:
                model_breakdown[model] = {
                    "tokens": 0,
                    "cost": 0.0,
                    "count": 0
                }
            model_breakdown[model]["tokens"] += entry["total_tokens"]
            model_breakdown[model]["cost"] += entry["cost"]
            model_breakdown[model]["count"] += 1
        
        # Time series data
        time_series = [
            {
                "timestamp": entry["timestamp"],
                "tokens": entry["total_tokens"],
                "cost": entry["cost"],
                "model": entry["model"]
            }
            for entry in relevant_entries
        ]
        
        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "model_breakdown": model_breakdown,
            "time_series": time_series
        }

    def get_model_usage(self, model: str) -> Dict[str, Any]:
        """Get usage statistics for a specific model."""
        model_entries = [entry for entry in self.usage_data if entry["model"] == model]
        
        if not model_entries:
            return {
                "total_tokens": 0,
                "total_cost": 0.0,
                "average_tokens_per_request": 0,
                "request_count": 0
            }
        
        total_tokens = sum(entry["total_tokens"] for entry in model_entries)
        total_cost = sum(entry["cost"] for entry in model_entries)
        
        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "average_tokens_per_request": total_tokens / len(model_entries),
            "request_count": len(model_entries)
        }

    def get_usage_summary(self) -> Dict:
        """Get summary of token usage and costs."""
        return {
            "total_tokens": self.usage_data[-1]["total_tokens"],
            "total_cost": self.usage_data[-1]["cost"],
            "model_usage": self.usage_data[-1]["metadata"],
            "last_updated": datetime.utcnow().isoformat()
        }

    def save_usage_report(self, filepath: str):
        """Save usage report to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_usage_summary(), f, indent=2)
            logger.info(f"Usage report saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving usage report: {str(e)}")

    def reset_usage(self):
        """Reset usage statistics."""
        self.usage_data = []
        self._save_usage_data()
        logger.info("Usage statistics reset") 