import tiktoken
from typing import Dict, Optional
import json
import logging
from datetime import datetime
from src.config import Config

logger = logging.getLogger(__name__)

class TokenTracker:
    def __init__(self):
        self.config = Config()
        self.usage = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "calls": [],
            "model_usage": {}
        }
        self.encoders = {}

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

    def track_usage(self, 
                   model: str, 
                   input_text: str, 
                   output_text: str, 
                   metadata: Optional[Dict] = None) -> Dict:
        """Track token usage and cost for an API call."""
        try:
            input_tokens = self.count_tokens(input_text, model)
            output_tokens = self.count_tokens(output_text, model)
            total_tokens = input_tokens + output_tokens
            
            cost = self.calculate_cost(input_tokens, output_tokens, model)
            
            call_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "metadata": metadata or {}
            }
            
            # Update usage statistics
            self.usage["total_tokens"] += total_tokens
            self.usage["total_cost"] += cost
            self.usage["calls"].append(call_data)
            
            # Update model-specific usage
            if model not in self.usage["model_usage"]:
                self.usage["model_usage"][model] = {
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "calls": 0
                }
            
            self.usage["model_usage"][model]["total_tokens"] += total_tokens
            self.usage["model_usage"][model]["total_cost"] += cost
            self.usage["model_usage"][model]["calls"] += 1
            
            logger.info(f"Token usage tracked: {json.dumps(call_data)}")
            return call_data
            
        except Exception as e:
            logger.error(f"Error tracking usage: {str(e)}")
            return {}

    def get_usage_summary(self) -> Dict:
        """Get summary of token usage and costs."""
        return {
            "total_tokens": self.usage["total_tokens"],
            "total_cost": self.usage["total_cost"],
            "model_usage": self.usage["model_usage"],
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
        self.usage = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "calls": [],
            "model_usage": {}
        }
        logger.info("Usage statistics reset") 