"""Token tracker for monitoring token usage."""

import tiktoken
from typing import Dict, Optional, Any
import json
import logging
from datetime import datetime, timezone
from config.config import Config
from pathlib import Path
import os
import asyncio
import re

logger = logging.getLogger(__name__)

class TokenTracker:
    """Token tracker for monitoring token usage."""
    
    def __init__(self, config: Config):
        """Initialize token tracker with configuration."""
        try:
            if not isinstance(config, Config):
                raise ValueError("config must be an instance of Config")
                
            self.config = config
            self._usage_data = None
            self._data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            self.usage_file = os.path.join(self._data_dir, "token_usage.json")
            self.log_file = os.path.join(self._data_dir, "token_usage.log")
            self.encoders = {}
            self._initialized = False
            self._init_lock = asyncio.Lock()
            
            # Validate file paths
            self._validate_paths()
            
            logger.info("Token tracker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize token tracker: {str(e)}")
            raise ValueError(f"Token tracker initialization failed: {str(e)}")

    def _validate_paths(self) -> None:
        """Validate file paths."""
        try:
            # Validate data directory
            if not os.path.exists(self._data_dir):
                os.makedirs(self._data_dir, exist_ok=True)
                
            # Validate usage file path
            usage_dir = os.path.dirname(self.usage_file)
            if not os.path.exists(usage_dir):
                os.makedirs(usage_dir, exist_ok=True)
                
            # Validate log file path
            log_dir = os.path.dirname(self.log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
        except Exception as e:
            logger.error(f"Error validating paths: {str(e)}")
            raise ValueError(f"Failed to validate paths: {str(e)}")

    @property
    def usage_data(self) -> Dict[str, Any]:
        """Lazy initialization of usage data."""
        if self._usage_data is None:
            self._usage_data = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "model_usage": {},
                "project_usage": {}
            }
            self._load_usage_data()
        return self._usage_data

    async def initialize(self):
        """Initialize async components."""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                # Ensure data directory exists
                os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                
                # Initialize usage data
                self._usage_data = {
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "model_usage": {},
                    "project_usage": {}
                }
                self._load_usage_data()
                
                # Test token counting
                test_text = "test token counting"
                test_model = "gpt-4"
                try:
                    token_count = self.count_tokens(test_text, test_model)
                    if token_count <= 0:
                        raise ValueError("Invalid token count")
                    logger.info("Token counting test successful")
                except Exception as e:
                    logger.error(f"Token counting test failed: {str(e)}")
                    raise ValueError(f"Failed to count tokens: {str(e)}")
                
                # Test cost calculation
                try:
                    cost = self.calculate_cost(100, 100, test_model)
                    if cost < 0:
                        raise ValueError("Invalid cost calculation")
                    logger.info("Cost calculation test successful")
                except Exception as e:
                    logger.error(f"Cost calculation test failed: {str(e)}")
                    raise ValueError(f"Failed to calculate costs: {str(e)}")
                
                # Test file operations
                try:
                    self._ensure_log_file()
                    self._save_usage_data()
                    logger.info("File operations test successful")
                except Exception as e:
                    logger.error(f"File operations test failed: {str(e)}")
                    raise ValueError(f"Failed to test file operations: {str(e)}")
                
                self._initialized = True
                logger.info("Token tracker async components initialized")
            except Exception as e:
                logger.error(f"Failed to initialize async components: {str(e)}")
                raise ValueError(f"Async initialization failed: {str(e)}")

    def _ensure_log_file(self):
        """Ensure the log file exists."""
        try:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            if not Path(self.log_file).exists():
                with open(self.log_file, 'w') as f:
                    json.dump([], f)
        except Exception as e:
            logger.error(f"Error ensuring log file exists: {str(e)}")
            raise ValueError(f"Failed to create log file: {str(e)}")

    def _load_usage_data(self) -> None:
        """Load usage data from file."""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    self._validate_usage_data(data)
                    self._usage_data = data
            else:
                self._usage_data = {
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "model_usage": {},
                    "project_usage": {}
                }
                self._save_usage_data()
        except Exception as e:
            logger.error(f"Error loading usage data: {str(e)}")
            self._usage_data = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "model_usage": {},
                "project_usage": {}
            }

    def _validate_usage_data(self, data: Dict[str, Any]) -> None:
        """Validate usage data structure."""
        required_keys = ["total_tokens", "total_cost", "model_usage", "project_usage"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key in usage data: {key}")
                
        if not isinstance(data["total_tokens"], int):
            raise ValueError("total_tokens must be an integer")
        if not isinstance(data["total_cost"], (int, float)):
            raise ValueError("total_cost must be a number")
        if not isinstance(data["model_usage"], dict):
            raise ValueError("model_usage must be a dictionary")
        if not isinstance(data["project_usage"], dict):
            raise ValueError("project_usage must be a dictionary")

    def _save_usage_data(self) -> None:
        """Save usage data to file."""
        try:
            self._validate_usage_data(self._usage_data)
            with open(self.usage_file, 'w') as f:
                json.dump(self._usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving usage data: {str(e)}")
            raise ValueError(f"Failed to save usage data: {str(e)}")

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
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-4o": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            "text-embedding-3-small": {"input": 0.00002, "output": 0},
            "text-embedding-ada-002": {"input": 0.0001, "output": 0}
        }
        
        # Get model pricing or use default
        model_pricing = pricing.get(model, {"input": 0.001, "output": 0.002})
        return model_pricing.get(token_type, 0.001)

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for a model."""
        try:
            encoder = self._get_encoder(model)
            return len(encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return 0

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for token usage."""
        try:
            input_cost = input_tokens * self._get_pricing(model, "input")
            output_cost = output_tokens * self._get_pricing(model, "output")
            return input_cost + output_cost
        except Exception as e:
            logger.error(f"Error calculating cost: {str(e)}")
            return 0.0

    def _validate_project_name(self, project_name: str) -> None:
        """Validate project name."""
        if project_name and not re.match(r'^[a-zA-Z0-9_-]+$', project_name):
            raise ValueError("Project name must contain only letters, numbers, underscores, and hyphens")

    async def track_usage(self, model: str, input_tokens: int, output_tokens: int, project_name: Optional[str] = None) -> None:
        """Track token usage.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            project_name: Optional project name
        """
        if not self._initialized:
            raise RuntimeError("Token tracker not initialized. Call initialize() first.")
            
        try:
            # Validate inputs
            if not model or not isinstance(model, str):
                raise ValueError("Invalid model name")
            if not isinstance(input_tokens, int) or input_tokens < 0:
                raise ValueError("Invalid input token count")
            if not isinstance(output_tokens, int) or output_tokens < 0:
                raise ValueError("Invalid output token count")
            if project_name:
                self._validate_project_name(project_name)
                
            # Calculate cost
            cost = self.calculate_cost(input_tokens, output_tokens, model)
            
            # Update usage data
            self.usage_data["total_tokens"] += input_tokens + output_tokens
            self.usage_data["total_cost"] += cost
            
            # Update model usage
            if model not in self.usage_data["model_usage"]:
                self.usage_data["model_usage"][model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0
                }
            self.usage_data["model_usage"][model]["input_tokens"] += input_tokens
            self.usage_data["model_usage"][model]["output_tokens"] += output_tokens
            self.usage_data["model_usage"][model]["total_tokens"] += input_tokens + output_tokens
            self.usage_data["model_usage"][model]["cost"] += cost
            
            # Update project usage if project name provided
            if project_name:
                if project_name not in self.usage_data["project_usage"]:
                    self.usage_data["project_usage"][project_name] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost": 0.0,
                        "models": {}
                    }
                self.usage_data["project_usage"][project_name]["input_tokens"] += input_tokens
                self.usage_data["project_usage"][project_name]["output_tokens"] += output_tokens
                self.usage_data["project_usage"][project_name]["total_tokens"] += input_tokens + output_tokens
                self.usage_data["project_usage"][project_name]["cost"] += cost
                
                # Update model usage for project
                if model not in self.usage_data["project_usage"][project_name]["models"]:
                    self.usage_data["project_usage"][project_name]["models"][model] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost": 0.0
                    }
                self.usage_data["project_usage"][project_name]["models"][model]["input_tokens"] += input_tokens
                self.usage_data["project_usage"][project_name]["models"][model]["output_tokens"] += output_tokens
                self.usage_data["project_usage"][project_name]["models"][model]["total_tokens"] += input_tokens + output_tokens
                self.usage_data["project_usage"][project_name]["models"][model]["cost"] += cost
            
            # Save usage data
            self._save_usage_data()
            
            # Log usage
            self._log_usage(model, input_tokens, output_tokens, cost, project_name)
            
        except Exception as e:
            logger.error(f"Error tracking usage: {str(e)}")
            raise ValueError(f"Failed to track usage: {str(e)}")

    def _log_usage(self, model: str, input_tokens: int, output_tokens: int, cost: float, project_name: Optional[str] = None) -> None:
        """Log token usage.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost of token usage
            project_name: Optional project name
        """
        try:
            # Create log entry
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": cost,
                "project_name": project_name
            }
            
            # Load existing log
            log_data = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
                    
            # Add new entry
            log_data.append(log_entry)
            
            # Save log
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging usage: {str(e)}")
            raise ValueError(f"Failed to log usage: {str(e)}")

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        if not self._initialized:
            raise RuntimeError("Token tracker not initialized. Call initialize() first.")
        return self.usage_data.copy()

    def get_project_usage(self, project_name: str) -> Dict[str, Any]:
        """Get usage for a project."""
        if not self._initialized:
            raise RuntimeError("Token tracker not initialized. Call initialize() first.")
            
        try:
            self._validate_project_name(project_name)
            return self.usage_data["project_usage"].get(project_name, {})
        except Exception as e:
            logger.error(f"Error getting project usage: {str(e)}")
            raise ValueError(f"Failed to get project usage: {str(e)}")

    def get_model_usage(self, model: str) -> Dict[str, Any]:
        """Get usage for a model."""
        if not self._initialized:
            raise RuntimeError("Token tracker not initialized. Call initialize() first.")
        return self.usage_data["model_usage"].get(model, {})

    def save_usage_report(self, filepath: str) -> None:
        """Save usage report to file."""
        if not self._initialized:
            raise RuntimeError("Token tracker not initialized. Call initialize() first.")
            
        try:
            # Create report
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": self.get_usage_summary(),
                "models": {model: self.get_model_usage(model) for model in self.usage_data["model_usage"]},
                "projects": {project: self.get_project_usage(project) for project in self.usage_data["project_usage"]}
            }
            
            # Save report
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving usage report: {str(e)}")
            raise ValueError(f"Failed to save usage report: {str(e)}")

    def reset_usage(self) -> None:
        """Reset usage data."""
        if not self._initialized:
            raise RuntimeError("Token tracker not initialized. Call initialize() first.")
            
        try:
            # Reset usage data
            self._usage_data = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "model_usage": {},
                "project_usage": {}
            }
            
            # Save usage data
            self._save_usage_data()
            
            # Reset log file
            with open(self.log_file, 'w') as f:
                json.dump([], f)
                
        except Exception as e:
            logger.error(f"Error resetting usage: {str(e)}")
            raise ValueError(f"Failed to reset usage: {str(e)}")

    def get_usage(self, model: str) -> Dict[str, Any]:
        """Get usage for a model."""
        if not self._initialized:
            raise RuntimeError("Token tracker not initialized. Call initialize() first.")
        return self.get_model_usage(model)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if not self._initialized:
            return
            
        try:
            # Save usage data
            self._save_usage_data()
            
            # Reset state
            self._initialized = False
            
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")
            raise ValueError(f"Failed to cleanup: {str(e)}")

async def main():
    """Main function for testing."""
    try:
        # Create config
        config = Config()
        await config.initialize()
        
        # Create token tracker
        tracker = TokenTracker(config)
        await tracker.initialize()
        
        # Test token tracking
        model = "gpt-4"
        input_tokens = 100
        output_tokens = 50
        project = "test_project"
        
        # Track usage
        await tracker.track_usage(model, input_tokens, output_tokens, project)
        
        # Get usage summary
        summary = tracker.get_usage_summary()
        print(f"Usage summary: {summary}")
        
        # Get project usage
        project_usage = tracker.get_project_usage(project)
        print(f"Project usage: {project_usage}")
        
        # Get model usage
        model_usage = tracker.get_model_usage(model)
        print(f"Model usage: {model_usage}")
        
        # Cleanup
        await tracker.cleanup()
        await config.cleanup()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 