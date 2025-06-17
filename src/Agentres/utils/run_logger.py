import json
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

class RunLogger:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "logs")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a new run log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.base_dir / f"run_{timestamp}.json"
        
        # Initialize the log structure
        self.current_run = {
            "timestamp": timestamp,
            "query": None,
            "planner": {
                "output": None,
                "steps": []
            },
            "researcher": {
                "output": None,
                "scraped_data": [],
                "synthesized_data": None
            },
            "coder": {
                "output": None,
                "generated_code": None,
                "execution_results": None
            },
            "errors": [],
            "warnings": [],
            "execution_time": None
        }
        
        # Save initial structure
        self._save_log()
    
    def _save_log(self):
        """Save the current run log to file"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_run, f, indent=2, ensure_ascii=False)
    
    def set_query(self, query: str):
        """Set the main query for this run"""
        self.current_run["query"] = query
        self._save_log()
    
    def add_planner_step(self, step: Dict[str, Any]):
        """Add a planner step"""
        self.current_run["planner"]["steps"].append(step)
        self._save_log()
    
    def set_planner_output(self, output: str):
        """Set the final planner output"""
        self.current_run["planner"]["output"] = output
        self._save_log()
    
    def add_scraped_data(self, url: str, content: str, metadata: Dict[str, Any] = None):
        """Add scraped data with its URL and metadata"""
        self.current_run["researcher"]["scraped_data"].append({
            "url": url,
            "content": content,
            "metadata": metadata or {}
        })
        self._save_log()
    
    def set_researcher_output(self, output: str):
        """Set the researcher's final output"""
        self.current_run["researcher"]["output"] = output
        self._save_log()
    
    def set_synthesized_data(self, data: Dict[str, Any]):
        """Set the synthesized data from research"""
        self.current_run["researcher"]["synthesized_data"] = data
        self._save_log()
    
    def set_coder_output(self, output: str):
        """Set the coder's output"""
        self.current_run["coder"]["output"] = output
        self._save_log()
    
    def set_generated_code(self, code: str, language: str, file_path: str):
        """Set the generated code with its metadata"""
        self.current_run["coder"]["generated_code"] = {
            "code": code,
            "language": language,
            "file_path": file_path
        }
        self._save_log()
    
    def set_execution_results(self, results: Dict[str, Any]):
        """Set the execution results"""
        self.current_run["coder"]["execution_results"] = results
        self._save_log()
    
    def add_error(self, error: str, context: Dict[str, Any] = None):
        """Add an error with optional context"""
        self.current_run["errors"].append({
            "message": error,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        })
        self._save_log()
    
    def add_warning(self, warning: str, context: Dict[str, Any] = None):
        """Add a warning with optional context"""
        self.current_run["warnings"].append({
            "message": warning,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        })
        self._save_log()
    
    def set_execution_time(self, start_time: datetime, end_time: datetime):
        """Set the total execution time"""
        duration = (end_time - start_time).total_seconds()
        self.current_run["execution_time"] = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_seconds": duration
        }
        self._save_log()
    
    def get_log_path(self) -> str:
        """Get the path to the current log file"""
        return str(self.log_file) 