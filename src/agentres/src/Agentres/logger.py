from functools import wraps
import json
from datetime import datetime
from pathlib import Path
from fastlogging import LogInit
import re
# from flask import request

from Agentres.config import Config


class Logger:
    def __init__(self, filename="agent.log"):
        config = Config()
        logs_dir = config.get_logs_dir()
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        self.logger = LogInit(pathName=logs_dir + "/" + filename, console=True, colors=True, encoding="utf-8")
        
        # Initialize JSON log structure
        self.json_log = {
            "timestamp": datetime.now().isoformat(),
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
        
        # Create JSON log file with timestamp as backup
        self.json_log_path = Path(logs_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._save_json_log()

    def _generate_filename_from_query(self, query: str) -> str:
        """Generate a filename from the query"""
        # Remove special characters and convert to lowercase
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
        # Take first 3-4 words and join with underscores
        words = clean_query.split()[:4]
        filename = '_'.join(words)
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{filename}_{timestamp}.json"

    def _save_json_log(self):
        """Save the current state of the JSON log to file"""
        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_log, f, indent=2, ensure_ascii=False)

    def read_log_file(self) -> str:
        with open(self.logger.pathName, "r") as file:
            return file.read()

    def info(self, message: str):
        """Log an info message"""
        self.logger.info(message)
        self.logger.flush()

    def error(self, message: str):
        """Log an error message"""
        self.logger.error(message)
        self.add_error(message)
        self.logger.flush()

    def warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)
        self.add_warning(message)
        self.logger.flush()

    def debug(self, message: str):
        self.logger.debug(message)
        self.logger.flush()

    def exception(self, message: str):
        self.logger.exception(message)
        self.json_log["errors"].append({
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        self._save_json_log()
        self.logger.flush()

    # New methods for structured logging
    def set_query(self, query: str):
        """Set the main query for this run"""
        self.json_log["query"] = query
        self._save_json_log()

    def add_planner_step(self, step: dict):
        """Add a step to the planner's steps"""
        self.json_log["planner"]["steps"].append(step)
        self._save_json_log()

    def set_planner_output(self, output: str):
        """Set the planner's output"""
        self.json_log["planner"]["output"] = output
        self._save_json_log()

    def add_scraped_data(self, url: str, content: str, metadata: dict = None):
        """Add scraped data to the researcher's data"""
        self.json_log["researcher"]["scraped_data"].append({
            "url": url,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        self._save_json_log()

    def set_researcher_output(self, output: str):
        """Set the researcher's output"""
        self.json_log["researcher"]["output"] = output
        self._save_json_log()

    def set_synthesized_data(self, data: dict):
        """Set the synthesized data"""
        self.json_log["researcher"]["synthesized_data"] = data
        self._save_json_log()

    def set_coder_output(self, output: str):
        """Set the coder's output"""
        self.json_log["coder"]["output"] = output
        self._save_json_log()

    def set_generated_code(self, code: str, language: str = None, file_path: str = None):
        """Set the generated code"""
        self.json_log["coder"]["generated_code"] = {
            "code": code,
            "language": language,
            "file_path": file_path,
            "timestamp": datetime.now().isoformat()
        }
        self._save_json_log()

    def set_execution_results(self, results: dict):
        """Set the execution results"""
        self.json_log["coder"]["execution_results"] = results
        self._save_json_log()

    def add_error(self, error: str):
        """Add an error to the log"""
        self.json_log["errors"].append({
            "message": error,
            "timestamp": datetime.now().isoformat()
        })
        self._save_json_log()

    def add_warning(self, warning: str):
        """Add a warning to the log"""
        self.json_log["warnings"].append({
            "message": warning,
            "timestamp": datetime.now().isoformat()
        })
        self._save_json_log()

    def set_execution_time(self, start_time: datetime, end_time: datetime):
        """Set the execution time"""
        self.json_log["execution_time"] = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds()
        }
        self._save_json_log()

    def get_json_log_path(self) -> str:
        """Get the path to the current JSON log file"""
        return str(self.json_log_path)


# def route_logger(logger: Logger):
#     """
#     Decorator factory that creates a decorator to log route entry and exit points.
#     The decorator uses the provided logger to log the information.
#
#     :param logger: The logger instance to use for logging.
#     """
#
#     log_enabled = Config().get_logging_rest_api()
#
#     def decorator(func):
#
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             # Log entry point
#             if log_enabled:
#                 logger.info(f"{request.path} {request.method}")
#
#             # Call the actual route function
#             response = func(*args, **kwargs)
#
#             from werkzeug.wrappers import Response
#
#             # Log exit point, including response summary if possible
#             try:
#                 if log_enabled:
#                     if isinstance(response, Response) and response.direct_passthrough:
#                         logger.debug(f"{request.path} {request.method} - Response: File response")
#                     else:
#                         response_summary = response.get_data(as_text=True)
#                         if 'settings' in request.path:
#                             response_summary = "*** Settings are not logged ***"
#                         logger.debug(f"{request.path} {request.method} - Response: {response_summary}")
#             except Exception as e:
#                 logger.exception(f"{request.path} {request.method} - {e})")
#
#             return response
#         return wrapper
#     return decorator


if __name__ == "__main__":
    # Real, practical example usage of the Logger
    try:
        logger = Logger()
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.debug("This is a debug message.")
        logger.exception("This is an exception message.")
        print("Log file content:")
        print(logger.read_log_file())
    except Exception as e:
        print(f"Error in logger example: {str(e)}")
