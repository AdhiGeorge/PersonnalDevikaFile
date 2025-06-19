import os
import sys
import time
from datetime import datetime
from Agentres.utils.run_logger import RunLogger
from Agentres.agents.planner.planner import Planner
from Agentres.agents.researcher.researcher import Researcher
from Agentres.agents.coder.coder import Coder
from Agentres.config import Config
from Agentres.logger import Logger

def main():
    # Initialize configuration and logging
    config = Config()
    logger = Logger()
    run_logger = RunLogger(config.get_logs_dir())
    
    start_time = datetime.now()
    
    try:
        # Get user query
        query = input("Enter your query: ")
        run_logger.set_query(query)
        
        # Initialize agents
        planner = Planner()
        researcher = Researcher()
        coder = Coder()
        
        # Planning phase
        logger.info("Starting planning phase...")
        plan = planner.plan(query)
        run_logger.set_planner_output(plan)
        
        # Research phase
        logger.info("Starting research phase...")
        research_results = researcher.research(query, plan)
        run_logger.set_researcher_output(research_results)
        
        # Coding phase
        logger.info("Starting coding phase...")
        code_results = coder.implement(query, plan, research_results)
        run_logger.set_coder_output(code_results)
        
        # Log execution time
        end_time = datetime.now()
        run_logger.set_execution_time(start_time, end_time)
        
        logger.info(f"Run completed successfully. Log saved to: {run_logger.get_log_path()}")
        
    except Exception as e:
        run_logger.add_error(str(e), {"traceback": str(sys.exc_info())})
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 