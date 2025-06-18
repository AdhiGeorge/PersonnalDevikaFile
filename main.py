import asyncio
import logging
import os
import sys
from typing import Optional
from Agentres.config.config import Config
from Agentres.agents.agent import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s: %(name)s: %(message)s',
    datefmt='%y.%m.%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def validate_input(input_str: str) -> Optional[str]:
    """Validate user input."""
    if not input_str or not isinstance(input_str, str):
        return None
        
    input_str = input_str.strip()
    if not input_str:
        return None
        
    # Check for maximum length
    if len(input_str) > 1000:
        logger.warning("Input exceeds maximum length of 1000 characters")
        return None
        
    # Check for potentially harmful content
    harmful_patterns = [
        "rm -rf",
        "del /f /s /q",
        "format",
        "drop database",
        "delete from",
        "drop table"
    ]
    
    input_lower = input_str.lower()
    if any(pattern in input_lower for pattern in harmful_patterns):
        logger.warning("Input contains potentially harmful content")
        return None
        
    return input_str

def display_files(result: dict) -> None:
    """Display the generated files to the user."""
    try:
        if not result or not isinstance(result, dict):
            return
            
        files = result.get("files", {})
        if not files:
            return
            
        print("\nGenerated Files:")
        if files.get("response"):
            response_path = os.path.abspath(files["response"])
            if os.path.exists(response_path):
                print(f"Response: {response_path}")
                # Show file size
                size = os.path.getsize(response_path)
                print(f"Size: {size/1024:.1f} KB")
                
        if files.get("code"):
            code_path = os.path.abspath(files["code"])
            if os.path.exists(code_path):
                print(f"Code: {code_path}")
                # Show file size
                size = os.path.getsize(code_path)
                print(f"Size: {size/1024:.1f} KB")
                
    except Exception as e:
        logger.error(f"Error displaying files: {str(e)}")

async def handle_user_interaction(agent: Agent, result: dict) -> None:
    """Handle user interaction after initial response."""
    try:
        while True:
            print("\nWhat would you like to do?")
            print("1. Run the code")
            print("2. Modify the code")
            print("3. Ask a follow-up question")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                if result.get("files", {}).get("code"):
                    print("\nRunning code...")
                    run_result = await agent.run_code(result["files"]["code"])
                    print("\nOutput:")
                    print(run_result["output"])
                    if run_result["error"]:
                        print("\nErrors:")
                        print(run_result["error"])
                else:
                    print("\nNo code file available to run.")
                    
            elif choice == "2":
                if result.get("files", {}).get("code"):
                    print("\nPlease describe the modifications needed:")
                    modification = input("> ").strip()
                    if modification:
                        # TODO: Implement code modification
                        print("Code modification not implemented yet.")
                    else:
                        print("No modification specified.")
                else:
                    print("\nNo code file available to modify.")
                    
            elif choice == "3":
                print("\nEnter your follow-up question:")
                follow_up = input("> ").strip()
                if follow_up.lower() == "quit":
                    break
                    
                validated_input = validate_input(follow_up)
                if validated_input:
                    result = await agent.process_query(validated_input)
                    display_files(result)
                else:
                    print("Invalid input. Please try again.")
                    
            elif choice == "4":
                break
                
            else:
                print("\nInvalid choice. Please try again.")
                
    except Exception as e:
        logger.error(f"Error in user interaction: {str(e)}")
        print(f"\nAn error occurred: {str(e)}")

async def main():
    try:
        # Initialize configuration
        config = Config()
        await config.initialize()
        logger.info("Configuration initialized")
        
        # Initialize agent
        agent = Agent(config)
        try:
            await agent.initialize()
            logger.info("Agent initialized with all components")
        except Exception as e:
            error_msg = f"Failed to initialize agent: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Main interaction loop
        while True:
            query = input("\nEnter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                logger.info("Session ended")
                break
                
            logger.info(f"Processing query: {query}")
            try:
                response = await agent.process_query(query)
                print("\n" + response)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing query: {error_msg}")
                print(f"\nAn error occurred while processing your query: {error_msg}")
                print("Please try again with a different query or contact support if the issue persists.")
                
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Agent error: {error_msg}")
        print(f"\nA system error occurred: {error_msg}")
        print("Please check the logs for more details and contact support if needed.")
        logger.info("Session ended")

if __name__ == "__main__":
    asyncio.run(main())
