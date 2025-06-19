import asyncio
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from Agentres.config import Config
from Agentres.llm.llm import LLM
from Agentres.agents.agent import Agent
from Agentres.utils.token_tracker import TokenTracker
from Agentres.logger import Logger

# Initialize our custom logger
logger = Logger()

class AgentError(Exception):
    """Custom exception for agent-related errors."""
    pass

async def initialize_components() -> tuple[Config, TokenTracker, LLM, Agent]:
    """Initialize all required components with proper error handling."""
    try:
        # Initialize configuration
        config = Config()
        logger.info("Configuration initialized")

        # Initialize token tracker with config
        token_tracker = TokenTracker(config)
        logger.info("Token tracker initialized")

        # Initialize LLM
        llm = LLM(config)
        logger.info(f"LLM initialized with model: {config.model}")

        # Initialize agent
        agent = Agent(config)
        logger.info("Agent initialized with all components")

        return config, token_tracker, llm, agent
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise AgentError(f"Initialization failed: {str(e)}")

def format_result(result: Dict[str, Any]) -> str:
    """Format the result in a clear and structured way."""
    output = []
    
    # Format answer
    if 'answer' in result:
        if isinstance(result['answer'], dict):
            output.append("\nAnswer:")
            if 'summary' in result['answer']:
                output.append(f"\nSummary: {result['answer']['summary']}")
            if 'key_points' in result['answer']:
                output.append("\nKey Points:")
                for point in result['answer']['key_points']:
                    output.append(f"- {point}")
            if 'implementation_details' in result['answer']:
                output.append(f"\nImplementation Details: {result['answer']['implementation_details']}")
        else:
            output.append(f"\nAnswer: {result['answer']}")

    # Format code
    if 'code' in result:
        if isinstance(result['code'], dict):
            output.append("\nCode:")
            if 'implementation' in result['code']:
                output.append(f"\n{result['code']['implementation']}")
            if 'dependencies' in result['code']:
                output.append(f"\nDependencies: {', '.join(result['code']['dependencies'])}")
            if 'setup_instructions' in result['code']:
                output.append(f"\nSetup Instructions: {result['code']['setup_instructions']}")
        else:
            output.append(f"\nCode: {result['code']}")

    # Format metadata
    if 'metadata' in result:
        output.append("\nMetadata:")
        if 'sources' in result['metadata']:
            output.append(f"\nSources: {', '.join(result['metadata']['sources'])}")
        if 'confidence' in result['metadata']:
            output.append(f"Confidence: {result['metadata']['confidence']}")
        if 'explanation' in result['metadata']:
            output.append(f"Explanation: {result['metadata']['explanation']}")
        if 'coverage' in result['metadata']:
            output.append(f"Coverage: {result['metadata']['coverage']}")

    return "\n".join(output)

async def main():
    """Main function to run the agent with enhanced error handling and logging."""
    start_time = datetime.now()
    try:
        # Initialize components
        config, token_tracker, llm, agent = await initialize_components()

        # Get user input with validation
        while True:
            prompt = input("\nEnter your query (or 'quit' to exit): ").strip()
            if prompt.lower() == 'quit':
                break
            if not prompt:
                print("Please enter a valid query.")
                continue

            try:
                # Set the query in the logger
                logger.set_query(prompt)
                
                # Execute agent
                logger.info(f"Processing query: {prompt}")
                result = await agent.execute(prompt)

                # Log the results
                if 'answer' in result:
                    logger.set_planner_output(str(result['answer']))
                if 'code' in result:
                    logger.set_generated_code(str(result['code']))
                if 'metadata' in result:
                    logger.set_researcher_output(str(result['metadata']))

                # Format and display results
                print(format_result(result))

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"\nAn error occurred while processing your query: {str(e)}")
                print("Please try again with a different query or contact support if the issue persists.")

    except AgentError as e:
        logger.error(f"Agent error: {str(e)}")
        print(f"\nA system error occurred: {str(e)}")
        print("Please check the logs for more details and contact support if needed.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print("\nAn unexpected error occurred. Please check the logs for more details.")
    finally:
        end_time = datetime.now()
        logger.set_execution_time(start_time, end_time)
        logger.info("Session ended")

if __name__ == "__main__":
    asyncio.run(main())
