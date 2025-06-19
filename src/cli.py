#!/usr/bin/env python3
"""
Command-line interface for the AgentRes workflow system.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
import aiohttp

from config.config import Config
from agents.agent import Agent
from knowledge_base.knowledge_base import KnowledgeBase
from file_manager.file_manager import FileManager
from workflow.workflow_manager import WorkflowManager, WorkflowState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize console for rich output
console = Console()

class CLI:
    """Command-line interface for the AgentRes application."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.config = None
        self.knowledge_base = None
        self.file_manager = None
        self.agent = None
        self.workflow_manager = None
        self._load_env()
        
    def _load_env(self):
        """Load environment variables."""
        # Get the directory where this script is located
        src_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(src_dir, '.env')
        
        # Load environment variables from the .env file in the src directory
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
            
            # Debug: Print QDRANT_URL to verify it's loaded correctly
            qdrant_url = os.getenv('QDRANT_URL')
            logger.info(f"[DEBUG] QDRANT_URL from .env: {qdrant_url}")
        else:
            logger.warning(f"No .env file found at {env_path}")
            load_dotenv()  # Fallback to default .env loading
        
    async def initialize_components(self):
        """Initialize all components."""
        try:
            print("\nInitializing components...")
            
            # 1. Initialize configuration
            print("1. Initializing configuration...")
            if not self.config:
                self.config = Config()
                if not await self.config.initialize():
                    print("\nError: Failed to initialize configuration.")
                    print("\nConfiguration Status:")
                    print("-------------------")
                    print(f"Azure OpenAI API Key: {'✓ Present' if self.config.get('openai', 'azure_api_key') else '✗ Missing'}")
                    print(f"Azure OpenAI Endpoint: {'✓ Present' if self.config.get('openai', 'azure_endpoint') else '✗ Missing'}")
                    print(f"Output Directory: {self.config.get('output', 'directory')}")
                    print(f"Database Path: {self.config.get('database', 'sqlite_path')}")
                    
                    print("\nPlease check the following:")
                    print("1. Azure OpenAI API key is set in .env file")
                    print("2. Azure OpenAI endpoint is set in .env file")
                    print("3. Required directories exist and are writable")
                    
                    print("\nTroubleshooting steps:")
                    print("1. Check your .env file for correct API keys")
                    print("2. Ensure the output and database directories are writable")
                    print("3. Try running with administrator privileges if needed")
                    return False
                    
            # 2. Initialize knowledge base
            print("2. Initializing knowledge base...")
            if not self.knowledge_base:
                self.knowledge_base = KnowledgeBase(self.config)
                if not await self.knowledge_base.initialize():
                    print("\nError: Failed to initialize knowledge base.")
                    return False
                    
            # 3. Initialize file manager
            print("3. Initializing file manager...")
            if not self.file_manager:
                self.file_manager = FileManager(self.config)
                if not await self.file_manager.initialize():
                    print("\nError: Failed to initialize file manager.")
                    return False
                    
            # 4. Initialize base agent
            print("4. Initializing base agent...")
            if not self.agent:
                self.agent = Agent(self.config)
                if not await self.agent.initialize():
                    print("\nError: Failed to initialize base agent.")
                    return False
                    
            # 5. Initialize workflow manager
            print("5. Initializing workflow manager...")
            if not self.workflow_manager:
                self.workflow_manager = WorkflowManager(self.config, self.agent)
                if not await self.workflow_manager.initialize():
                    print("\nError: Failed to initialize workflow manager.")
                    return False
                    
            print("\nAll components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}", exc_info=True)
            print(f"\nError: Failed to initialize components: {str(e)}")
            return False
            
    async def cmd_new(self, args: List[str]) -> bool:
        """Create a new workflow.
        
        Args:
            args: Command arguments
            
        Returns:
            bool: True if command was successful
        """
        try:
            if not args:
                print("Error: Please provide a query")
                return False
                
            # Join all arguments to handle queries with spaces
            query = ' '.join(args)
            if not query.strip():
                print("Error: Query cannot be empty")
                return False
                
            print("\nInitializing components...")
            
            # Initialize configuration
            print("1. Initializing configuration...")
            config = Config()
            await config.initialize()
            
            # Initialize knowledge base
            print("2. Initializing knowledge base...")
            knowledge_base = KnowledgeBase(config)
            await knowledge_base.initialize()
            
            # Initialize file manager
            print("3. Initializing file manager...")
            file_manager = FileManager(config)
            await file_manager.initialize()
            
            # Initialize base agent
            print("4. Initializing base agent...")
            agent = Agent(config)
            await agent.initialize()
            
            # Initialize workflow manager
            print("5. Initializing workflow manager...")
            self.workflow_manager = WorkflowManager(config, agent)
            await self.workflow_manager.initialize()
            
            print("\nAll components initialized successfully!")
            
            # Initialize and start workflow
            if await self.workflow_manager.initialize_workflow(query):
                print(f"\nWorkflow initialized with query: {query}")
                print("\nStarting workflow execution...")
                
                # Run workflow
                try:
                    success = await self.workflow_manager.run()
                    if success:
                        # Display results
                        self.workflow_manager.display_results()
                    else:
                        print("\nWorkflow failed to complete")
                        
                except Exception as e:
                    print(f"\nError: {str(e)}")
                    
            return True
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            return False
            
    async def cmd_run_code(self, args: List[str]) -> bool:
        """Run the generated code.
        
        Args:
            args: Command arguments
            
        Returns:
            bool: True if command was successful
        """
        try:
            if not self.workflow_manager:
                print("Error: No workflow initialized")
                return False
                
            status = await self.workflow_manager.get_status()
            if not status['files']['code']:
                print("Error: No code file available")
                return False
                
            print(f"\nRunning code from: {status['files']['code']}")
            
            # Execute code based on file extension
            code_file = status['files']['code']
            if code_file.endswith('.py'):
                import subprocess
                result = subprocess.run(['python', code_file], capture_output=True, text=True)
                print("\nOutput:")
                print("-------")
                print(result.stdout)
                if result.stderr:
                    print("\nErrors:")
                    print("-------")
                    print(result.stderr)
            else:
                print(f"Error: Unsupported file type: {code_file}")
                
            return True
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            return False
            
    async def cmd_modify(self, args: List[str]) -> bool:
        """Modify the generated code.
        
        Args:
            args: Command arguments
            
        Returns:
            bool: True if command was successful
        """
        try:
            if not self.workflow_manager:
                print("Error: No workflow initialized")
                return False
                
            if not args:
                print("Error: Please provide modification instructions")
                return False
                
            status = await self.workflow_manager.get_status()
            if not status['files']['code']:
                print("Error: No code file available")
                return False
                
            # Get modification instructions
            instructions = ' '.join(args)
            print(f"\nModifying code based on: {instructions}")
            
            # Modify code using agent
            modified_code = await self.workflow_manager.agent.coder.modify_code(
                status['files']['code'],
                instructions
            )
            
            # Save modified code
            code_file = await self.workflow_manager.file_manager.save_code(
                modified_code,
                os.path.basename(status['files']['code'])
            )
            
            print(f"\nModified code saved to: {code_file}")
            print("\nModified Code:")
            print("-------------")
            print(modified_code)
            
            return True
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            return False
            
    async def run(self):
        """Run the CLI."""
        print("\nWelcome to AgentRes CLI!")
        print("Type 'help' for available commands.")
        
        while True:
            try:
                cmd_line = input("\nAgentRes> ").strip()
                if not cmd_line:
                    continue
                    
                cmd_parts = cmd_line.split()
                cmd = cmd_parts[0].lower()
                args = cmd_parts[1:] if len(cmd_parts) > 1 else []
                
                if cmd in ('exit', 'quit'):
                    if await self.cmd_exit(args):
                        break
                elif cmd == 'help':
                    self.cmd_help(args)
                elif cmd == 'new':
                    await self.cmd_new(args)
                elif cmd == 'status':
                    await self.cmd_status(args[0] if args else None)
                elif cmd == 'run':
                    await self.cmd_run(args[0] if args else None)
                elif cmd == 'stop':
                    await self.cmd_stop(args[0] if args else None)
                elif cmd == 'list':
                    await self.cmd_list(args[0] if args else None)
                elif cmd == 'show':
                    await self.cmd_show(args[0] if args else None)
                elif cmd == 'save':
                    await self.cmd_save(args[0] if args else 'workflow_state.json')
                elif cmd == 'load':
                    await self.cmd_load(args[0] if args else 'workflow_state.json')
                elif cmd == 'run_code':
                    await self.cmd_run_code(args)
                elif cmd == 'modify':
                    await self.cmd_modify(args)
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                logger.error(f"Command processing error: {str(e)}", exc_info=True)
                print(f"Error: {str(e)}")
                
        print("\nGoodbye!")

    # Command implementations
    
    async def cmd_exit(self, arg):
        """Exit the application."""
        if self.workflow_manager and self.workflow_manager.is_running():
            print("Stopping workflow before exit...")
            try:
                await self.workflow_manager.stop()
            except Exception as e:
                logger.error(f"Failed to stop workflow: {str(e)}", exc_info=True)
                print(f"Error: Failed to stop workflow: {str(e)}")
                
        print("Exiting...")
        return True
    
    def cmd_help(self, args):
        """Show help message."""
        help_text = """
Available Commands:

  help                 Show this help message
  exit                 Exit the application
  new <query>          Start a new workflow with the given query
  run                  Run the current workflow
  stop                 Stop the current workflow
  status               Show the status of the current workflow
  list                 List generated files or research results
  show <file>          Show the contents of a generated file
  save                 Save the current workflow state (default: workflow_state.json)
  load                 Load a workflow state from a file (default: workflow_state.json)
  run_code             Run the generated code
  modify               Modify the generated code

Examples:
  new "Create a Python script to analyze stock data"
  run
  list files
  show stock_analysis.py
  save my_workflow.json
  load my_workflow.json
"""
        print(help_text)
    
    async def cmd_run(self, arg):
        """Run the current workflow."""
        if not self.workflow_manager:
            print("Error: No workflow initialized. Use 'new <query>' first.")
            return
            
        if self.workflow_manager.is_running():
            print("Error: Workflow is already running.")
            return
            
        try:
            await self.workflow_manager.run()
        except Exception as e:
            logger.error(f"Failed to run workflow: {str(e)}", exc_info=True)
            print(f"Error: Failed to run workflow: {str(e)}")
            
    async def cmd_stop(self, arg):
        """Stop the current workflow."""
        if not self.workflow_manager:
            print("Error: No workflow initialized.")
            return
            
        if not self.workflow_manager.is_running():
            print("Error: No workflow is running.")
            return
            
        try:
            await self.workflow_manager.stop()
            print("Workflow stopped.")
        except Exception as e:
            logger.error(f"Failed to stop workflow: {str(e)}", exc_info=True)
            print(f"Error: Failed to stop workflow: {str(e)}")
            
    async def cmd_status(self, arg):
        """Show the current workflow status."""
        if not self.workflow_manager:
            print("Error: No workflow initialized.")
            return
            
        try:
            status = await self.workflow_manager.get_status()
            print("\nWorkflow Status:")
            print("----------------")
            print(f"Query: {status.get('query', 'None')}")
            print(f"State: {status.get('state', 'None')}")
            print(f"Running: {status.get('is_running', False)}")
            
            if status.get('errors'):
                print("\nErrors:")
                for error in status['errors']:
                    print(f"- {error}")
                    
            if status.get('warnings'):
                print("\nWarnings:")
                for warning in status['warnings']:
                    print(f"- {warning}")
                    
            if status.get('results'):
                print("\nResults:")
                for key, value in status['results'].items():
                    print(f"\n{key}:")
                    print(value)
                    
        except Exception as e:
            logger.error(f"Failed to get workflow status: {str(e)}", exc_info=True)
            print(f"Error: Failed to get workflow status: {str(e)}")
    
    async def cmd_list(self, type_filter: Optional[str] = None):
        """List generated files or research results."""
        if not self.workflow_manager:
            print("Error: No workflow initialized.")
            return
            
        try:
            files = await self.workflow_manager.list_files(type_filter)
            if not files:
                print("No files found.")
                return
                
            print("\nGenerated Files:")
            print("---------------")
            for file in files:
                print(f"- {file}")
                
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}", exc_info=True)
            print(f"Error: Failed to list files: {str(e)}")
    
    async def cmd_show(self, filename: str):
        """Show the contents of a generated file."""
        if not self.workflow_manager:
            print("Error: No workflow initialized.")
            return
            
        if not filename:
            print("Error: Please specify a filename.")
            return
            
        try:
            content = await self.workflow_manager.show_file(filename)
            if not content:
                print(f"File not found: {filename}")
                return
                
            print(f"\nContents of {filename}:")
            print("-------------------")
            print(content)
            
        except Exception as e:
            logger.error(f"Failed to show file: {str(e)}", exc_info=True)
            print(f"Error: Failed to show file: {str(e)}")
    
    async def cmd_save(self, filename: str = 'workflow_state.json'):
        """Save the current workflow state."""
        if not self.workflow_manager:
            print("Error: No workflow initialized.")
            return
            
        try:
            await self.workflow_manager.save_state(filename)
            print(f"Workflow state saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save workflow state: {str(e)}", exc_info=True)
            print(f"Error: Failed to save workflow state: {str(e)}")
    
    async def cmd_load(self, filename: str = 'workflow_state.json'):
        """Load a workflow state from a file."""
        if not self.workflow_manager:
            print("Error: No workflow initialized.")
            return
            
        try:
            await self.workflow_manager.load_state(filename)
            print(f"Workflow state loaded from {filename}")
            
            # Show the status of the loaded workflow
            await self.cmd_status(None)
            
        except Exception as e:
            logger.error(f"Failed to load workflow state: {str(e)}", exc_info=True)
            print(f"Error: Failed to load workflow state: {str(e)}")

async def shutdown():
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.sleep(0.1)
    loop = asyncio.get_event_loop()
    await loop.shutdown_asyncgens()
    async with aiohttp.ClientSession() as session:
        await session.close()

def main():
    """Main entry point for the CLI."""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            console.print("[red]Error: Python 3.8 or higher is required.[/red]")
            sys.exit(1)
            
        # Create and run the CLI
        cli = CLI()
        asyncio.run(cli.run())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        logger.exception("Fatal error in main")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(shutdown())
