import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import uuid

from ..agents.planner.planner import Planner, SubQuery, QueryType
from ..agents.researcher.researcher import Researcher, ResearchResult
from ..agents.coder.coder import Coder
from ..utils.file_manager import FileManager
from ..knowledge_base.knowledge_base import KnowledgeBase
from ..utils.logger import get_logger
from ..agents.base_agent import BaseAgent
from Agentres.config.config import Config
from .workflow_context import WorkflowContext
from .workflow_state import WorkflowState
from ..agents.agent import Agent
from ..file_manager.file_manager import FileManager

logger = get_logger(__name__)

class WorkflowManager:
    """Manages the execution of research workflows."""
    
    def __init__(self, config: Config, agent: Agent):
        """Initialize the workflow manager.
        
        Args:
            config: Configuration instance
            agent: Agent instance
        """
        self.config = config
        self.agent = agent
        self.current_workflow = None
        self.file_manager = FileManager(config)
        self.context = None
        self.state = WorkflowState.INITIALIZED
        self.planner = None
        self.researcher = None
        self.coder = None
        self._is_running = False
        self._stop_requested = False
        self._init_lock = asyncio.Lock()
        
    async def initialize(self) -> bool:
        """Initialize the workflow manager.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing workflow manager...")
            await self.file_manager.initialize()
            
            # Initialize planner
            self.planner = Planner(config=self.config, model=self.agent.model)
            await self.planner.initialize()
            
            logger.info("Workflow manager initialized (no active workflow)")
            return True
        except Exception as e:
            error_msg = f"Failed to initialize workflow manager: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
    async def initialize_workflow(self, query: str) -> bool:
        """Initialize a new workflow with the given query.
        
        Args:
            query: The initial query for the workflow
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Create a new workflow
            self.current_workflow = {
                'query': query,
                'status': 'initialized',
                'start_time': datetime.now().isoformat(),
                'steps': [],
                'results': {}
            }
            
            # Initialize workflow components
            if not self.agent._initialized:
                if not await self.agent.initialize():
                    raise ValueError("Failed to initialize agent")
                
            # Initialize planner if not already initialized
            if not self.planner or not self.planner._initialized:
                self.planner = Planner(config=self.config, model=self.agent.model)
                await self.planner.initialize()
                
            # Log workflow initialization
            logger.info(f"Workflow initialized with query: {query}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize workflow: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def is_running(self) -> bool:
        """Check if a workflow is currently running.
        
        Returns:
            bool: True if workflow is running
        """
        return self._is_running
        
    async def stop(self) -> None:
        """Stop the current workflow."""
        if self._is_running:
            logger.info("Stopping workflow...")
            self._stop_requested = True
            self._is_running = False
            
            if self.current_workflow:
                self.current_workflow['status'] = 'stopped'
                self.current_workflow['end_time'] = datetime.now().isoformat()
                
            logger.info("Workflow stopped")
        else:
            logger.info("No workflow is running")
            
    def display_results(self) -> None:
        """Display the current workflow results."""
        if not self.current_workflow:
            print("No workflow results to display")
            return
            
        print("\nWorkflow Results:")
        print("=" * 50)
        
        # Display original query
        print("\nOriginal Query:")
        print("-" * 20)
        print(self.current_workflow['query'])
        
        # Display planning phase results
        if 'plan' in self.current_workflow:
            print("\nPlanning Phase:")
            print("-" * 20)
            plan = self.current_workflow['plan']
            if isinstance(plan, dict):
                print("Plan Type:", plan.get('type', 'unknown'))
                print("\nSteps:")
                for step in plan.get('steps', []):
                    print(f"\nStep {step.get('id', 'unknown')}:")
                    print(f"Agent: {step.get('agent', 'unknown')}")
                    print(f"Description: {step.get('description', 'No description')}")
                    if step.get('queries'):
                        print("\nQueries:")
                        for i, query in enumerate(step.get('queries', []), 1):
                            print(f"{i}. {query}")
                    if step.get('dependencies'):
                        print("\nDependencies:", ", ".join(step.get('dependencies', [])))
                    print(f"Expected Output: {step.get('expected_output', 'No output specified')}")
                
                if 'final_answer' in plan:
                    print("\nFinal Answer:")
                    print(f"Agent: {plan['final_answer'].get('agent', 'unknown')}")
                    print(f"Description: {plan['final_answer'].get('description', 'No description')}")
                    print("Required Components:", ", ".join(plan['final_answer'].get('required_components', [])))
            else:
                print(plan)
                
        # Display research phase results
        if 'research_results' in self.current_workflow:
            print("\nResearch Phase:")
            print("-" * 20)
            results = self.current_workflow['research_results']
            if results:
                print(f"Found {len(results)} research results:")
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Source: {result.get('source', 'Unknown')}")
                    print(f"Content: {result.get('content', 'No content')[:200]}...")
                    print(f"Relevance Score: {result.get('relevance_score', 0.0)}")
                    if 'metadata' in result:
                        print("\nMetadata:")
                        if 'key_points' in result['metadata']:
                            print("Key Points:", ", ".join(result['metadata']['key_points']))
                        if 'formulas' in result['metadata']:
                            print("Formulas:", ", ".join(result['metadata']['formulas']))
                        if 'examples' in result['metadata']:
                            print("Examples:", ", ".join(result['metadata']['examples']))
            else:
                print("No research results found")
                
        # Display synthesis phase results
        if 'synthesis' in self.current_workflow:
            print("\nSynthesis Phase:")
            print("-" * 20)
            print(self.current_workflow['synthesis'])
            
        # Display code generation results
        if 'code' in self.current_workflow:
            print("\nCode Generation Phase:")
            print("-" * 20)
            print(self.current_workflow['code'])
            
            if 'code_file' in self.current_workflow:
                print(f"\nCode saved to: {self.current_workflow['code_file']}")
                
        # Display any errors
        if 'error' in self.current_workflow:
            print("\nErrors:")
            print("-" * 20)
            print(self.current_workflow['error'])
            
        # Display workflow status
        print("\nWorkflow Status:")
        print("-" * 20)
        print(f"Status: {self.current_workflow['status']}")
        print(f"Start Time: {self.current_workflow.get('start_time', 'N/A')}")
        print(f"End Time: {self.current_workflow.get('end_time', 'N/A')}")
        
        print("\nNext steps:")
        print("1. Use 'run_code' to execute the generated code")
        print("2. Use 'modify' to make changes to the code")
        print("3. Use 'exit' to quit")
        
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step.
        
        Args:
            step: The step to execute
            
        Returns:
            Dict with step execution results
        """
        step_type = step.get('type', '')
        agent = step.get('agent', '')
        
        try:
            if agent == 'researcher':
                return await self._execute_research_step(step)
            elif agent == 'coder':
                return await self._execute_code_step(step)
            else:
                return {'status': 'skipped', 'reason': f'Unknown agent type: {agent}'}
                
        except Exception as e:
            error_msg = f"Error executing step {step.get('id')}: {str(e)}"
            logger.error(error_msg)
            return {'status': 'failed', 'error': error_msg}
    
    async def _execute_research_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research step."""
        if not self.researcher:
            self.researcher = Researcher(config=self.config, model=self.agent.model)
            await self.researcher.initialize()
        
        # Execute each query in the step
        results = []
        for query in step.get('queries', []):
            try:
                result = await self.researcher.research(query)
                if result and 'content' in result:
                    results.append({
                        'query': query,
                        'result': result,
                        'status': 'completed'
                    })
            except Exception as e:
                logger.error(f"Error researching query '{query}': {str(e)}")
                results.append({
                    'query': query,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return {
            'type': 'research',
            'status': 'completed' if any(r['status'] == 'completed' for r in results) else 'failed',
            'results': results
        }
    
    async def _execute_code_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a code generation step."""
        if not self.coder:
            self.coder = Coder(config=self.config, model=self.agent.model)
            await self.coder.initialize()
        
        # Get dependencies
        dependencies = {}
        for dep_id in step.get('dependencies', []):
            dep_step = next((s for s in self.current_workflow['steps'] if s['id'] == dep_id), None)
            if dep_step and 'results' in dep_step:
                dependencies[dep_id] = dep_step['results']
        
        # Generate code
        try:
            code_result = await self.coder.generate_code(
                task=step.get('description', ''),
                requirements=step.get('queries', []),
                context=dependencies
            )
            
            # Save code to file
            code_file = None
            if code_result and 'code' in code_result:
                code_file = await self.file_manager.save_code(
                    code=code_result['code'],
                    filename=f"code_{step['id']}.py"
                )
            
            return {
                'type': 'code',
                'status': 'completed',
                'code': code_result.get('code', ''),
                'explanation': code_result.get('explanation', ''),
                'code_file': code_file
            }
            
        except Exception as e:
            error_msg = f"Code generation failed: {str(e)}"
            logger.error(error_msg)
            return {
                'type': 'code',
                'status': 'failed',
                'error': error_msg
            }
    
    async def run(self) -> bool:
        """Run the current workflow.
        
        Returns:
            bool: True if workflow completed successfully
        """
        try:
            if not self.current_workflow:
                raise ValueError("No workflow initialized")
                
            if self._is_running:
                raise ValueError("Workflow already running")
                
            self._is_running = True
            self._stop_requested = False
            self.current_workflow['status'] = 'running'
            
            # Start planning phase
            logger.info("Starting planning phase")
            try:
                plan = await self.planner.plan(self.current_workflow['query'])
                self.current_workflow['plan'] = plan
                logger.info("Planning phase completed")
            except Exception as e:
                error_msg = f"Planning phase failed: {str(e)}"
                logger.error(error_msg)
                self.current_workflow['error'] = error_msg
                self.current_workflow['status'] = 'failed'
                return False
            
            # Execute each step in the plan
            if 'steps' in plan and isinstance(plan['steps'], list):
                self.current_workflow['steps'] = []
                
                for step in plan['steps']:
                    if self._stop_requested:
                        logger.info("Workflow stopped by user")
                        self.current_workflow['status'] = 'stopped'
                        return False
                        
                    logger.info(f"Executing step: {step.get('id')} - {step.get('description', '')}")
                    
                    # Execute the step
                    step_result = await self._execute_step(step)
                    
                    # Store step results
                    step_result['id'] = step['id']
                    step_result['agent'] = step.get('agent', 'unknown')
                    step_result['description'] = step.get('description', '')
                    step_result['timestamp'] = datetime.now().isoformat()
                    
                    self.current_workflow['steps'].append(step_result)
                    
                    # Update workflow status based on step result
                    if step_result.get('status') == 'failed':
                        self.current_workflow['status'] = 'failed'
                        self.current_workflow['error'] = step_result.get('error', 'Step execution failed')
                        break
            
            # Generate final output if all steps completed
            if self.current_workflow['status'] == 'running':
                self.current_workflow['status'] = 'completed'
                
                # Generate final answer if specified in plan
                if 'final_answer' in plan:
                    await self._generate_final_answer(plan['final_answer'])
            
            return self.current_workflow['status'] == 'completed'
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.current_workflow:
                self.current_workflow['error'] = error_msg
                self.current_workflow['status'] = 'failed'
            return False
            
        finally:
            self._is_running = False
            if 'end_time' not in self.current_workflow:
                self.current_workflow['end_time'] = datetime.now().isoformat()
    
    async def _generate_final_answer(self, final_answer_spec: Dict[str, Any]) -> None:
        """Generate the final answer based on workflow results."""
        try:
            # Collect all relevant information from completed steps
            context = {
                'steps': self.current_workflow.get('steps', []),
                'query': self.current_workflow['query'],
                'requirements': final_answer_spec.get('required_components', [])
            }
            
            # Generate final answer using the appropriate agent
            if final_answer_spec.get('agent') == 'researcher' and self.researcher:
                result = await self.researcher.synthesize_research(
                    query=self.current_workflow['query'],
                    context=context
                )
                self.current_workflow['final_answer'] = result
                
                # Save final answer to file
                if 'content' in result:
                    self.current_workflow['text_file'] = await self.file_manager.save_text(
                        text=result['content'],
                        filename='final_answer.md'
                    )
            
            logger.info("Final answer generated successfully")
            
        except Exception as e:
            error_msg = f"Failed to generate final answer: {str(e)}"
            logger.error(error_msg)
            self.current_workflow['error'] = error_msg
            
    async def get_status(self) -> Dict[str, Any]:
        """Get the current workflow status.
        
        Returns:
            Dict[str, Any]: Current workflow status
        """
        if not self.current_workflow:
            return {'status': 'no_workflow'}
            
        return {
            'status': self.current_workflow['status'],
            'query': self.current_workflow['query'],
            'steps': self.current_workflow['steps'],
            'start_time': self.current_workflow.get('start_time'),
            'end_time': self.current_workflow.get('end_time'),
            'error': self.current_workflow.get('error'),
            'files': {
                'text': self.current_workflow.get('text_file'),
                'code': self.current_workflow.get('code_file')
            }
        }
