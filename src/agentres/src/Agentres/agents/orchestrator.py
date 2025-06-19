from typing import Dict, List, Any, Optional, Callable
from swarm import Swarm, Agent
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentContext:
    """Context shared between agents during orchestration"""
    state: AgentState
    messages: List[Dict[str, str]]
    variables: Dict[str, Any]
    metadata: Dict[str, Any]

class AgentOrchestrator:
    def __init__(self):
        self.swarm = Swarm()
        self.agents: Dict[str, Agent] = {}
        self.handoff_rules: Dict[str, List[Callable]] = {}
        self.context: Optional[AgentContext] = None
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents with their roles and capabilities"""
        # Planner Agent
        self.agents["planner"] = Agent(
            name="Planner",
            instructions="""You are a planning agent responsible for breaking down complex tasks into manageable steps.
            Create clear, actionable plans that can be executed by other agents.
            Focus on clarity and feasibility in your planning.""",
            functions=[self._handoff_to_researcher]
        )

        # Researcher Agent
        self.agents["researcher"] = Agent(
            name="Researcher",
            instructions="""You are a research agent responsible for gathering information and knowledge.
            Search for relevant information and compile it in a structured format.
            Focus on accuracy and relevance in your research.""",
            functions=[self._handoff_to_coder]
        )

        # Coder Agent
        self.agents["coder"] = Agent(
            name="Coder",
            instructions="""You are a coding agent responsible for implementing solutions.
            Write clean, efficient, and well-documented code.
            Follow best practices and coding standards.""",
            functions=[self._handoff_to_runner]
        )

        # Runner Agent
        self.agents["runner"] = Agent(
            name="Runner",
            instructions="""You are a runner agent responsible for executing code and commands.
            Run code safely and handle any errors that occur.
            Provide clear feedback about execution results.""",
            functions=[self._handoff_to_patcher]
        )

        # Patcher Agent
        self.agents["patcher"] = Agent(
            name="Patcher",
            instructions="""You are a patcher agent responsible for fixing issues in code.
            Identify and resolve bugs and problems.
            Ensure code quality and functionality.""",
            functions=[self._handoff_to_reporter]
        )

        # Reporter Agent
        self.agents["reporter"] = Agent(
            name="Reporter",
            instructions="""You are a reporter agent responsible for documenting and summarizing work.
            Create clear, concise reports of actions taken and results achieved.
            Focus on clarity and completeness in your reporting.""",
            functions=[self._handoff_to_planner]
        )

    def _handoff_to_researcher(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the researcher agent"""
        logger.info("Handing off to Researcher agent")
        return self.agents["researcher"]

    def _handoff_to_coder(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the coder agent"""
        logger.info("Handing off to Coder agent")
        return self.agents["coder"]

    def _handoff_to_runner(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the runner agent"""
        logger.info("Handing off to Runner agent")
        return self.agents["runner"]

    def _handoff_to_patcher(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the patcher agent"""
        logger.info("Handing off to Patcher agent")
        return self.agents["patcher"]

    def _handoff_to_reporter(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the reporter agent"""
        logger.info("Handing off to Reporter agent")
        return self.agents["reporter"]

    def _handoff_to_planner(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the planner agent"""
        logger.info("Handing off to Planner agent")
        return self.agents["planner"]

    def _determine_next_agent(self, current_agent: str, context: Dict[str, Any]) -> str:
        """
        Determine the next agent based on the current context and state.
        This implements the A2A protocol for dynamic handoffs.
        """
        # Example logic for dynamic handoff determination
        if current_agent == "planner":
            if "research_needed" in context and context["research_needed"]:
                return "researcher"
            return "coder"
        
        elif current_agent == "researcher":
            if "code_generation_needed" in context and context["code_generation_needed"]:
                return "coder"
            return "planner"
        
        elif current_agent == "coder":
            if "execution_needed" in context and context["execution_needed"]:
                return "runner"
            return "planner"
        
        elif current_agent == "runner":
            if "errors_found" in context and context["errors_found"]:
                return "patcher"
            return "reporter"
        
        elif current_agent == "patcher":
            return "reporter"
        
        elif current_agent == "reporter":
            return "planner"
        
        return "planner"  # Default fallback

    def run(self, 
            start_agent: str, 
            messages: List[Dict[str, str]], 
            context_variables: Optional[Dict[str, Any]] = None,
            max_turns: int = 10) -> Dict[str, Any]:
        """
        Run the agent orchestration starting with the specified agent.
        
        Args:
            start_agent: Name of the agent to start with
            messages: Initial messages to process
            context_variables: Optional context variables
            max_turns: Maximum number of agent handoffs
            
        Returns:
            Dictionary containing the final response and state
        """
        if start_agent not in self.agents:
            raise ValueError(f"Unknown agent: {start_agent}")

        # Initialize context
        self.context = AgentContext(
            state=AgentState.ACTIVE,
            messages=messages,
            variables=context_variables or {},
            metadata={"turn_count": 0}
        )

        current_agent = start_agent
        turn_count = 0

        while turn_count < max_turns:
            logger.info(f"Turn {turn_count + 1}: Running {current_agent} agent")
            
            try:
                # Run the current agent
                response = self.swarm.run(
                    agent=self.agents[current_agent],
                    messages=self.context.messages,
                    context_variables=self.context.variables
                )

                # Update context with response
                self.context.messages.extend(response.messages)
                self.context.variables.update(response.context_variables)
                
                # Determine next agent based on context
                next_agent = self._determine_next_agent(current_agent, self.context.variables)
                
                # Check if we should continue or stop
                if self._should_stop_orchestration(next_agent, self.context):
                    self.context.state = AgentState.COMPLETED
                    break
                
                current_agent = next_agent
                turn_count += 1
                self.context.metadata["turn_count"] = turn_count

            except Exception as e:
                logger.error(f"Error during orchestration: {str(e)}")
                self.context.state = AgentState.ERROR
                self.context.metadata["error"] = str(e)
                break

        return {
            "messages": self.context.messages,
            "context_variables": self.context.variables,
            "state": self.context.state,
            "metadata": self.context.metadata
        }

    def _should_stop_orchestration(self, next_agent: str, context: Dict[str, Any]) -> bool:
        """
        Determine if the orchestration should stop based on context and next agent.
        """
        # Example conditions for stopping:
        if "task_completed" in context and context["task_completed"]:
            return True
        
        if "error_occurred" in context and context["error_occurred"]:
            return True
        
        if next_agent == "reporter" and "report_generated" in context:
            return True
        
        return False

    def get_agent_state(self) -> Dict[str, Any]:
        """Get the current state of the orchestration"""
        if not self.context:
            return {"state": AgentState.IDLE}
        
        return {
            "state": self.context.state,
            "turn_count": self.context.metadata.get("turn_count", 0),
            "variables": self.context.variables,
            "metadata": self.context.metadata
        }

if __name__ == "__main__":
    # Real, practical example usage of the AgentOrchestrator
    try:
        orchestrator = AgentOrchestrator()
        # Example: User wants to build a FastAPI web server
        messages = [
            {"role": "user", "content": "Build a FastAPI web server that returns 'Hello, World!' on the root endpoint."}
        ]
        context_variables = {
            "research_needed": True,
            "code_generation_needed": True,
            "execution_needed": True,
            "errors_found": False,
            "task_completed": False
        }
        result = orchestrator.run(
            start_agent="planner",
            messages=messages,
            context_variables=context_variables,
            max_turns=6
        )
        print("\nOrchestration Result:")
        print(f"State: {result['state']}")
        print(f"Turn Count: {result['metadata'].get('turn_count')}")
        print("Messages:")
        for msg in result["messages"]:
            print(f"  {msg}")
        print("Context Variables:")
        for k, v in result["context_variables"].items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error in orchestrator example: {str(e)}")
