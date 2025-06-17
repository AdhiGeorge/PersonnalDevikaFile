from swarm import Swarm, Agent
from typing import List, Dict, Any, Optional
import json

class AgentManager:
    def __init__(self):
        self.client = Swarm()
        self.agents = {}
        self.context_variables = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents with their specific roles and capabilities."""
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
        """Handoff function to transfer control to the researcher agent."""
        return self.agents["researcher"]

    def _handoff_to_coder(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the coder agent."""
        return self.agents["coder"]

    def _handoff_to_runner(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the runner agent."""
        return self.agents["runner"]

    def _handoff_to_patcher(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the patcher agent."""
        return self.agents["patcher"]

    def _handoff_to_reporter(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the reporter agent."""
        return self.agents["reporter"]

    def _handoff_to_planner(self, context: Dict[str, Any]) -> Agent:
        """Handoff function to transfer control to the planner agent."""
        return self.agents["planner"]

    def run(self, 
            agent_name: str, 
            messages: List[Dict[str, str]], 
            context_variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run an agent with the given messages and context variables.
        
        Args:
            agent_name: Name of the agent to run
            messages: List of message objects
            context_variables: Optional dictionary of context variables
            
        Returns:
            Dictionary containing the response and updated state
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")

        if context_variables is None:
            context_variables = {}

        # Update context variables
        self.context_variables.update(context_variables)

        # Run the agent
        response = self.client.run(
            agent=self.agents[agent_name],
            messages=messages,
            context_variables=self.context_variables
        )

        # Update context variables with any changes
        self.context_variables.update(response.context_variables)

        return {
            "messages": response.messages,
            "agent": response.agent.name,
            "context_variables": response.context_variables
        }

if __name__ == "__main__":
    # Real, practical example usage of the AgentManager
    try:
        manager = AgentManager()
        # Example: User wants to create a project plan for a REST API
        messages = [
            {"role": "user", "content": "Create a project plan for a REST API that manages a todo list."}
        ]
        context_variables = {
            "research_needed": True,
            "code_generation_needed": True,
            "execution_needed": False,
            "errors_found": False,
            "task_completed": False
        }
        # Run the planner agent
        result = manager.run(
            agent_name="planner",
            messages=messages,
            context_variables=context_variables
        )
        print("\nAgentManager Result:")
        print(f"Agent: {result['agent']}")
        print("Messages:")
        for msg in result["messages"]:
            print(f"  {msg}")
        print("Context Variables:")
        for k, v in result["context_variables"].items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error in agent manager example: {str(e)}")
