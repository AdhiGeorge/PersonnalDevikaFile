import json
import os
from datetime import datetime
from typing import Optional, Any
from sqlmodel import Field, Session, SQLModel, create_engine
from src.config import Config


class AgentStateModel(SQLModel, table=True):
    __tablename__ = "agent_state"

    id: Optional[int] = Field(default=None, primary_key=True)
    project: str
    state_stack_json: str


class AgentState:
    def __init__(self):
        config = Config()
        sqlite_path = config.get_sqlite_db()
        # Ensure the directory for the SQLite database exists
        db_dir = os.path.dirname(sqlite_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{sqlite_path}")
        SQLModel.metadata.create_all(self.engine)

    def new_state(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "internal_monologue": '',
            "browser_session": {
                "url": None,
                "screenshot": None
            },
            "terminal_session": {
                "command": None,
                "output": None,
                "title": None
            },
            "step": int(),
            "message": None,
            "completed": False,
            "agent_is_active": True,
            "token_usage": 0,
            "timestamp": timestamp,
            "step_results": {}  # Add step_results to store results for each step
        }

    def create_state(self, project: str):
        with Session(self.engine) as session:
            new_state = self.new_state()
            new_state["step"] = 1
            new_state["internal_monologue"] = "I'm starting the work..."
            agent_state = AgentStateModel(project=project, state_stack_json=json.dumps([new_state]))
            session.add(agent_state)
            session.commit()

    def delete_state(self, project: str):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).all()
            if agent_state:
                for state in agent_state:
                    session.delete(state)
                session.commit()

    def add_to_current_state(self, project: str, state: dict):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                state_stack = json.loads(agent_state.state_stack_json)
                state_stack.append(state)
                agent_state.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [state]
                agent_state = AgentStateModel(project=project, state_stack_json=json.dumps(state_stack))
                session.add(agent_state)
                session.commit()

    def get_current_state(self, project: str):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                return json.loads(agent_state.state_stack_json)
            return None

    def update_latest_state(self, project: str, state: dict):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                state_stack = json.loads(agent_state.state_stack_json)
                state_stack[-1] = state
                agent_state.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [state]
                agent_state = AgentStateModel(project=project, state_stack_json=json.dumps(state_stack))
                session.add(agent_state)
                session.commit()

    def get_latest_state(self, project: str):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                return json.loads(agent_state.state_stack_json)[-1]
            return None

    def set_agent_active(self, project: str, is_active: bool):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                state_stack = json.loads(agent_state.state_stack_json)
                state_stack[-1]["agent_is_active"] = is_active
                agent_state.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [self.new_state()]
                state_stack[-1]["agent_is_active"] = is_active
                agent_state = AgentStateModel(project=project, state_stack_json=json.dumps(state_stack))
                session.add(agent_state)
                session.commit()

    def is_agent_active(self, project: str):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                return json.loads(agent_state.state_stack_json)[-1]["agent_is_active"]
            return None

    def set_agent_completed(self, project: str, is_completed: bool):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                state_stack = json.loads(agent_state.state_stack_json)
                state_stack[-1]["internal_monologue"] = "Agent has completed the task."
                state_stack[-1]["completed"] = is_completed
                agent_state.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [self.new_state()]
                state_stack[-1]["completed"] = is_completed
                agent_state = AgentStateModel(project=project, state_stack_json=json.dumps(state_stack))
                session.add(agent_state)
                session.commit()

    def is_agent_completed(self, project: str):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                return json.loads(agent_state.state_stack_json)[-1]["completed"]
            return None
            
    def update_token_usage(self, project: str, token_usage: int):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                state_stack = json.loads(agent_state.state_stack_json)
                state_stack[-1]["token_usage"] += token_usage
                agent_state.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [self.new_state()]
                state_stack[-1]["token_usage"] = token_usage
                agent_state = AgentStateModel(project=project, state_stack_json=json.dumps(state_stack))
                session.add(agent_state)
                session.commit()

    def get_latest_token_usage(self, project: str):
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                return json.loads(agent_state.state_stack_json)[-1]["token_usage"]
            return 0

    def add_step_result(self, project: str, step_id: int, result: Any):
        """Add a result for a specific step."""
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                state_stack = json.loads(agent_state.state_stack_json)
                if "step_results" not in state_stack[-1]:
                    state_stack[-1]["step_results"] = {}
                state_stack[-1]["step_results"][str(step_id)] = result
                agent_state.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [self.new_state()]
                state_stack[-1]["step_results"][str(step_id)] = result
                agent_state = AgentStateModel(project=project, state_stack_json=json.dumps(state_stack))
                session.add(agent_state)
                session.commit()

    def get_step_result(self, project: str, step_id: int) -> Any:
        """Get the result for a specific step."""
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                state_stack = json.loads(agent_state.state_stack_json)
                return state_stack[-1].get("step_results", {}).get(str(step_id))
            return None

    def is_step_completed(self, project: str, step_id: int) -> bool:
        """Check if a specific step is completed."""
        return self.get_step_result(project, step_id) is not None

    def get_all_step_results(self, project: str) -> dict:
        """Get all step results."""
        with Session(self.engine) as session:
            agent_state = session.query(AgentStateModel).filter(AgentStateModel.project == project).first()
            if agent_state:
                state_stack = json.loads(agent_state.state_stack_json)
                return state_stack[-1].get("step_results", {})
            return {}

if __name__ == "__main__":
    # Real, practical example usage of the AgentState
    try:
        sm = AgentState()
        project_name = "Test Project"
        # Create a new state for the project
        sm.create_state(project_name)
        print(f"Created state for project: {project_name}")
        # Add a new state
        new_state = sm.new_state()
        new_state["internal_monologue"] = "Working on the first step."
        sm.add_to_current_state(project_name, new_state)
        print("Added a new state.")
        # Retrieve the current state stack
        state_stack = sm.get_current_state(project_name)
        print("\nCurrent State Stack:")
        for state in state_stack:
            print(state)
        # Update the latest state
        latest = sm.get_latest_state(project_name)
        latest["internal_monologue"] = "Step completed."
        sm.update_latest_state(project_name, latest)
        print("Updated the latest state.")
        # Mark agent as completed
        sm.set_agent_completed(project_name, True)
        print("Marked agent as completed.")
        # Clean up: delete the state
        sm.delete_state(project_name)
        print(f"Deleted state for project: {project_name}")
    except Exception as e:
        print(f"Error in state manager example: {str(e)}")