"""Standalone CLI for Devika.

This script lets you interact with the Devika agent entirely from the
terminal, without running the Flask/Socket.IO backend.  It replicates the
basic flow used by `main.py` but prompts for inputs and prints results
inline.

Run it from the project root (where `config.yaml` lives):

    (venv) $ python terminal.py

Press Ctrl-C or type `exit` / `quit` to leave.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from src.agents.agent import Agent
from src.config import Config
from src.project import ProjectManager
from src.logger import Logger


def _prompt(text: str, default: Optional[str] = None) -> str:
    """Prompt the user, returning their input or *default* if blank."""
    value = input(text).strip()
    return value or (default or "")


async def interactive_cli() -> None:
    banner = (
        "=" * 60
        + "\nDevika – Terminal mode\n"
        + "=" * 60
    )
    print(banner)

    # Load environment variables first so they can override config.yaml
    load_dotenv()

    cfg = Config()
    logger = Logger()
    pm = ProjectManager()

    # ------------------------------------------------------------------
    # 1. Choose / create project
    # ------------------------------------------------------------------
    default_project = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_name = _prompt(f"Project name [{default_project}]: ", default_project)
    try:
        existing_projects = set(pm.get_project_list())
    except Exception:
        existing_projects = set()

    if project_name not in existing_projects:
        pm.create_project(project_name)
        print(f"Created new project '{project_name}'.")
    else:
        print(f"Using existing project '{project_name}'.")

    # ------------------------------------------------------------------
    # 2. Select base model
    # ------------------------------------------------------------------
    # Determine a sensible default model: prefer env var, then config, then GPT-4o
    default_model = os.getenv("AZURE_OPENAI_DEPLOYMENT") or cfg.get("LLM.model", "gpt-4o")

    base_model = _prompt("Base LLM (leave blank for config default): ", default_model)

    # ------------------------------------------------------------------
    # 3. Instantiate agent
    # ------------------------------------------------------------------
    agent = Agent(base_model=base_model, search_engine=cfg.search_engines.primary)

    print("\nEnter your prompt. Type 'exit' to quit.\n")

    while True:
        try:
            user_msg = input(">>> ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in {"exit", "quit"}:
                break

            # ----------------------------------------------------------
            # 4. Primary agent response (LLM inference)
            # ----------------------------------------------------------
            response = await agent.execute(user_msg, project_name)
            print("\n--- Initial Agent Reply ---")
            print(response)

            # ----------------------------------------------------------
            # 5. Trigger the full multi-agent orchestration automatically
            # ----------------------------------------------------------
            print("\nRunning multi-agent pipeline...\n")
            agent.subsequent_execute(user_msg, project_name)
            print("\nPipeline completed.\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as exc:
            logger.error(str(exc))
            print(f"Error: {exc}\n")

    # ------------------------------------------------------------------
    # 5. Offer to zip project on exit
    # ------------------------------------------------------------------
    project_dir = pm.get_project_path(project_name)
    if os.path.isdir(project_dir):
        zip_choice = _prompt("Zip project for download? [y/N]: ", "n").lower()
        if zip_choice == "y":
            pm.project_to_zip(project_name)
            print(f"Archive created at: {pm.get_zip_path(project_name)}")


if __name__ == "__main__":
    asyncio.run(interactive_cli())
