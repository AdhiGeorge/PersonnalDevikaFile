import subprocess
import tempfile
import os
import shutil
import threading
import time
import logging
from typing import List, Optional, Callable

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("terminal_runner")

class TerminalRunner:
    def __init__(self, timeout: int = 30, memory_limit_mb: int = 256):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb

    def run(self, command: List[str], input_text: Optional[str] = None) -> dict:
        """
        Run a command in a temporary directory with resource limits.
        Returns a dict with stdout, stderr, exit_code, and duration.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Running command: {' '.join(command)} in {tmpdir}")
            start_time = time.time()
            try:
                if os.name == 'nt':
                    # Windows: no preexec_fn, but can use psutil for monitoring
                    proc = subprocess.Popen(
                        command,
                        cwd=tmpdir,
                        stdin=subprocess.PIPE if input_text else None,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=False,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    # Unix: set resource limits in preexec_fn
                    import resource
                    def set_limits():
                        resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit_mb * 1024 * 1024, self.memory_limit_mb * 1024 * 1024))
                        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
                    proc = subprocess.Popen(
                        command,
                        cwd=tmpdir,
                        stdin=subprocess.PIPE if input_text else None,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=False,
                        preexec_fn=set_limits
                    )
                # Monitor process for timeout and memory
                timer = threading.Timer(self.timeout, proc.kill)
                timer.start()
                try:
                    stdout, stderr = proc.communicate(input=input_text.encode() if input_text else None)
                finally:
                    timer.cancel()
                duration = time.time() - start_time
                exit_code = proc.returncode
                # Optionally check memory usage with psutil
                if psutil and proc.pid:
                    try:
                        p = psutil.Process(proc.pid)
                        mem = p.memory_info().rss // (1024 * 1024)
                        logger.info(f"Process memory usage: {mem} MB")
                    except Exception:
                        pass
                logger.info(f"Command finished with exit code {exit_code} in {duration:.2f}s")
                return {
                    "stdout": stdout.decode(errors="replace"),
                    "stderr": stderr.decode(errors="replace"),
                    "exit_code": exit_code,
                    "duration": duration,
                }
            except Exception as e:
                logger.error(f"Error running command: {e}")
                return {
                    "stdout": "",
                    "stderr": str(e),
                    "exit_code": -1,
                    "duration": time.time() - start_time,
                }

    def run_stream(
        self,
        command: List[str],
        input_text: Optional[str] = None,
        on_update: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run *command* and stream output via *on_update* callback.

        The callback receives the **cumulative** output each time a new line is
        produced, allowing callers to push live updates (e.g. over Socket.IO).
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"[stream] Running command: {' '.join(command)} in {tmpdir}")
            start_time = time.time()

            try:
                popen_args = {
                    "args": command,
                    "cwd": tmpdir,
                    "stdin": subprocess.PIPE if input_text else None,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "bufsize": 1,  # line-buffered
                    "shell": False,
                }

                if os.name == "nt":
                    popen_args["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    import resource

                    def set_limits():
                        resource.setrlimit(
                            resource.RLIMIT_AS,
                            (self.memory_limit_mb * 1024 * 1024, self.memory_limit_mb * 1024 * 1024),
                        )
                        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))

                    popen_args["preexec_fn"] = set_limits

                proc = subprocess.Popen(**popen_args)

                if input_text:
                    proc.stdin.write(input_text)
                    proc.stdin.close()

                timer = threading.Timer(self.timeout, proc.kill)
                timer.start()

                output_lines: List[str] = []

                try:
                    # Stream line by line
                    for line in proc.stdout:
                        output_lines.append(line)
                        if on_update:
                            try:
                                on_update("".join(output_lines))
                            except Exception as cb_err:
                                logger.error(f"run_stream on_update callback error: {cb_err}")
                    proc.wait()
                finally:
                    timer.cancel()

                duration = time.time() - start_time
                exit_code = proc.returncode
                full_output = "".join(output_lines)

                logger.info(
                    f"[stream] Command finished with exit code {exit_code} in {duration:.2f}s"
                )

                return {
                    "stdout": full_output,
                    "stderr": "",  # stderr is merged into stdout above
                    "exit_code": exit_code,
                    "duration": duration,
                }

            except Exception as e:
                logger.error(f"[stream] Error running command: {e}")
                return {
                    "stdout": "",
                    "stderr": str(e),
                    "exit_code": -1,
                    "duration": time.time() - start_time,
                }