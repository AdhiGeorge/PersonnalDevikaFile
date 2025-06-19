import subprocess
import tempfile
import os
import shutil
import threading
import time
import logging
import asyncio
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
        self._initialized = False

    async def initialize(self):
        """Initialize async components."""
        try:
            # Check if psutil is available for memory monitoring
            if psutil is None:
                logger.warning("psutil not available - memory monitoring will be disabled")
            
            # Create a temporary directory to test permissions
            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = os.path.join(tmpdir, "test.txt")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            
            self._initialized = True
            logger.info("TerminalRunner async components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize async components: {str(e)}")
            raise ValueError(f"Async initialization failed: {str(e)}")

    async def run(self, command: List[str], input_text: Optional[str] = None) -> dict:
        """
        Run a command in a temporary directory with resource limits.
        Returns a dict with stdout, stderr, exit_code, and duration.
        """
        try:
            # Ensure async components are initialized
            if not self._initialized:
                await self.initialize()
                
            with tempfile.TemporaryDirectory() as tmpdir:
                logger.info(f"Running command: {' '.join(command)} in {tmpdir}")
                start_time = time.time()
                try:
                    if os.name == 'nt':
                        # Windows: no preexec_fn, but can use psutil for monitoring
                        proc = await asyncio.create_subprocess_exec(
                            *command,
                            cwd=tmpdir,
                            stdin=asyncio.subprocess.PIPE if input_text else None,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                        )
                    else:
                        # Unix: set resource limits in preexec_fn
                        import resource
                        def set_limits():
                            resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit_mb * 1024 * 1024, self.memory_limit_mb * 1024 * 1024))
                            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
                        proc = await asyncio.create_subprocess_exec(
                            *command,
                            cwd=tmpdir,
                            stdin=asyncio.subprocess.PIPE if input_text else None,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            preexec_fn=set_limits
                        )
                    # Monitor process for timeout and memory
                    try:
                        if input_text:
                            await proc.stdin.write(input_text.encode())
                            await proc.stdin.drain()
                            proc.stdin.close()
                        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        raise TimeoutError(f"Command timed out after {self.timeout} seconds")
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
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "duration": time.time() - start_time,
            }

    async def run_stream(
        self,
        command: List[str],
        input_text: Optional[str] = None,
        on_update: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run *command* and stream output via *on_update* callback.

        The callback receives the **cumulative** output each time a new line is
        produced, allowing callers to push live updates.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"[stream] Running command: {' '.join(command)} in {tmpdir}")
            start_time = time.time()

            try:
                if os.name == "nt":
                    proc = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=tmpdir,
                        stdin=asyncio.subprocess.PIPE if input_text else None,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    import resource
                    def set_limits():
                        resource.setrlimit(
                            resource.RLIMIT_AS,
                            (self.memory_limit_mb * 1024 * 1024, self.memory_limit_mb * 1024 * 1024),
                        )
                        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
                    proc = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=tmpdir,
                        stdin=asyncio.subprocess.PIPE if input_text else None,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        preexec_fn=set_limits
                    )

                if input_text:
                    await proc.stdin.write(input_text.encode())
                    await proc.stdin.drain()
                    proc.stdin.close()

                output_lines: List[str] = []

                try:
                    # Stream line by line
                    async for line in proc.stdout:
                        output_lines.append(line.decode())
                        if on_update:
                            try:
                                on_update("".join(output_lines))
                            except Exception as cb_err:
                                logger.error(f"run_stream on_update callback error: {cb_err}")
                    await proc.wait()
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    raise TimeoutError(f"Command timed out after {self.timeout} seconds")

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