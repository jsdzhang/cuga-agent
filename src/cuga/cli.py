#!/usr/bin/env python3
import os
import subprocess
import time
import typer
import signal
import sys
import threading
import platform
import psutil
import httpx
from typing import List, Optional
from loguru import logger
from cuga.config import settings, PACKAGE_ROOT, get_user_data_path, TRAJECTORY_DATA_DIR

os.environ["DYNACONF_ADVANCED_FEATURES__TRACKER_ENABLED"] = "true"

app = typer.Typer(
    help="Cuga CLI for managing services with direct execution",
    short_help="Service management tool for Cuga components",
)

# Global variables to track running direct processes (registry/demo)
direct_processes = {}
shutdown_event = threading.Event()

# OS detection
IS_WINDOWS = platform.system().lower().startswith("win")

# Playwright launcher state (for extension mode)
_playwright_thread: Optional[threading.Thread] = None
_playwright_started: bool = False


def kill_processes_by_port(ports: List[int]):
    """Kill processes listening on specified ports."""
    for port in ports:
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    # Get connections separately to handle cases where it's not available
                    try:
                        connections = proc.net_connections()
                    except (psutil.AccessDenied, AttributeError):
                        connections = []

                    for conn in connections:
                        if hasattr(conn, 'laddr') and conn.laddr and conn.laddr.port == port:
                            logger.info(
                                f"Force killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}"
                            )
                            psutil.Process(proc.info['pid']).terminate()
                            time.sleep(0.5)
                            try:
                                psutil.Process(proc.info['pid']).kill()
                            except psutil.NoSuchProcess:
                                pass
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            logger.debug(f"Error killing processes on port {port}: {e}")


def wait_for_registry_server(port: int, max_retries: int = 30, retry_interval: float = 0.5):
    """
    Wait for the registry server to be ready by pinging its health endpoint.

    Args:
        port: The port number the registry server is running on
        max_retries: Maximum number of retry attempts (default: 30)
        retry_interval: Time in seconds between retries (default: 0.5)

    Raises:
        TimeoutError: If the server doesn't become ready within max_retries attempts
    """
    url = f"http://127.0.0.1:{port}/"

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=1.0) as client:
                response = client.get(url)
                if response.status_code == 200:
                    logger.info("Registry server is ready!")
                    return
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
            else:
                raise TimeoutError(
                    f"Registry server did not become ready after {max_retries * retry_interval:.1f} seconds. "
                    f"Please check if the server started correctly on port {port}."
                )


def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Terminate children first
        for child in children:
            try:
                logger.debug(f"Terminating child process {child.pid}")
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Wait a bit for graceful termination
        psutil.wait_procs(children, timeout=3)

        # Kill any remaining children
        for child in children:
            try:
                if child.is_running():
                    logger.debug(f"Force killing child process {child.pid}")
                    child.kill()
            except psutil.NoSuchProcess:
                pass

        # Now terminate the parent
        try:
            logger.debug(f"Terminating parent process {pid}")
            parent.terminate()
            parent.wait(timeout=3)
        except psutil.TimeoutExpired:
            logger.debug(f"Force killing parent process {pid}")
            parent.kill()
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.debug(f"Error killing process tree {pid}: {e}")


def start_extension_browser_if_configured():
    """Start a Chromium instance with the MV3 extension if config enables it.

    Uses Playwright persistent context to load the extension from
    `frontend_workspaces/extension/releases/chrome-mv3`.
    Runs in a daemon thread and stops when the CLI receives a shutdown signal.
    """
    global _playwright_thread, _playwright_started

    use_extension = getattr(getattr(settings, "advanced_features", {}), "use_extension", False)
    if not use_extension:
        return

    if _playwright_started and _playwright_thread and _playwright_thread.is_alive():
        logger.info("Extension browser already running.")
        return

    extension_dir = os.path.join(
        PACKAGE_ROOT, "..", "frontend_workspaces", "extension", "releases", "chrome-mv3"
    )
    if not os.path.isdir(extension_dir):
        logger.error(
            f"Chrome MV3 extension directory not found: {extension_dir}. "
            "Build the extension or adjust your installation."
        )
        return

    def _runner():
        try:
            # Import here to avoid hard dependency if feature is off
            from playwright.sync_api import sync_playwright

            user_data_dir = get_user_data_path() or os.path.join(os.getcwd(), "logging", "pw_user_data")
            os.makedirs(user_data_dir, exist_ok=True)

            logger.info("Launching Chromium with extension (Playwright persistent context)...")
            with sync_playwright() as p:
                ctx = p.chromium.launch_persistent_context(
                    user_data_dir,
                    headless=False,
                    args=[
                        f"--disable-extensions-except={extension_dir}",
                        f"--load-extension={extension_dir}",
                    ],
                    no_viewport=True,
                )
                # Open a page to the demo start URL (if available), otherwise about:blank
                try:
                    start_url = getattr(getattr(settings, "demo_mode", {}), "start_url", None)
                except Exception:
                    start_url = None
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                if start_url:
                    page.goto(start_url, timeout=20000)
                else:
                    page.goto("about:blank", timeout=20000)

                # Keep context alive until shutdown
                while not shutdown_event.is_set():
                    time.sleep(0.2)

                try:
                    ctx.close()
                except Exception:
                    pass
        except ImportError:
            logger.error(
                "Playwright is not installed. Install with 'pip install playwright' "
                "and run 'playwright install chromium'."
            )
        except Exception as e:
            logger.error(f"Failed to launch Playwright with extension: {e}")

    _playwright_thread = threading.Thread(target=_runner, name="playwright-extension", daemon=True)
    _playwright_thread.start()
    _playwright_started = True


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) to gracefully shutdown direct processes."""
    logger.info("Received interrupt signal. Forcefully shutting down all processes...")
    shutdown_event.set()

    # Force stop direct processes
    stop_direct_processes()

    # Kill processes by common ports used by the services
    kill_processes_by_port([settings.server_ports.registry, settings.server_ports.demo])

    logger.info("All processes stopped.")
    sys.exit(0)


def stop_direct_processes():
    """Stop all direct processes gracefully, then forcefully."""
    for service_name, process in direct_processes.items():
        if process and process.poll() is None:
            logger.info(f"Stopping {service_name}...")
            try:
                # First try to kill the entire process tree
                kill_process_tree(process.pid)
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
                # Fallback to original method
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing {service_name}...")
                        process.kill()
                        process.wait()
                except Exception as e2:
                    logger.error(f"Error in fallback kill for {service_name}: {e2}")

    direct_processes.clear()


def run_direct_service(service_name: str, command: List[str], cwd: Optional[str] = None):
    """Run a service command directly and return the process."""
    try:
        logger.info(f"Starting {service_name} directly with command: {' '.join(command)}")

        # Force colored output and ensure proper environment variables
        env = os.environ.copy()
        env['FORCE_COLOR'] = '1'

        # Ensure APPWORLD_ROOT is set correctly for appworld commands
        if 'appworld' in ' '.join(command).lower():
            cwd = env.get('APPWORLD_ROOT')
        # Log environment variables for debugging
        logger.debug(f"APPWORLD_ROOT: {env.get('APPWORLD_ROOT')}")
        logger.debug(f"Working directory: {cwd or os.getcwd()}")

        # Start the process with a new process group to make it easier to kill
        kwargs = {'cwd': cwd, 'env': env, 'preexec_fn': os.setsid if not IS_WINDOWS else None}

        process = subprocess.Popen(command, **kwargs)

        direct_processes[service_name] = process
        return process

    except Exception as e:
        logger.error(f"Error starting {service_name}: {e}")
        return None


def wait_for_direct_processes():
    """Wait for all direct processes to complete or be interrupted."""
    try:
        while direct_processes and not shutdown_event.is_set():
            # Check if any process has terminated
            terminated = []
            for service_name, process in direct_processes.items():
                if process.poll() is not None:
                    terminated.append(service_name)
                    logger.info(f"{service_name} has terminated")

            # Remove terminated processes
            for service_name in terminated:
                del direct_processes[service_name]

            if not direct_processes:
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stop_direct_processes()


@app.callback()
def callback(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output with detailed logging information"
    ),
):
    """
    Cuga CLI: A management tool for Cuga services with direct execution.

    This tool helps you control various components of the Cuga ecosystem:

    - demo: Both registry and demo agent (runs directly)
    - registry: The MCP registry service only (runs directly)
    - appworld: AppWorld environment and API servers (runs directly)

    Examples:
      cuga start demo           # Start both registry and demo agent directly
      cuga start registry       # Start registry only
      cuga start appworld       # Start AppWorld servers
    """
    if verbose:
        logger.level("DEBUG")

    # Set up signal handler for graceful shutdown of direct processes
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Helper function to validate service
def validate_service(service: str):
    """Validate service name."""
    valid_services = ["demo", "registry", "appworld"]

    if service not in valid_services:
        logger.error(f"Unknown service: {service}. Valid options are: {', '.join(valid_services)}")
        raise typer.Exit(1)


@app.command(help="Start a specified service", short_help="Start service(s)")
def start(
    service: str = typer.Argument(
        ...,
        help="Service to start: demo (registry + demo agent), registry (registry only), or appworld (environment + api servers)",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind to (default: 127.0.0.1). Use 0.0.0.0 to allow external connections.",
    ),
    sandbox: bool = typer.Option(
        False,
        "--sandbox",
        help="Enable remote sandbox mode with llm-sandbox (requires --group sandbox to be installed)",
    ),
):
    """
    Start the specified service.

    Available services:
      - demo: Starts both registry and demo agent directly (registry on port 8001, demo on port 8005)
      - registry: Starts only the registry service directly (uvicorn on port 8001)
      - appworld: Starts AppWorld environment and API servers (environment on port 8000, api on port 9000)

    Examples:
      cuga start demo                # Start with local sandbox (default)
      cuga start demo --sandbox      # Start with remote sandbox (Docker/Podman)
      cuga start registry            # Start registry only
      cuga start appworld            # Start AppWorld servers
    """
    validate_service(service)

    # Handle direct execution services (demo and registry)
    if service == "demo":
        try:
            # Set environment variable for host
            os.environ["CUGA_HOST"] = host

            # If sandbox mode is enabled, update settings dynamically
            if sandbox:
                logger.info("Starting demo with remote sandbox mode enabled (features.local_sandbox=false)")
                os.environ["DYNACONF_FEATURES__LOCAL_SANDBOX"] = "false"
            else:
                # No override - let default configuration be used
                pass

            # Start registry first - using explicit uvicorn command
            registry_process = run_direct_service(
                "registry",
                [
                    "uvicorn",
                    "cuga.backend.tools_env.registry.registry.api_registry_server:app",
                    "--host",
                    host,
                    "--port",
                    str(settings.server_ports.registry),
                ],
            )

            # Check if registry failed to start
            if registry_process is None or registry_process.poll() is not None:
                logger.error("Registry service failed to start. Exiting.")
                stop_direct_processes()
                raise typer.Exit(1)

            # Wait for registry to start
            logger.info("Waiting for registry to start...")
            time.sleep(7)

            # Double-check registry is still running after wait
            if registry_process.poll() is not None:
                logger.error("Registry service terminated during startup. Exiting.")
                stop_direct_processes()
                raise typer.Exit(1)

            # Then start demo - using explicit command with optional sandbox group
            demo_command = []
            if sandbox:
                demo_command = [
                    "uv",
                    "run",
                    "--group",
                    "sandbox",
                    "fastapi",
                    "dev",
                    os.path.join(PACKAGE_ROOT, "backend", "server", "main.py"),
                    "--host",
                    host,
                    "--no-reload",
                    "--port",
                    str(settings.server_ports.demo),
                ]
            else:
                demo_command = [
                    "fastapi",
                    "dev",
                    os.path.join(PACKAGE_ROOT, "backend", "server", "main.py"),
                    "--host",
                    host,
                    "--no-reload",
                    "--port",
                    str(settings.server_ports.demo),
                ]

            run_direct_service("demo", demo_command)

            # Optionally start Chromium with MV3 extension if configured

            if direct_processes:
                logger.info(
                    "\n\033[1;36m┌──────────────────────────────────────────────────┐\n"
                    "\033[1;36m│\033[0m \033[1;33mDemo services are running. Press Ctrl+C to stop\033[0m \033[1;36m │\033[0m\n"
                    f"\033[1;36m│\033[0m \033[1;37mRegistry: http://localhost:{settings.server_ports.registry}                 \033[0m \033[1;36m│\033[0m\n"
                    f"\033[1;36m│\033[0m \033[1;37mDemo: http://localhost:{settings.server_ports.demo}                     \033[0m \033[1;36m│\033[0m\n"
                    "\033[1;36m└──────────────────────────────────────────────────┘\033[0m"
                )
                wait_for_direct_processes()

        except Exception as e:
            logger.error(f"Error starting demo services: {e}")
            stop_direct_processes()
            raise typer.Exit(1)
        return

    elif service == "registry":
        try:
            run_direct_service(
                "registry",
                [
                    "uvicorn",
                    "cuga.backend.tools_env.registry.registry.api_registry_server:app",
                    "--host",
                    host,
                    "--port",
                    str(settings.server_ports.registry),
                ],
            )

            if direct_processes:
                logger.info(
                    f"\n\033[1;36m┌────────────────────────────────────────┐\n\033[1;36m│\033[0m \033[1;33mRegistry service is running. Press Ctrl+C to stop\033[0m \033[1;36m│\033[0m\n\033[1;36m│\033[0m \033[1;37mRegistry: http://localhost:{settings.server_ports.registry}\033[0m         \033[1;36m│\033[0m\n\033[1;36m└────────────────────────────────────────┘\033[0m"
                )
                wait_for_direct_processes()
        except Exception as e:
            logger.error(f"Error starting registry service: {e}")
            stop_direct_processes()
            raise typer.Exit(1)
        return

    elif service == "appworld":
        try:
            # Start environment server first
            run_direct_service(
                "appworld-environment",
                ["appworld", "serve", "environment", "--port", str(settings.server_ports.environment_url)],
            )

            # Wait for environment server to start
            logger.info("Waiting for AppWorld environment server to start...")
            time.sleep(5)

            # Then start API server
            run_direct_service(
                "appworld-api", ["appworld", "serve", "apis", "--port", str(settings.server_ports.apis_url)]
            )

            if direct_processes:
                logger.info(
                    "\n\033[1;36m┌──────────────────────────────────────────────────┐\n"
                    "\033[1;36m│\033[0m \033[1;33mAppWorld services are running. Press Ctrl+C to stop\033[0m \033[1;36m │\033[0m\n"
                    f"\033[1;36m│\033[0m \033[1;37mEnvironment: http://localhost:{settings.server_ports.environment_url}              \033[0m \033[1;36m│\033[0m\n"
                    f"\033[1;36m│\033[0m \033[1;37mAPI: http://localhost:{settings.server_ports.apis_url}                      \033[0m \033[1;36m│\033[0m\n"
                    "\033[1;36m└──────────────────────────────────────────────────┘\033[0m"
                )
                wait_for_direct_processes()

        except Exception as e:
            logger.error(f"Error starting AppWorld services: {e}")
            stop_direct_processes()
            raise typer.Exit(1)
        return


def manage_service(action: str, service: str):
    """Common function for stopping or restarting services."""
    validate_service(service)

    if action == "stop":
        if service == "demo":
            # Stop both registry and demo for demo service
            stopped_any = False
            for service_name in ["registry", "demo"]:
                if service_name in direct_processes:
                    process = direct_processes[service_name]
                    if process and process.poll() is None:
                        logger.info(f"Stopping {service_name}...")
                        kill_process_tree(process.pid)
                        stopped_any = True
                    del direct_processes[service_name]
            if not stopped_any:
                logger.info("Demo services are not running")
        elif service == "registry":
            # Stop only registry for registry service
            if "registry" in direct_processes:
                process = direct_processes["registry"]
                if process and process.poll() is None:
                    logger.info("Stopping registry...")
                    kill_process_tree(process.pid)
                del direct_processes["registry"]
            else:
                logger.info("Registry service is not running")
        elif service == "appworld":
            # Stop both appworld services
            stopped_any = False
            for service_name in ["appworld-environment", "appworld-api"]:
                if service_name in direct_processes:
                    process = direct_processes[service_name]
                    if process and process.poll() is None:
                        logger.info(f"Stopping {service_name}...")
                        kill_process_tree(process.pid)
                        stopped_any = True
                    del direct_processes[service_name]
            if not stopped_any:
                logger.info("AppWorld services are not running")
    elif action == "restart":
        # Stop if running, then start
        manage_service("stop", service)
        time.sleep(1)
        # Call start command
        start(service)


@app.command(help="Stop a specified service", short_help="Stop service(s)")
def stop(
    service: str = typer.Argument(
        ...,
        help="Service to stop: demo (registry + demo agent), registry (registry only), or appworld (environment + api servers)",
    ),
):
    """
    Stop the specified service.

    Available services:
      - demo: Stops both registry and demo agent (direct processes)
      - registry: Stops only the registry service (direct process)
      - appworld: Stops both AppWorld environment and API servers (direct processes)

    Examples:
      cuga stop demo       # Stop both registry and demo services
      cuga stop registry   # Stop only the registry service
      cuga stop appworld   # Stop AppWorld servers
    """
    manage_service("stop", service)


@app.command(help="Start trajectory viewer", short_help="Start trajectory viewer")
def viz():
    """
    Start the trajectory viewer.

    This command launches a web-based dashboard for viewing and analyzing trajectory data from agent executions.

    Example:
      cuga viz         # Start the trajectory viewer
    """
    try:
        trajectory_data_path = TRAJECTORY_DATA_DIR
        subprocess.run(
            ["uv", "run", "--group", "dev", "cuga-viz", "run", trajectory_data_path],
            capture_output=False,
            text=False,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting dashboard: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False


@app.command(help="Show status of services", short_help="Display service status")
def status(
    service: str = typer.Argument(
        "all",
        help="Service to check status: demo (registry + demo agent), registry (registry only), appworld (environment + api servers), or all (all services)",
    ),
):
    """
    Display the current status of services.

    Available services:
      - demo: Shows status of both registry and demo agent (direct processes)
      - registry: Shows status of registry service only (direct process)
      - appworld: Shows status of both AppWorld environment and API servers (direct processes)
      - all: Shows status of all services (default)

    Examples:
      cuga status              # Show status of all services
      cuga status demo         # Show status of demo services (registry + demo)
      cuga status registry     # Show status of registry only
      cuga status appworld     # Show status of AppWorld servers
    """
    if service == "demo":
        # Show status of both registry and demo for demo service
        for service_name in ["registry", "demo"]:
            if service_name in direct_processes:
                process = direct_processes[service_name]
                if process.poll() is None:
                    logger.info(f"{service_name.capitalize()} service: Running (PID: {process.pid})")
                else:
                    logger.info(f"{service_name.capitalize()} service: Terminated")
            else:
                logger.info(f"{service_name.capitalize()} service: Not running")
        return

    elif service == "registry":
        if "registry" in direct_processes:
            process = direct_processes["registry"]
            if process.poll() is None:
                logger.info(f"Registry service: Running (PID: {process.pid})")
            else:
                logger.info("Registry service: Terminated")
        else:
            logger.info("Registry service: Not running")
        return

    elif service == "appworld":
        # Show status of both appworld services
        for service_name in ["appworld-environment", "appworld-api"]:
            if service_name in direct_processes:
                process = direct_processes[service_name]
                if process.poll() is None:
                    logger.info(
                        f"{service_name.replace('appworld-', '').capitalize()} service: Running (PID: {process.pid})"
                    )
                else:
                    logger.info(f"{service_name.replace('appworld-', '').capitalize()} service: Terminated")
            else:
                logger.info(f"{service_name.replace('appworld-', '').capitalize()} service: Not running")
        return

    elif service == "all":
        # Show direct processes status
        logger.info("Services:")
        for service_name in ["demo", "registry", "appworld-environment", "appworld-api"]:
            if service_name in direct_processes:
                process = direct_processes[service_name]
                if process.poll() is None:
                    display_name = (
                        service_name.replace('appworld-', 'appworld-')
                        if 'appworld-' in service_name
                        else service_name
                    )
                    logger.info(f"  {display_name}: Running (PID: {process.pid})")
                else:
                    display_name = (
                        service_name.replace('appworld-', 'appworld-')
                        if 'appworld-' in service_name
                        else service_name
                    )
                    logger.info(f"  {display_name}: Terminated")
            else:
                display_name = (
                    service_name.replace('appworld-', 'appworld-')
                    if 'appworld-' in service_name
                    else service_name
                )
                logger.info(f"  {display_name}: Not running")
        return

    # Validate service for any other service
    validate_service(service)


@app.command(help="Test sandbox execution", short_help="Test sandbox")
def test_sandbox(
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Test with remote sandbox (Docker/Podman) instead of local execution",
    ),
):
    """
    Test sandbox execution to verify code execution works correctly.

    Examples:
      cuga test-sandbox           # Test local sandbox (default)
      cuga test-sandbox --remote  # Test remote sandbox with Docker/Podman
    """
    try:
        from scripts.commands import test_sandbox as run_test

        if remote:
            # Ensure sandbox dependencies are available
            logger.info("Testing remote sandbox mode (requires --group sandbox)")
            run_test(remote=True)
        else:
            logger.info("Testing local sandbox mode")
            run_test(remote=False)

        logger.info("✅ Sandbox test completed successfully")
    except Exception as e:
        logger.error(f"❌ Sandbox test failed: {e}")
        raise typer.Exit(1)


@app.command(help="Evaluate Cuga on your test cases", short_help="Run Cuga Evaluation")
def evaluate(
    test_cases_file_path: str = typer.Argument(
        "",
        help="Path to your test cases file",
    ),
    output_file_path: str = typer.Argument(
        default="results.json",
        help="Path to your output file, it defaults to 'results.json'",
    ),
):
    """
    Run Cuga on your test cases.
    """
    # start the registry
    try:
        run_direct_service(
            "registry",
            [
                "uvicorn",
                "cuga.backend.tools_env.registry.registry.api_registry_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(settings.server_ports.registry),
            ],
        )

        if direct_processes:
            logger.info(
                f"\n\033[1;36m┌────────────────────────────────────────┐\n\033[1;36m│\033[0m \033[1;33mRegistry service is running. Press Ctrl+C to stop\033[0m \033[1;36m│\033[0m\n\033[1;36m│\033[0m \033[1;37mRegistry: http://localhost:{settings.server_ports.registry}\033[0m         \033[1;36m│\033[0m\n\033[1;36m└────────────────────────────────────────┘\033[0m"
            )
            # Wait for registry to start
            logger.info("Waiting for registry to start...")
            wait_for_registry_server(settings.server_ports.registry)

            # Then start demo - using explicit fastapi command
            run_direct_service(
                "evaluation",
                [
                    "uv",
                    "run",
                    "--group",
                    "dev",
                    os.path.join(PACKAGE_ROOT, "evaluation/evaluate_cuga.py"),
                    "-t",
                    test_cases_file_path,
                    "-r",
                    output_file_path,
                ],
            )
        wait_for_direct_processes()

    except Exception as e:
        logger.error(f"Error starting registry service: {e}")
        stop_direct_processes()
        raise typer.Exit(1)
    return


if __name__ == "__main__":
    app()
