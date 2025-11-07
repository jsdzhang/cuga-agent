import asyncio
import datetime
import platform
import re
import shutil
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Union, Optional
from cuga.backend.utils.id_utils import random_id_with_timestamp, mask_with_timestamp
import traceback
from pydantic import BaseModel, ValidationError

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage
from loguru import logger

from cuga.backend.activity_tracker.tracker import ActivityTracker
from cuga.cli import start_extension_browser_if_configured
from cuga.backend.browser_env.browser.extension_env_async import ExtensionEnv
from cuga.backend.browser_env.browser.gym_obs.http_stream_comm import (
    ChromeExtensionCommunicatorHTTP,
    ChromeExtensionCommunicatorProtocol,
)
from cuga.backend.cuga_graph.nodes.browser.action_agent.tools.tools import format_tools
from cuga.backend.cuga_graph.graph import DynamicAgentGraph
from cuga.backend.cuga_graph.utils.controller import AgentRunner
from cuga.backend.cuga_graph.utils.event_porcessors.action_agent_event_processor import (
    ActionAgentEventProcessor,
)
from cuga.backend.cuga_graph.nodes.human_in_the_loop.followup_model import ActionResponse
from cuga.backend.cuga_graph.state.agent_state import AgentState, default_state
from cuga.backend.browser_env.browser.gym_env_async import BrowserEnvGymAsync
from cuga.backend.browser_env.browser.open_ended_async import OpenEndedTaskAsync
from cuga.backend.cuga_graph.utils.agent_loop import AgentLoop, AgentLoopAnswer, StreamEvent, OutputFormat
from cuga.config import (
    get_app_name_from_url,
    get_user_data_path,
    settings,
    PACKAGE_ROOT,
    LOGGING_DIR,
    TRACES_DIR,
)
from langfuse.langchain import CallbackHandler
from fastapi.responses import StreamingResponse, JSONResponse
import json

# Import embedded assets with feature flag
USE_EMBEDDED_ASSETS = os.getenv("USE_EMBEDDED_ASSETS", "false").lower() in ("true", "1", "yes", "on")

if USE_EMBEDDED_ASSETS:
    try:
        from .embedded_assets import embedded_assets

        print("✅ Using embedded assets (enabled via USE_EMBEDDED_ASSETS)")
    except ImportError:
        USE_EMBEDDED_ASSETS = False
        print("❌ Embedded assets enabled but not found, falling back to file system")
else:
    print("📁 Using file system assets (embedded assets disabled)")

try:
    from agent_analytics.instrumentation.configs import OTLPCollectorConfig
    from agent_analytics.instrumentation import agent_analytics_sdk
except ImportError as e:
    logger.warning(f"Failed to import agent_analytics: {e}")
    OTLPCollectorConfig = None
    agent_analytics_sdk = None

# Moved to top of file

# Path constants

TRACE_LOG_PATH = os.path.join(TRACES_DIR, "trace.log")
FRONTEND_DIST_DIR = os.path.join(PACKAGE_ROOT, "..", "frontend_workspaces", "frontend", "dist")
EXTENSION_DIR = os.path.join(PACKAGE_ROOT, "..", "frontend_workspaces", "extension", "releases", "chrome-mv3")
STATIC_DIR_FLOWS_PATH = os.path.join(PACKAGE_ROOT, "backend", "server", "flows")
SAVE_REUSE_PY_PATH = os.path.join(
    PACKAGE_ROOT, "backend", "tools_env", "registry", "mcp_servers", "saved_flows.py"
)

# Create logging directory
if settings.advanced_features.tracker_enabled:
    os.makedirs(LOGGING_DIR, exist_ok=True)


class AppState:
    """A class to hold and manage all application state variables."""

    def __init__(self):
        # Initializing all state variables to None or default values.
        self.tracker: Optional[ActivityTracker] = None
        self.obs: Optional[Any] = None
        self.info: Optional[Dict[str, Any]] = None
        self.env: Optional[BrowserEnvGymAsync | ExtensionEnv] = None
        self.state: Optional[AgentState] = None
        self.agent: Optional[DynamicAgentGraph] = (
            None  # Replace Any with your Agent's class type if available
        )
        self.thread_id: Optional[str] = None
        self.stop_agent: bool = False
        self.output_format: OutputFormat = (
            OutputFormat.WXO if settings.advanced_features.wxo_integration else OutputFormat.DEFAULT
        )
        self.package_dir: str = PACKAGE_ROOT

        # Set up static directories - use embedded assets if available
        if USE_EMBEDDED_ASSETS:
            try:
                frontend_path, extension_path = embedded_assets.extract_assets()
                self.STATIC_DIR_HTML: Optional[str] = str(frontend_path)
                self.EXTENSION_PATH: Optional[str] = str(extension_path)
                print(f"✅ Using embedded frontend: {self.STATIC_DIR_HTML}")
                print(f"✅ Using embedded extension: {self.EXTENSION_PATH}")
            except Exception as e:
                print(f"❌ Failed to extract embedded assets: {e}")
                self.static_dirs: List[str] = [FRONTEND_DIST_DIR]
                self.STATIC_DIR_HTML: Optional[str] = next(
                    (d for d in self.static_dirs if os.path.exists(d)), None
                )
                self.EXTENSION_PATH: Optional[str] = EXTENSION_DIR
        else:
            self.static_dirs: List[str] = [FRONTEND_DIST_DIR]
            self.STATIC_DIR_HTML: Optional[str] = next(
                (d for d in self.static_dirs if os.path.exists(d)), None
            )
            self.EXTENSION_PATH: Optional[str] = EXTENSION_DIR
        self.STATIC_DIR_FLOWS: str = STATIC_DIR_FLOWS_PATH
        self.save_reuse_process: Optional[asyncio.subprocess.Process] = None
        self.initialize_sdk()

    def initialize_sdk(self):
        """Initializes the analytics SDK and logging."""
        logs_dir_path = TRACES_DIR

        if agent_analytics_sdk is not None and OTLPCollectorConfig is not None:
            os.makedirs(logs_dir_path, exist_ok=True)
            agent_analytics_sdk.initialize_logging(
                tracer_type=agent_analytics_sdk.SUPPORTED_TRACER_TYPES.LOG,
                logs_dir_path=logs_dir_path,
                log_filename="trace",
                config=OTLPCollectorConfig(
                    endpoint="",
                    app_name='cuga',
                ),
            )


# Create a single instance of the AppState class to be used throughout the application.
app_state = AppState()


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    stream: bool = False


def format_time_custom():
    """Formats the current time as HH-MM-SS."""
    now = datetime.datetime.now()
    return f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"


async def manage_save_reuse_server():
    """Checks for, starts, or restarts the save_reuse server as a subprocess."""
    if not settings.features.save_reuse:
        return

    # Define the path to the save_reuse.py file
    save_reuse_py_path = SAVE_REUSE_PY_PATH

    if not os.path.exists(save_reuse_py_path):
        logger.warning(f"save_reuse.py not found at {save_reuse_py_path}. Server will not be started.")
        return

    # If the process exists and is running, terminate it for a restart.
    if app_state.save_reuse_process and app_state.save_reuse_process.returncode is None:
        logger.info("Restarting save_reuse server...")
        app_state.save_reuse_process.terminate()
        await app_state.save_reuse_process.wait()

    logger.info("Starting save_reuse server...")
    # Assumes the file save_reuse.py contains a FastAPI instance named 'app'
    # and it is intended to be run with uvicorn.
    try:
        app_state.save_reuse_process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            SAVE_REUSE_PY_PATH,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.sleep(6)
        logger.info(f"save_reuse server started successfully with PID: {app_state.save_reuse_process.pid}")
    except FileNotFoundError:
        logger.error("Could not find 'uvicorn'. Please ensure it's installed in your environment.")
    except Exception as e:
        logger.error(f"Failed to start save_reuse server: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Asynchronous context manager for application startup and shutdown."""
    logger.info("Application is starting up...")

    # Start the save_reuse server if configured
    await manage_save_reuse_server()
    app_state.tracker = ActivityTracker()
    if settings.advanced_features.use_extension:
        app_state.env = ExtensionEnv(
            OpenEndedTaskAsync,
            ChromeExtensionCommunicatorHTTP(),
            feedback=[],
            user_data_dir=get_user_data_path(),
            task_kwargs={"start_url": settings.demo_mode.start_url},
        )
        start_extension_browser_if_configured()
    else:
        app_state.env = BrowserEnvGymAsync(
            OpenEndedTaskAsync,
            headless=False,
            resizeable_window=True,
            interface_mode="none" if settings.advanced_features.mode == "api" else "browser_only",
            feedback=[],
            user_data_dir=get_user_data_path(),
            channel="chromium",
            task_kwargs={"start_url": settings.demo_mode.start_url},
            pw_extra_args=[
                *settings.get("PLAYWRIGHT_ARGS", []),
                f"--disable-extensions-except={app_state.EXTENSION_PATH}",
                f"--load-extension={app_state.EXTENSION_PATH}",
            ],
        )
    await asyncio.sleep(3)
    app_state.tracker.start_experiment(task_ids=['demo'], experiment_name='demo', description="")
    app_state.obs, app_state.info = await app_state.env.reset()
    app_state.stop_agent = False  # Reset stop flag on startup
    app_state.state = default_state(page=None, observation=None, goal="")
    app_state.agent = DynamicAgentGraph(None)
    await app_state.agent.build_graph()
    app_state.thread_id = str(uuid.uuid4())

    logger.info("Application finished starting up...")
    url = f"http://localhost:{settings.server_ports.demo}?t={random_id_with_timestamp()}"
    if settings.advanced_features.mode == "api" and os.getenv("CUGA_TEST_ENV", "false").lower() not in (
        "true",
        "1",
        "yes",
        "on",
    ):
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', url], check=False)
            elif platform.system() == 'Windows':  # Windows
                subprocess.run(['cmd', '/c', 'start', '', url], check=False, shell=False)
            else:  # Linux
                subprocess.run(['xdg-open', url], check=False)
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")
    yield
    logger.info("Application is shutting down...")

    # Terminate the save_reuse server process if it's running
    if app_state.save_reuse_process and app_state.save_reuse_process.returncode is None:
        logger.info("Terminating save_reuse server...")
        app_state.save_reuse_process.terminate()
        await app_state.save_reuse_process.wait()
        logger.info("save_reuse server terminated.")

    # Clean up embedded assets
    if USE_EMBEDDED_ASSETS:
        embedded_assets.cleanup()
        logger.info("Cleaned up embedded assets")


def get_element_names(tool_calls, elements):
    """Extracts element names from tool calls."""
    elements_map = {}
    for tool in tool_calls:
        element_bid = tool.get("args", {}).get("bid", None)
        if element_bid:
            elements_map[element_bid] = ActionAgentEventProcessor.get_element_name(elements, element_bid)
    return elements_map


async def copy_file_async(file_path, new_name):
    """Asynchronously copies a file to a new name in the same directory."""
    try:
        loop = asyncio.get_event_loop()
        if not await loop.run_in_executor(None, os.path.isfile, file_path):
            print(f"Error: File '{file_path}' does not exist.")
            return None
        directory = os.path.dirname(file_path)
        new_file_path = os.path.join(directory, new_name)
        await loop.run_in_executor(None, shutil.copy2, file_path, new_file_path)
        print(f"File successfully copied to '{new_file_path}'")
        return new_file_path
    except Exception as e:
        print(f"Error copying file: {e}")
        return None


async def setup_page_info(state: AgentState, env: ExtensionEnv | BrowserEnvGymAsync):
    """Setup page URL, app name, and description from environment."""
    # Get URL and title
    state.url = env.get_url()
    title = await env.get_title()
    url_app_name = get_app_name_from_url(state.url)
    # Sanitize title
    sanitized_title = re.sub(r'[^\w\s-]', '', title) if title else ""
    sanitized_title = re.sub(r'[-\s]+', '_', sanitized_title).strip('_').lower()
    # Create app name: url + sanitized title
    state.current_app = (
        f"{url_app_name}_{sanitized_title}" if sanitized_title else url_app_name or "unknown_app"
    )
    # Create description
    state.current_app_description = f"web application for '{title}' and url '{url_app_name}'"


async def event_stream(query: str, api_mode=False, resume=None):
    """Handles the main agent event stream."""
    app_state.stop_agent = False
    if not resume:
        app_state.state.input = query
        app_state.tracker.intent = query

    if not api_mode:
        app_state.obs, _, _, _, app_state.info = await app_state.env.step("")
        pu_answer = await app_state.env.pu_processor.transform(
            transformer_params={"filter_visible_only": True}
        )
        app_state.tracker.collect_image(pu_answer.img)
        app_state.state.elements_as_string = pu_answer.string_representation
        app_state.state.focused_element_bid = pu_answer.focused_element_bid
        app_state.state.read_page = pu_answer.page_content
        app_state.state.url = app_state.env.get_url()
        await setup_page_info(app_state.state, app_state.env)
    app_state.tracker.task_id = 'demo'

    langfuse_handler = CallbackHandler() if settings.advanced_features.langfuse_tracing else None

    # Print Langfuse trace ID if tracing is enabled
    if langfuse_handler and settings.advanced_features.langfuse_tracing:
        print(f"Langfuse tracing enabled. Handler created: {langfuse_handler}")
        # The trace ID will be available after the first LLM call
        print("Note: Trace ID will be available after the first LLM operation")

    agent_loop_obj = AgentLoop(
        graph=app_state.agent.graph, langfuse_handler=langfuse_handler, thread_id=app_state.thread_id
    )
    logger.debug(f"Resume: {resume.model_dump_json() if resume else ''}")
    agent_stream_gen = agent_loop_obj.run_stream(state=app_state.state if not resume else None, resume=resume)

    # Print initial trace ID status
    if langfuse_handler and settings.advanced_features.langfuse_tracing:
        initial_trace_id = agent_loop_obj.get_langfuse_trace_id()
        if initial_trace_id:
            print(f"Initial Langfuse Trace ID: {initial_trace_id}")
        else:
            print("Langfuse Trace ID will be generated after first LLM call")

    try:
        while True:
            if app_state.stop_agent:
                logger.info("Agent execution stopped by user")
                yield StreamEvent(name="Stopped", data="Agent execution was stopped by user.").format()
                return

            async for event in agent_stream_gen:
                # await asyncio.sleep(0.5)
                if app_state.stop_agent:
                    logger.info("Agent execution stopped by user during event processing")
                    yield StreamEvent(name="Stopped", data="Agent execution was stopped by user.").format()
                    return

                if isinstance(event, AgentLoopAnswer):
                    if event.flow_generalized:
                        await manage_save_reuse_server()
                        await app_state.agent.chat.chat_agent.cleanup()
                        await app_state.agent.chat.chat_agent.setup()

                    if event.interrupt and not event.has_tools:
                        app_state.state = AgentState(
                            **app_state.agent.graph.get_state(
                                {"configurable": {"thread_id": app_state.thread_id}}
                            ).values
                        )
                        return
                    if event.end:
                        app_state.tracker.finish_task(
                            intent=app_state.state.input,
                            site="",
                            task_id="demo",
                            eval="",
                            score=1.0,
                            agent_answer=event.answer,
                            exception=False,
                            num_steps=0,
                            agent_v="",
                        )
                        logger.debug("!!!!!!!Task is done!!!!!!!")

                        yield StreamEvent(
                            name="Answer",
                            data=event.answer
                            if settings.advanced_features.wxo_integration
                            else json.dumps({"data": event.answer})
                            if event.answer
                            else "Done.",
                        ).format(app_state.output_format, thread_id=app_state.thread_id)

                        app_state.state = AgentState(
                            **app_state.agent.graph.get_state(
                                {"configurable": {"thread_id": app_state.thread_id}}
                            ).values
                        )
                        try:
                            await copy_file_async(
                                TRACE_LOG_PATH,
                                f"trace_{mask_with_timestamp(full_date=True, id='')}.log",
                            )
                            await copy_file_async(TRACE_LOG_PATH, "trace_backup.log")
                            os.remove(TRACE_LOG_PATH)
                        except Exception as e:
                            logger.warning(e)
                        return
                    elif event.has_tools:
                        app_state.state = AgentState(
                            **app_state.agent.graph.get_state(
                                {"configurable": {"thread_id": app_state.thread_id}}
                            ).values
                        )
                        msg: AIMessage = app_state.state.messages[-1]
                        yield StreamEvent(name="tool_call", data=format_tools(msg.tool_calls)).format()

                        feedback = await AgentRunner.process_event_async(
                            app_state.state.messages[-1].tool_calls,
                            app_state.state.elements,
                            None if api_mode else app_state.env.page,
                            app_state.env.tool_implementation_provider,
                            session_id="demo",
                            page_data=app_state.obs,
                            communicator=getattr(app_state.env, "extension_communicator", None),
                        )
                        app_state.state.feedback += feedback

                        if not api_mode:
                            app_state.obs, _, _, _, app_state.info = await app_state.env.step("")
                            pu_answer = await app_state.env.pu_processor.transform(
                                transformer_params={"filter_visible_only": True}
                            )
                            app_state.tracker.collect_image(pu_answer.img)
                            app_state.state.elements_as_string = pu_answer.string_representation
                            app_state.state.focused_element_bid = pu_answer.focused_element_bid
                            app_state.state.read_page = pu_answer.page_content
                            app_state.state.url = app_state.env.get_url()

                        app_state.agent.graph.update_state(
                            {"configurable": {"thread_id": app_state.thread_id}}, app_state.state.model_dump()
                        )
                        agent_stream_gen = agent_loop_obj.run_stream(state=None)
                        break
                else:
                    logger.debug("Yield {}".format(event))
                    app_state.state = AgentState(
                        **app_state.agent.graph.get_state(
                            {"configurable": {"thread_id": app_state.thread_id}}
                        ).values
                    )
                    name = ((event.split("\n")[0]).split(":")[1]).strip()
                    logger.debug("Yield {}".format(event))
                    if name not in ["ChatAgent"]:
                        yield StreamEvent(name=name, data=event).format(
                            app_state.output_format, thread_id=app_state.thread_id
                        )
    except Exception as e:
        logger.exception(e)
        logger.error(traceback.format_exc())
        app_state.tracker.finish_task(
            intent=app_state.state.input,
            site="",
            task_id="demo",
            eval="",
            score=0.0,
            agent_answer="",
            exception=True,
            num_steps=0,
            agent_v="",
        )
        yield StreamEvent(name="Error", data=str(e)).format()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if getattr(settings.advanced_features, "use_extension", False):
    print(settings.advanced_features.use_extension)

    def get_communicator() -> ChromeExtensionCommunicatorProtocol:
        comm: ChromeExtensionCommunicatorProtocol | None = getattr(
            app_state.env, "extension_communicator", None
        )
        if not comm:
            raise Exception("Cannot use streaming outside of extension")

        return comm

    @app.get("/extension/command_stream")
    async def extension_command_stream():
        comm = get_communicator()

        async def event_gen():
            while True:
                cmd = await comm.get_next_command()
                yield f"data: {json.dumps(cmd)}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.post("/extension/command_result")
    async def extension_command_result(request: Request):
        comm = get_communicator()
        data = await request.json()
        req_id = data.get("request_id")
        comm.resolve_request(req_id, data)
        return JSONResponse({"status": "ok"})

    @app.post("/extension/agent_query")
    async def extension_agent_query(request: Request):
        body = await request.json()
        query = body.get("query", "")
        request_id = body.get("request_id", None)
        if not query:
            return JSONResponse({"type": "agent_error", "message": "Missing query"}, status_code=400)

        async def event_gen():
            # Initial processing message
            yield (
                json.dumps(
                    {
                        "type": "agent_response",
                        "content": f"Processing query: {query}\n\n",
                        "request_id": request_id,
                    }
                )
                + "\n"
            )
            try:
                async for chunk in event_stream(
                    query,
                    api_mode=settings.advanced_features.mode == "api",
                    resume=query if isinstance(query, ActionResponse) else None,
                ):
                    if chunk.strip():
                        # Remove 'data: ' prefix if present
                        if chunk.startswith("data: "):
                            chunk = chunk[6:]
                        try:
                            chunk_data = json.loads(chunk)
                            content = chunk_data.get("data", chunk)
                        except Exception:
                            content = chunk
                        yield (
                            json.dumps(
                                {"type": "agent_response", "content": content, "request_id": request_id}
                            )
                            + "\n"
                        )
                # Completion message
                yield json.dumps({"type": "agent_complete", "request_id": request_id}) + "\n"
            except Exception as e:
                yield json.dumps({"type": "agent_error", "message": str(e), "request_id": request_id}) + "\n"

        return StreamingResponse(event_gen(), media_type="application/jsonlines")


@app.post("/stream")
async def stream(request: Request):
    """Endpoint to start the agent stream."""
    query = await get_query(request)
    return StreamingResponse(
        event_stream(
            query if isinstance(query, str) else None,
            api_mode=settings.advanced_features.mode == "api",
            resume=query if isinstance(query, ActionResponse) else None,
        ),
        media_type="text/event-stream",
    )


@app.post("/stop")
async def stop():
    """Endpoint to stop the agent execution."""
    logger.info("Received stop request")
    app_state.stop_agent = True
    return {"status": "success", "message": "Stop request received"}


@app.post("/reset")
async def reset_agent_state():
    """Endpoint to reset the agent state to default values."""
    logger.info("Received reset request")
    try:
        # Reset agent state to default
        app_state.state = default_state(page=None, observation=None, goal="")
        app_state.stop_agent = False
        app_state.thread_id = str(uuid.uuid4())

        # Reset the agent graph
        if app_state.agent:
            app_state.agent = DynamicAgentGraph(None)
            await app_state.agent.build_graph()

        # Reset environment if available
        if app_state.env:
            app_state.obs, app_state.info = await app_state.env.reset()

        logger.info("Agent state reset successfully")
        return {"status": "success", "message": "Agent state reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset agent state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset agent state: {str(e)}")


async def get_query(request: Request) -> Union[str, ActionResponse]:
    """Parses the incoming request to extract the user query or action."""
    try:
        data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Request body was not valid JSON.")

    if isinstance(data, dict) and set(data.keys()) == {"query"} and isinstance(data["query"], str):
        query_text = data["query"]
        if not query_text.strip():
            raise HTTPException(status_code=422, detail="`query` may not be empty.")
        return query_text
    elif isinstance(data, dict) and "action_id" in data:
        try:
            return ActionResponse(**data)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"Invalid ActionResponse JSON: {e.errors()}")
    else:
        try:
            chat_obj = ChatRequest.model_validate(data)
            query_text = ""
            for mes in reversed(chat_obj.messages):
                if mes['role'] == 'user':
                    query_text = mes['content']
                    break
            if not query_text.strip():
                raise HTTPException(status_code=422, detail="No user message found or content is empty.")
            return query_text
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"Invalid ChatRequest JSON: {e.errors()}")


@app.get("/flows/{full_path:path}")
async def serve_flows(full_path: str, request: Request):
    """Serves files from the flows directory."""
    file_path = os.path.join(app_state.STATIC_DIR_FLOWS, full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Flow file not found.")


@app.get("/{full_path:path}")
async def serve_react(full_path: str, request: Request):
    """Serves the main React application and its static files."""
    if not app_state.STATIC_DIR_HTML:
        raise HTTPException(status_code=500, detail="Frontend build directory not found.")

    file_path = os.path.join(app_state.STATIC_DIR_HTML, full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)

    index_path = os.path.join(app_state.STATIC_DIR_HTML, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)

    raise HTTPException(status_code=404, detail="Frontend files not found. Did you run the build process?")
