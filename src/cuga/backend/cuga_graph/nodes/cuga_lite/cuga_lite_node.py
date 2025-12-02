"""
CugaLite Node - Fast execution node using CugaAgent
"""

import json
from typing import Literal, Dict, Any, List, Optional
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field

from cuga.backend.cuga_graph.nodes.cuga_lite import CugaAgent
from cuga.backend.cuga_graph.nodes.shared.base_node import BaseNode
from cuga.backend.cuga_graph.state.agent_state import AgentState, SubTaskHistory
from cuga.backend.activity_tracker.tracker import ActivityTracker
from cuga.backend.cuga_graph.nodes.api.api_planner_agent.prompts.load_prompt import ActionName
from cuga.backend.cuga_graph.state.api_planner_history import CoderAgentHistoricalOutput
from cuga.config import settings
from langchain_core.messages import AIMessage


try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except ImportError:
    try:
        from langfuse.callback.langchain import LangchainCallbackHandler as LangfuseCallbackHandler
    except ImportError:
        LangfuseCallbackHandler = None


from cuga.configurations.instructions_manager import get_all_instructions_formatted


tracker = ActivityTracker()


class CugaLiteOutput(BaseModel):
    """Output model for CugaLite execution (similar to CodeAgentOutput)."""

    code: str = ""
    execution_output: str = ""
    steps_summary: List[str] = Field(default_factory=list)
    summary: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""


class CugaLiteNode(BaseNode):
    """Node wrapper for CugaAgent - fast execution mode."""

    def __init__(self, langfuse_handler: Optional[Any] = None):
        super().__init__()
        self.name = "CugaLite"
        self.agent: CugaAgent = None
        self.initialized = False
        self.langfuse_handler = langfuse_handler

    async def initialize_agent(self, app_names=None, force_reinit=False):
        """Initialize the CugaAgent with optional app filtering."""
        if not self.initialized or force_reinit:
            if force_reinit and self.initialized:
                logger.info("Re-initializing CugaLite agent with new app configuration...")
                self.initialized = False  # Reset initialization flag
            else:
                logger.info("Initializing CugaLite agent...")

            # Create langfuse handler if not provided and tracing is enabled
            if self.langfuse_handler is None and settings.advanced_features.langfuse_tracing:
                if LangfuseCallbackHandler is not None:
                    self.langfuse_handler = LangfuseCallbackHandler()
                    logger.info("Langfuse tracing enabled for CugaLite")
                else:
                    logger.warning("Langfuse tracing enabled but langfuse package not available")

            self.agent = CugaAgent(
                app_names=app_names,
                langfuse_handler=self.langfuse_handler,
                instructions=get_all_instructions_formatted(),
            )
            await self.agent.initialize()
            self.initialized = True
            logger.info(f"CugaLite initialized with {len(self.agent.tools)} tools")

    async def node(self, state: AgentState) -> Command[Literal['FinalAnswerAgent']]:
        """Execute the CugaAgent for fast task execution.

        Args:
            state: Current agent state

        Returns:
            Command to proceed to FinalAnswerAgent with the result
        """
        logger.info(f"CugaLite executing task: {state.input}")

        # Add initialization message
        state.messages.append(
            AIMessage(
                content=json.dumps(
                    {
                        "status": "initializing",
                        "message": f"Initializing CugaLite with {len(state.api_intent_relevant_apps) if state.api_intent_relevant_apps else 'all'} apps",
                    }
                )
            )
        )

        # Extract app names if available
        app_names = None

        # Use sub_task as the input if available (preferred over state.input)
        task_input = state.sub_task if state.sub_task else state.input

        # Check state.sub_task_app first (single app from API planner)
        if state.sub_task_app:
            app_names = [state.sub_task_app]
            logger.info(f"Using app from state.sub_task_app: {app_names}")
            # Force re-initialization for new sub_task_app
            await self.initialize_agent(app_names=app_names, force_reinit=True)
        elif state.api_intent_relevant_apps:
            app_names = [app.name for app in state.api_intent_relevant_apps if app.type == 'api']
            logger.info(f"Using apps from state.api_intent_relevant_apps: {app_names}")
            # Initialize agent if not already done
            await self.initialize_agent(app_names=app_names)
        else:
            # Initialize agent if not already done
            await self.initialize_agent()

        # Add execution start message
        state.messages.append(
            AIMessage(
                content=json.dumps(
                    {
                        "status": "executing",
                        "message": f"Executing task with {len(self.agent.tools)} available tools",
                        "tools_count": len(self.agent.tools),
                    }
                )
            )
        )

        # Get current variables from state to pass as initial context
        initial_var_names = state.variables_manager.get_variable_names()
        initial_context = {name: state.variables_manager.get_variable(name) for name in initial_var_names}

        logger.info(f"Passing {len(initial_context)} variables to CugaAgent as initial context")
        logger.info(f"Variable names: {initial_var_names}")
        for var_name in initial_var_names:
            logger.info(
                f"  - {var_name}: {type(initial_context[var_name]).__name__} = {str(initial_context[var_name])[:100]}"
            )
        logger.info(f"Variables summary: {state.variables_manager.get_variables_summary()}")

        # Execute the task - messages will be added automatically by CugaAgent
        answer, metrics, updated_messages = await self.agent.execute(
            task_input,
            recursion_limit=15,
            show_progress=False,
            state_messages=state.messages,
            chat_messages=state.chat_messages if state.chat_messages else None,
            initial_context=initial_context,
        )

        # Check if execution failed (graph-level errors or code execution errors)
        has_error = metrics is not None and metrics.get('error') is not None

        # Also check if the answer itself indicates an error
        if not has_error and answer:
            error_indicators = ['Error during execution:', 'Error:', 'Exception:', 'Traceback', 'Failed to']
            has_error = any(indicator in answer for indicator in error_indicators)
            if has_error:
                logger.warning(f"Detected error in answer content: {answer[:200]}...")

        if has_error:
            error_msg = (
                metrics.get('error', 'Code execution error detected in output')
                if metrics
                else 'Code execution error detected in output'
            )
            logger.error(f"CugaLite execution failed with error: {error_msg}")
            logger.error(f"Full answer: {answer}")

            # Update state with error information
            if state.sub_task:
                # For sub-tasks, add error to history and return to plan controller
                if state.api_planner_history:
                    state.api_planner_history[-1].agent_output = CoderAgentHistoricalOutput(
                        variables_summary="Execution failed",
                        final_output=answer,  # This already contains the error message
                    )

                state.stm_all_history.append(
                    SubTaskHistory(
                        sub_task=state.format_subtask(),
                        steps=[],
                        final_answer=answer,  # Contains error message
                    )
                )
                state.last_planner_answer = answer
                state.sender = "CugaLiteNode"
                logger.info("CugaLite sub-task execution failed, returning error to PlanControllerAgent")
                return Command(update=state.model_dump(), goto="PlanControllerAgent")
            else:
                # For regular execution, set final answer with error
                state.final_answer = answer  # Contains error message
                state.sender = self.name
                logger.info("CugaLite execution failed, proceeding to FinalAnswerAgent with error")
                return Command(update=state.model_dump(), goto="FinalAnswerAgent")

        # Update chat_messages with the updated messages from execution
        if updated_messages:
            state.chat_messages = updated_messages

        # Extract updated context from messages and sync to var_manager
        updated_variables = {}
        if updated_messages:
            for msg in reversed(updated_messages):
                if (
                    isinstance(msg, AIMessage)
                    and hasattr(msg, 'additional_kwargs')
                    and 'context' in msg.additional_kwargs
                ):
                    updated_variables = msg.additional_kwargs['context']
                    logger.debug(
                        f"Extracted {len(updated_variables)} updated variables from CodeAct execution"
                    )
                    break

        # Sync new variables from CodeAct execution to var_manager
        initial_var_set = set(initial_var_names)
        new_var_names = []

        if updated_variables:
            # Identify newly created variables
            for var_name, var_value in updated_variables.items():
                if not var_name.startswith('_') and var_name not in initial_var_set:
                    new_var_names.append(var_name)

            # For sub-tasks, keep only the last variable if multiple were generated
            vars_to_add = new_var_names
            if state.sub_task and len(new_var_names) > 1:
                # Keep only the last variable name (last in the updated_variables dict order)
                last_var_name = new_var_names[-1]
                vars_to_add = [last_var_name]
                logger.info(
                    f"Sub-task execution: keeping only last variable '{last_var_name}' out of {len(new_var_names)} generated"
                )

            # Add the variables to state
            for var_name in vars_to_add:
                var_value = updated_variables[var_name]
                state.variables_manager.add_variable(
                    var_value, name=var_name, description="Generated by CugaLite"
                )
                logger.info(f"Added new variable '{var_name}' to variables_manager")

        logger.info(
            f"After execution, variables_manager now has {state.variables_manager.get_variable_count()} variables"
        )
        logger.info(f"Variable names after execution: {state.variables_manager.get_variable_names()}")

        logger.info(f"CugaLite completed in {metrics.get('duration_seconds', 0) if metrics else 0}s")
        logger.info(f"Used {metrics.get('total_tokens', 0) if metrics else 0} tokens")
        logger.info(
            f"Steps: {metrics.get('step_count', 0) if metrics else 0}, Tokens: {metrics.get('total_tokens', 0) if metrics else 0}"
        )

        # Log Langfuse trace ID if available
        if self.agent and self.langfuse_handler:
            trace_id = self.agent.get_langfuse_trace_id()
            if trace_id:
                logger.info(f"Langfuse Trace ID: {trace_id}")
                print(f"🔍 Langfuse Trace ID: {trace_id}")

        # Check if we're executing a sub-task
        if state.sub_task:
            # Sub-task execution - return to PlanControllerAgent
            state.api_last_step = ActionName.CONCLUDE_TASK
            state.guidance = None

            # Add to previous steps

            # Update api_planner_history with CoderAgentHistoricalOutput
            if state.api_planner_history:
                # Use the new variable names that were actually added to variables_manager
                state.api_planner_history[-1].agent_output = CoderAgentHistoricalOutput(
                    variables_summary=state.variables_manager.get_variables_summary(
                        new_var_names, max_length=5000
                    )
                    if new_var_names
                    else "No new variables",
                    final_output=answer,
                )

            state.stm_all_history.append(
                SubTaskHistory(
                    sub_task=state.format_subtask(),
                    steps=[],
                    final_answer=answer,
                )
            )
            state.last_planner_answer = answer
            state.sender = "CugaLiteNode"
            logger.info("CugaLite sub-task execution successful, proceeding to PlanControllerAgent")
            return Command(update=state.model_dump(), goto="PlanControllerAgent")
        else:
            # Regular execution - proceed to FinalAnswerAgent
            state.final_answer = answer
            state.sender = self.name
            logger.info("CugaLite execution successful, proceeding to FinalAnswerAgent")
            return Command(update=state.model_dump(), goto="FinalAnswerAgent")
