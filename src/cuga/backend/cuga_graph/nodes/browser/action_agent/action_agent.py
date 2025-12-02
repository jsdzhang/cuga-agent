from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from cuga.backend.cuga_graph.nodes.browser.action_agent.tools.tools import setup_tools
from cuga.backend.cuga_graph.nodes.shared.base_agent import BaseAgent
from cuga.backend.cuga_graph.state.agent_state import AgentState
from cuga.backend.llm.models import LLMManager
from cuga.backend.llm.utils.helpers import load_prompt_simple
from cuga.config import settings

llm_manager = LLMManager()


class ActionAgent(BaseAgent):
    def __init__(self, prompt_template: ChatPromptTemplate, llm: BaseChatModel, tools: Any = None):
        super().__init__()
        self.name = "ActionAgent"
        prompt = prompt_template.partial(tool_names=", ".join([tool.name for key, tool in tools.items()]))
        tools = tools.values()
        self.chain = prompt | llm.bind_tools(tools)

    @staticmethod
    def output_parser(result: BaseMessage, name) -> BaseMessage:
        result.name = name
        return result

    def run(self, input_variables: AgentState) -> AIMessage:
        data = input_variables.model_dump()
        if settings.advanced_features.mode == "hybrid":
            data["variables_history"] = input_variables.variables_manager.get_variables_summary(last_n=1)
        else:
            data["variables_history"] = ""

        return self.chain.invoke(data)

    @staticmethod
    def create():
        dyna_model = settings.agent.action.model
        return ActionAgent(
            prompt_template=load_prompt_simple(
                "./prompts/system.jinja2",
                "./prompts/user.jinja2",
                model_config=dyna_model,
            ),
            llm=llm_manager.get_model(dyna_model),
            tools=setup_tools(),
        )
