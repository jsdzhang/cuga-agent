#!/usr/bin/env python3
"""
End-to-End tests for Runtime Tools functionality
Tests StructuredTool loading from langchain, API utilities, and code execution
"""

import unittest
from unittest.mock import patch

from cuga.backend.activity_tracker.tracker import ActivityTracker
from cuga.backend.tools_env.registry.utils.api_utils import get_apis, get_apps
from cuga.backend.tools_env.code_sandbox.sandbox import run_code
from cuga.backend.cuga_graph.state.agent_state import AgentState
from cuga.backend.cuga_graph.utils.controller import AgentRunner as CugaAgent
from system_tests.e2e.calculator_tool import (
    tools as calculator_tools,
    evaluate_expression,
    get_pi,
    calculate_factorial,
)
from langchain_core.tools import StructuredTool


# Global tracker instance since ActivityTracker is a singleton
tracker = ActivityTracker()


class TestRuntimeTools(unittest.IsolatedAsyncioTestCase):
    """Test runtime tools functionality including StructuredTool loading, API utils, and code execution"""

    def setUp(self):
        """Set up test environment"""
        self.cuga_agent = None

    async def asyncSetUp(self):
        """Async setup for tests requiring CugaAgent"""
        pass

    def tearDown(self):
        """Clean up after tests"""
        # Clear any tools set in tracker
        if hasattr(tracker, '_tools'):
            tracker._tools.clear()

    def test_structured_tool_loading_from_langchain(self):
        """Test that StructuredTool.from_function works correctly with langchain tools"""
        # Test evaluate_expression tool
        evaluate_tool = StructuredTool.from_function(evaluate_expression)
        self.assertIsInstance(evaluate_tool, StructuredTool)
        self.assertEqual(evaluate_tool.name, "evaluate_expression")
        self.assertIn("expression", evaluate_tool.args_schema.model_fields)

        # Test get_pi tool
        pi_tool = StructuredTool.from_function(get_pi)
        self.assertIsInstance(pi_tool, StructuredTool)
        self.assertEqual(pi_tool.name, "get_pi")

        # Test factorial tool
        factorial_tool = StructuredTool.from_function(calculate_factorial)
        self.assertIsInstance(factorial_tool, StructuredTool)
        self.assertEqual(factorial_tool.name, "calculate_factorial")
        self.assertIn("n", factorial_tool.args_schema.model_fields)

        # Test that all calculator tools are properly created
        self.assertEqual(len(calculator_tools), 3)
        tool_names = [tool.name for tool in calculator_tools]
        self.assertIn("evaluate_expression", tool_names)
        self.assertIn("get_pi", tool_names)
        self.assertIn("calculate_factorial", tool_names)

    def test_calculator_tool_functionality(self):
        """Test the calculator tool functions work correctly"""
        # Test evaluate_expression
        result = evaluate_expression("2 + 3 * 4")
        self.assertTrue(result.success)
        self.assertEqual(result.result, 14.0)
        self.assertEqual(result.expression, "2 + 3 * 4")

        # Test with math functions
        result = evaluate_expression("sin(pi/2)")
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.result, 1.0, places=5)

        # Test error handling
        result = evaluate_expression("invalid syntax +++")
        self.assertFalse(result.success)
        self.assertIn("error_message", result.model_dump())

        # Test get_pi
        result = get_pi()
        self.assertAlmostEqual(result.pi_value, 3.141592653589793, places=10)

        # Test factorial
        result = calculate_factorial(5)
        self.assertTrue(result.success)
        self.assertEqual(result.result, 120)
        self.assertEqual(result.n, 5)

        # Test factorial error
        result = calculate_factorial(-1)
        self.assertFalse(result.success)
        self.assertIn("error_message", result.model_dump())

    async def test_get_apis_with_runtime_tools(self):
        """Test get_apis function with runtime tools setup"""
        # Set metadata similar to main.py example
        for tool in calculator_tools:
            tool.metadata = {'server_name': "calculator"}

        tracker.set_tools(calculator_tools)

        # Test get_apis when registry is disabled (using external tools)
        # with patch('cuga.config.settings.advanced_features.registry', False):
        result = await get_apis("calculator")
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        tool_names = list(result.keys())
        self.assertIn("evaluate_expression", tool_names)
        self.assertIn("get_pi", tool_names)
        self.assertIn("calculate_factorial", tool_names)

    async def test_get_apps_with_runtime_tools(self):
        """Test get_apps function with runtime tools setup"""
        # Set metadata similar to main.py example
        for tool in calculator_tools:
            tool.metadata = {'server_name': "calculator"}

        tracker.set_tools(calculator_tools)

        # Test get_apps when registry is disabled (using external apps)
        with patch('cuga.config.settings.advanced_features.registry', False):
            result = await get_apps()
            self.assertIsInstance(result, list)
            app_names = [app.name for app in result]
            # Should include apps from tools
            self.assertIn("calculator", app_names)

    async def test_run_code_with_api_calls(self):
        """Test run_code functionality with code that makes API calls"""
        for tool in calculator_tools:
            tool.metadata = {'server_name': "calculator"}

        tracker.set_tools(calculator_tools)
        # Create code that uses call_api to invoke calculator functions
        code = '''
# Test evaluate expression via API
result1 = await call_api("calculator", "evaluate_expression", {"expression": "10 + 5"})
print(f"10 + 5 = {result1['result']}")

# Test get pi via API
pi_result = await call_api("calculator", "get_pi", {})
print(f"Pi = {pi_result}")

# Test factorial via API
fact_result = await call_api("calculator", "calculate_factorial", {"n": 4})
print(f"4! = {fact_result['result']}")

# Test API-style calls with different expressions
api_result1 = await call_api("calculator", "evaluate_expression", {"expression": "2 * 3 + 4"})
print(f"API call result: {api_result1['result']}")

api_result2 = await call_api("calculator", "calculate_factorial", {"n": 3})
print(f"API factorial result: {api_result2['result']}")
'''

        # Run the code in sandbox
        state = AgentState(input="test", url="")
        output, locals_dict = await run_code(code, state)
        print(output)
        # Verify output contains expected results
        self.assertIn("10 + 5 = 15.0", output)
        self.assertIn("Pi =", output)
        self.assertIn("4! = 24", output)
        self.assertIn("API call result: 10.0", output)
        self.assertIn("API factorial result: 6", output)

    async def test_full_runtime_tools_workflow(self):
        """Test the full workflow similar to main.py example"""
        # Initialize CugaAgent similar to main.py
        cuga_agent = CugaAgent(browser_enabled=False)
        await cuga_agent.initialize_appworld_env()

        # Set up tools similar to main.py lines 27-33
        tools = calculator_tools
        for tool in tools:
            tool.metadata = {'server_name': "calculator"}
        tracker.set_tools(tools)

        # Verify tools are registered
        registered_tools = tracker.get_tools_by_server("calculator")
        self.assertEqual(len(registered_tools), 3)

        tool_names = list(registered_tools.keys())
        self.assertIn("evaluate_expression", tool_names)
        self.assertIn("get_pi", tool_names)
        self.assertIn("calculate_factorial", tool_names)

        # Verify apps are created
        self.assertGreater(len(tracker.apps), 0)
        app_names = [app.name for app in tracker.apps]
        self.assertIn("calculator", app_names)

    def test_langchain_tool_metadata(self):
        """Test that langchain tools have proper metadata and descriptions"""
        for tool in calculator_tools:
            # Check tool has name
            self.assertIsNotNone(tool.name)
            self.assertIsInstance(tool.name, str)

            # Check tool has description
            self.assertIsNotNone(tool.description)
            self.assertIsInstance(tool.description, str)

            # Check tool has args_schema
            self.assertIsNotNone(tool.args_schema)

            # Verify metadata can be set (as done in main.py)
            tool.metadata = {'server_name': "calculator"}
            self.assertEqual(tool.metadata['server_name'], "calculator")

    async def test_asyncio_run_from_running_loop_fixed(self):
        """Test that run_code works correctly even when called from an async context

        This test runs in an async context (note the 'async def' above).
        The fix detects the running event loop and creates a new loop in a separate thread,
        avoiding the 'asyncio.run() cannot be called from a running event loop' error.
        Tests multiple consecutive calls to ensure the fix is robust.
        """
        for tool in calculator_tools:
            tool.metadata = {'server_name': "calculator"}

        tracker.set_tools(calculator_tools)

        # First code execution
        code1 = '''
result = await call_api("calculator", "evaluate_expression", {"expression": "5 + 3"})
print(f"Result 1: {result['result']}")
'''

        state = AgentState(input="test", url="")
        output1, locals_dict1 = await run_code(code1, state)
        print(f"First Output: {output1}")

        # Verify first execution worked correctly
        self.assertNotIn("asyncio.run() cannot be called from a running event loop", output1)
        self.assertIn("Result 1: 8.0", output1)

        # Second consecutive code execution
        code2 = '''
factorial_result = await call_api("calculator", "calculate_factorial", {"n": 5})
print(f"Result 2: {factorial_result['result']}")
'''

        output2, locals_dict2 = await run_code(code2, state)
        print(f"Second Output: {output2}")

        # Verify second execution also worked correctly
        self.assertNotIn("asyncio.run() cannot be called from a running event loop", output2)
        self.assertIn("Result 2: 120", output2)


if __name__ == "__main__":
    # Run tests
    unittest.main()
