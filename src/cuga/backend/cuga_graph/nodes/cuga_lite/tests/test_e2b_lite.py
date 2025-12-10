"""Unit tests for E2B sandbox integration in CUGA Lite mode."""

import pytest
from unittest.mock import patch

from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import (
    _filter_new_variables,
    _execute_in_e2b_sandbox,
    eval_with_tools_async,
)


class TestFilterNewVariables:
    """Test suite for _filter_new_variables helper function."""

    def test_filter_basic_serializable_types(self):
        """Test filtering with basic serializable types."""
        original_keys = {'existing_var'}
        all_locals = {
            'existing_var': 'old',
            'new_str': 'hello',
            'new_int': 42,
            'new_float': 3.14,
            'new_bool': True,
            'new_none': None,
        }

        result = _filter_new_variables(all_locals, original_keys)

        assert len(result) == 5
        assert result['new_str'] == 'hello'
        assert result['new_int'] == 42
        assert result['new_float'] == 3.14
        assert result['new_bool'] is True
        assert result['new_none'] is None

    def test_filter_collections(self):
        """Test filtering with lists, dicts, and tuples."""
        original_keys = set()
        all_locals = {
            'my_list': [1, 2, 3],
            'my_dict': {'a': 1, 'b': 2},
            'my_tuple': (1, 2, 3),
            'nested': {'list': [1, 2], 'dict': {'x': 10}},
        }

        result = _filter_new_variables(all_locals, original_keys)

        assert len(result) == 4
        assert result['my_list'] == [1, 2, 3]
        assert result['my_dict'] == {'a': 1, 'b': 2}
        assert result['my_tuple'] == (1, 2, 3)
        assert result['nested'] == {'list': [1, 2], 'dict': {'x': 10}}

    def test_filter_excludes_internal_variables(self):
        """Test that internal variables (starting with _) are filtered out."""
        original_keys = set()
        all_locals = {
            'public_var': 'visible',
            '_private_var': 'hidden',
            '__dunder__': 'hidden',
            '_internal': 'hidden',
        }

        result = _filter_new_variables(all_locals, original_keys)

        assert len(result) == 1
        assert result == {'public_var': 'visible'}

    def test_filter_excludes_non_serializable_types(self):
        """Test that non-serializable types (functions, classes, modules) are filtered out."""

        def test_function():
            pass

        class TestClass:
            pass

        import types

        test_module = types.ModuleType('test_module')

        original_keys = set()
        all_locals = {
            'serializable': 'keep',
            'function': test_function,
            'class': TestClass,
            'module': test_module,
        }

        result = _filter_new_variables(all_locals, original_keys)

        assert len(result) == 1
        assert result == {'serializable': 'keep'}

    def test_filter_empty_new_vars(self):
        """Test when there are no new variables."""
        original_keys = {'var1', 'var2'}
        all_locals = {'var1': 'a', 'var2': 'b'}

        result = _filter_new_variables(all_locals, original_keys)

        assert result == {}


class TestExecuteInE2BSandbox:
    """Test suite for _execute_in_e2b_sandbox helper function."""

    def test_import_error_handling(self):
        """Test that RuntimeError is raised when e2b-code-interpreter is not available."""
        with patch('cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base.E2B_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="e2b-code-interpreter package not installed"):
                _execute_in_e2b_sandbox("print('hello')")

    def test_successful_execution_with_output(self):
        """Test successful E2B execution with stdout output."""
        from unittest.mock import Mock, MagicMock

        with patch('cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base.Sandbox') as mock_sandbox_class:
            mock_execution = Mock()
            mock_execution.error = None
            # Include the dict in stdout (this is where we parse it from)
            mock_execution.logs.stdout = ['Hello from E2B', 'Result: 42', "{'result': 42}"]
            mock_execution.text = None

            mock_sandbox = MagicMock()
            mock_sandbox.run_code.return_value = mock_execution
            mock_sandbox.__enter__.return_value = mock_sandbox
            mock_sandbox.__exit__.return_value = None
            mock_sandbox_class.create.return_value = mock_sandbox

            result, locals_dict = _execute_in_e2b_sandbox("print('test')")

            assert "Hello from E2B" in result
            assert "Result: 42" in result
            assert locals_dict == {'result': 42}

    def test_execution_error_handling(self):
        """Test E2B execution with error."""
        from unittest.mock import Mock, MagicMock

        with patch('cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base.Sandbox') as mock_sandbox_class:
            mock_execution = Mock()
            mock_execution.error = "NameError: name 'undefined_var' is not defined"

            mock_sandbox = MagicMock()
            mock_sandbox.run_code.return_value = mock_execution
            mock_sandbox.__enter__.return_value = mock_sandbox
            mock_sandbox.__exit__.return_value = None
            mock_sandbox_class.create.return_value = mock_sandbox

            with pytest.raises(RuntimeError, match="E2B execution error"):
                _execute_in_e2b_sandbox("print(undefined_var)")

    def test_empty_locals_handling(self):
        """Test when E2B returns empty locals."""
        from unittest.mock import Mock, MagicMock

        with patch('cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base.Sandbox') as mock_sandbox_class:
            mock_execution = Mock()
            mock_execution.error = None
            mock_execution.logs.stdout = ['Hello']
            mock_execution.text = ''

            mock_sandbox = MagicMock()
            mock_sandbox.run_code.return_value = mock_execution
            mock_sandbox.__enter__.return_value = mock_sandbox
            mock_sandbox.__exit__.return_value = None
            mock_sandbox_class.create.return_value = mock_sandbox

            result, locals_dict = _execute_in_e2b_sandbox("print('Hello')")

            assert result == "Hello"
            assert locals_dict == {}


class TestEvalWithToolsAsyncE2B:
    """Test suite for eval_with_tools_async with E2B integration."""

    @pytest.mark.asyncio
    async def test_eval_with_e2b_enabled(self):
        """Test eval_with_tools_async routes to E2B when enabled."""
        with patch('cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base.settings') as mock_settings:
            with patch(
                'cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base._execute_in_e2b_sandbox'
            ) as mock_e2b:
                mock_settings.advanced_features.e2b_sandbox = True
                mock_e2b.return_value = ("42", {'result': 42})

                code = "result = 40 + 2\nprint(result)"
                _locals = {}

                output, new_vars = await eval_with_tools_async(code, _locals)

                mock_e2b.assert_called_once()
                assert output == "42"
                assert new_vars == {'result': 42}

    @pytest.mark.asyncio
    async def test_eval_with_e2b_disabled(self):
        """Test local execution when E2B is disabled."""
        with patch('cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base.settings') as mock_settings:
            mock_settings.advanced_features.e2b_sandbox = False

            code = "y = 20"
            _locals = {}

            output, new_vars = await eval_with_tools_async(code, _locals)

            assert new_vars == {'y': 20}
            # Output includes variable summary, so just check y is in there
            assert 'y' in str(new_vars)

    @pytest.mark.asyncio
    async def test_eval_filters_internal_variables(self):
        """Test that internal variables are filtered from results."""
        with patch('cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base.settings') as mock_settings:
            with patch(
                'cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base._execute_in_e2b_sandbox'
            ) as mock_e2b:
                mock_settings.advanced_features.e2b_sandbox = True

                # E2B returns both public and internal variables
                mock_e2b.return_value = (
                    "Success",
                    {'public_var': 'visible', '_private_var': 'hidden', '__dunder__': 'hidden', 'result': 42},
                )

                code = "result = 42"
                _locals = {}

                output, new_vars = await eval_with_tools_async(code, _locals)

                # Only public variables should be in new_vars
                assert 'public_var' in new_vars
                assert 'result' in new_vars
                assert '_private_var' not in new_vars
                assert '__dunder__' not in new_vars


class TestE2BWithVariablesAndTools:
    """Test suite for E2B execution with variables and async tool functions."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not pytest.importorskip("e2b_code_interpreter", reason="e2b-code-interpreter not installed"),
        reason="E2B not available",
    )
    async def test_e2b_with_variables_and_tools(self):
        """Test that E2B execution can access variables and call async tools from _locals."""

        # Define dummy async tool functions
        async def get_account_name(account_id: str) -> str:
            """Get account name by ID."""
            accounts = {"acc_1": "Acme Corp", "acc_2": "TechStart Inc"}
            return accounts.get(account_id, "Unknown")

        async def get_account_revenue(account_id: str) -> float:
            """Get account revenue by ID."""
            revenues = {"acc_1": 1500000.0, "acc_2": 850000.0}
            return revenues.get(account_id, 0.0)

        # Set up _locals with variables and tools
        _locals = {
            "get_account_name": get_account_name,
            "get_account_revenue": get_account_revenue,
            "target_account": "acc_1",
            "threshold": 1000000.0,
        }

        # Code that uses variables and calls tools
        code = """
# Use variable from previous execution
account_id = target_account

# Call async tools
name = await get_account_name(account_id)
revenue = await get_account_revenue(account_id)

# Compute result using another variable
is_high_value = revenue > threshold

print(f"Account: {name}")
print(f"Revenue: ${revenue:,.0f}")
print(f"High value: {is_high_value}")

result = {
    "account_id": account_id,
    "name": name,
    "revenue": revenue,
    "is_high_value": is_high_value
}
print("Done")  # Prevent auto-print of last line
"""

        with patch('cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base.settings') as mock_settings:
            mock_settings.advanced_features.e2b_sandbox = True

            # Execute in E2B
            output, new_vars = await eval_with_tools_async(code, _locals)

            # Verify output
            assert "Account: Acme Corp" in output
            assert "Revenue: $1,500,000" in output
            assert "High value: True" in output

            # Verify new variables
            assert 'result' in new_vars
            assert new_vars['result']['name'] == "Acme Corp"
            assert new_vars['result']['revenue'] == 1500000.0
            assert new_vars['result']['is_high_value'] is True
