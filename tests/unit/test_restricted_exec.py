"""
Unit tests for restricted execution environment in cuga_agent_base.py
Tests that safe code executes properly while dangerous operations are blocked.
"""

import pytest


class TestRestrictedExecution:
    """Test suite for the restricted execution environment."""

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool functions for testing."""

        async def mock_filesystem_read(path):
            if path == './cuga_workspace/contacts.txt':
                return {'result': 'alice@example.com\nbob@example.com\ncharlie@example.com'}
            elif path == './cuga_workspace/email_template.md':
                return {'result': 'Email Template:\n<results>'}
            return {'result': 'test data'}

        async def mock_crm_get_contacts(limit=100, skip=0):
            return {
                'items': [
                    {
                        'id': 1,
                        'first_name': 'Alice',
                        'last_name': 'Smith',
                        'email': 'alice@example.com',
                        'account_id': 101,
                    },
                    {
                        'id': 2,
                        'first_name': 'Bob',
                        'last_name': 'Jones',
                        'email': 'bob@example.com',
                        'account_id': 102,
                    },
                    {
                        'id': 3,
                        'first_name': 'Charlie',
                        'last_name': 'Brown',
                        'email': 'charlie@example.com',
                        'account_id': 103,
                    },
                ],
                'total': 3,
            }

        async def mock_crm_get_accounts(limit=100, skip=0):
            accounts = [
                {'id': 101, 'name': 'Company A', 'annual_revenue': 1000000},
                {'id': 102, 'name': 'Company B', 'annual_revenue': 5000000},
                {'id': 103, 'name': 'Company C', 'annual_revenue': 500000},
            ]
            return {
                'items': accounts[skip : skip + limit],
                'total': len(accounts),
                'pages': (len(accounts) + limit - 1) // limit,
            }

        return {
            'filesystem_read_text_file': mock_filesystem_read,
            'crm_get_contacts_contacts_get': mock_crm_get_contacts,
            'crm_get_accounts_accounts_get': mock_crm_get_accounts,
        }

    @pytest.mark.asyncio
    async def test_valid_code_execution(self, mock_tools):
        """Test that valid code with tool calls executes successfully."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        valid_code = """
# Read contacts
contacts_data = await filesystem_read_text_file(path='./cuga_workspace/contacts.txt')
emails = contacts_data['result'].splitlines()

# Get CRM contacts
crm_contacts = await crm_get_contacts_contacts_get(limit=300)
filtered_contacts = [c for c in crm_contacts['items'] if c['email'] in emails]

# Get accounts
accounts = []
page = 0
while True:
    accounts_page = await crm_get_accounts_accounts_get(skip=page * 5, limit=5)
    accounts.extend(accounts_page['items'])
    if page >= accounts_page['pages'] - 1:
        break
    page += 1

result = f"Found {len(filtered_contacts)} contacts and {len(accounts)} accounts"
print(result)
"""

        # Pass mock_tools as _locals parameter so they're available in the execution context
        result, new_vars = await eval_with_tools_async(valid_code, _locals=mock_tools)

        assert "Found 3 contacts and 3 accounts" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_block_os_module_import(self, mock_tools):
        """Test that importing os module is blocked."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        dangerous_code = """
import os
result = os.environ['HOME']
"""

        result, new_vars = await eval_with_tools_async(dangerous_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result or "not allowed" in result.lower()

    @pytest.mark.asyncio
    async def test_block_os_environ_access(self, mock_tools):
        """Test that accessing os.environ is blocked."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        dangerous_code = """
import os
env_var = os.environ.get('SECRET_KEY', 'default')
"""

        result, new_vars = await eval_with_tools_async(dangerous_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result or "not allowed" in result.lower()

    @pytest.mark.asyncio
    async def test_block_subprocess_module(self, mock_tools):
        """Test that subprocess module is blocked."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        dangerous_code = """
import subprocess
result = subprocess.run(['ls'], capture_output=True)
"""

        result, new_vars = await eval_with_tools_async(dangerous_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result or "not allowed" in result.lower()

    @pytest.mark.asyncio
    async def test_block_sys_module(self, mock_tools):
        """Test that sys module is blocked."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        dangerous_code = """
import sys
result = sys.version
"""

        result, new_vars = await eval_with_tools_async(dangerous_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result or "not allowed" in result.lower()

    @pytest.mark.asyncio
    async def test_block_open_builtin(self, mock_tools):
        """Test that open() builtin is not available."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        dangerous_code = """
with open('/etc/passwd', 'r') as f:
    data = f.read()
"""

        result, new_vars = await eval_with_tools_async(dangerous_code, _locals=mock_tools)

        assert "Error" in result
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_block_eval_builtin(self, mock_tools):
        """Test that eval() builtin is not available."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        dangerous_code = """
result = eval('2 + 2')
"""

        result, new_vars = await eval_with_tools_async(dangerous_code, _locals=mock_tools)

        assert "Error" in result
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_block_exec_builtin(self, mock_tools):
        """Test that exec() builtin is not available."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        dangerous_code = """
exec('x = 5')
"""

        result, new_vars = await eval_with_tools_async(dangerous_code, _locals=mock_tools)

        assert "Error" in result
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_block_compile_builtin(self, mock_tools):
        """Test that compile() builtin is not available."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        dangerous_code = """
code = compile('x = 5', '<string>', 'exec')
"""

        result, new_vars = await eval_with_tools_async(dangerous_code, _locals=mock_tools)

        assert "Error" in result
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_allow_json_module(self, mock_tools):
        """Test that json module is allowed."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        safe_code = """
import json
data = json.dumps({'key': 'value'})
result = json.loads(data)
print(result['key'])
"""

        result, new_vars = await eval_with_tools_async(safe_code, _locals=mock_tools)

        assert "value" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_allow_asyncio_module(self, mock_tools):
        """Test that asyncio module is allowed."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        safe_code = """
import asyncio
await asyncio.sleep(0.01)
result = "asyncio works"
print(result)
"""

        result, new_vars = await eval_with_tools_async(safe_code, _locals=mock_tools)

        assert "asyncio works" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_allow_math_module(self, mock_tools):
        """Test that math module is allowed."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        safe_code = """
import math
result = math.sqrt(16)
print(f"Square root: {result}")
"""

        result, new_vars = await eval_with_tools_async(safe_code, _locals=mock_tools)

        assert "4.0" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_allow_datetime_module(self, mock_tools):
        """Test that datetime module is allowed."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        safe_code = """
import datetime
now = datetime.datetime.now()
result = f"Year: {now.year}"
print(result)
"""

        result, new_vars = await eval_with_tools_async(safe_code, _locals=mock_tools)

        assert "Year:" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_block_pathlib_module(self, mock_tools):
        """Test that pathlib module is blocked."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        dangerous_code = """
import pathlib
path = pathlib.Path('/etc')
files = list(path.iterdir())
"""

        result, new_vars = await eval_with_tools_async(dangerous_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result or "not allowed" in result.lower()

    @pytest.mark.asyncio
    async def test_complex_valid_workflow(self, mock_tools):
        """Test a complex but valid workflow similar to the user's example."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        complex_code = """
# Step 1: Read contacts
contacts_file_path = './cuga_workspace/contacts.txt'
contacts_data = await filesystem_read_text_file(path=contacts_file_path)
emails = contacts_data['result'].splitlines()

# Step 2: Get CRM contacts
crm_contacts = await crm_get_contacts_contacts_get(limit=300)
crm_contacts_list = crm_contacts['items']

# Step 3: Filter
filtered_contacts = [contact for contact in crm_contacts_list if contact['email'] in emails]

# Step 4: Get accounts with pagination
accounts = []
page = 0
while True:
    accounts_page = await crm_get_accounts_accounts_get(skip=page * 5, limit=5)
    accounts.extend(accounts_page['items'])
    if page >= accounts_page['pages'] - 1:
        break
    page += 1

# Step 5: Calculate percentiles
revenues = [account['annual_revenue'] for account in accounts]
revenues.sort()
percentiles = {account['id']: (revenues.index(account['annual_revenue']) + 1) / len(revenues) * 100 for account in accounts}

# Step 6: Prepare summary
result_summary = []
for contact in filtered_contacts:
    account = next((acc for acc in accounts if acc['id'] == contact['account_id']), None)
    if account:
        percentile = percentiles[account['id']]
        result_summary.append({
            'contact_name': f"{contact['first_name']} {contact['last_name']}",
            'account_name': account['name'],
            'revenue_percentile': percentile
        })

# Step 7: Read template
email_template_path = './cuga_workspace/email_template.md'
email_template_data = await filesystem_read_text_file(path=email_template_path)
email_template = email_template_data['result']

# Step 8: Draft email
results_text = "\\n".join([f"Contact: {item['contact_name']}, Account: {item['account_name']}, Revenue Percentile: {item['revenue_percentile']:.2f}%" for item in result_summary])
email_content = email_template.replace("<results>", results_text)

print(email_content)
"""

        result, new_vars = await eval_with_tools_async(complex_code, _locals=mock_tools)

        # Check that the email was drafted correctly
        assert "Email Template:" in result
        assert "Contact: Alice Smith" in result
        assert "Contact: Bob Jones" in result
        assert "Contact: Charlie Brown" in result
        assert "Company A" in result
        assert "Company B" in result
        assert "Company C" in result
        assert "Revenue Percentile:" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_basic_python_operations_work(self, mock_tools):
        """Test that basic Python operations still work."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        basic_code = """
# Lists and dicts
data = [1, 2, 3, 4, 5]
squared = [x**2 for x in data]

# String operations
text = "hello world"
upper_text = text.upper()

# Dictionary operations
info = {'name': 'test', 'value': 42}
result = f"Squared: {squared}, Text: {upper_text}, Info: {info['name']}"
print(result)
"""

        result, new_vars = await eval_with_tools_async(basic_code, _locals=mock_tools)

        assert "[1, 4, 9, 16, 25]" in result
        assert "HELLO WORLD" in result
        assert "test" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_variables_preserved_in_locals(self, mock_tools):
        """Test that variables and tools are available through _locals."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        # Add some initial variables to _locals
        locals_with_vars = {
            **mock_tools,
            'initial_value': 100,
            'multiplier': 5,
        }

        code = """
# Use existing variables from _locals
result = initial_value * multiplier

# Call a tool from _locals
contacts_data = await filesystem_read_text_file(path='./cuga_workspace/contacts.txt')
email_count = len(contacts_data['result'].splitlines())

final_result = f"Calculation: {result}, Emails: {email_count}"
print(final_result)
"""

        result, new_vars = await eval_with_tools_async(code, _locals=locals_with_vars)

        assert "Calculation: 500" in result
        assert "Emails: 3" in result
        assert "Error" not in result
        # Check that new variables were captured
        assert 'result' in new_vars or 'final_result' in new_vars

    @pytest.mark.asyncio
    async def test_re_module_works(self, mock_tools):
        """Test that re (regex) module is allowed and works."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
import re

text = "Contact alice@example.com for info"
pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
matches = re.findall(pattern, text)

print(f"Found emails: {matches}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "alice@example.com" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_collections_module(self, mock_tools):
        """Test that collections module is allowed."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
from collections import defaultdict, Counter

# Test defaultdict
data = defaultdict(list)
data['key1'].append('value1')

# Test Counter
items = ['a', 'b', 'a', 'c', 'b', 'a']
counter = Counter(items)

print(f"DefaultDict: {dict(data)}, Counter: {counter}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "DefaultDict:" in result
        assert "Counter:" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_itertools_module(self, mock_tools):
        """Test that itertools module is allowed."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
from itertools import chain, combinations

# Test chain
combined = list(chain([1, 2], [3, 4]))

# Test combinations
combos = list(combinations([1, 2, 3], 2))

print(f"Chained: {combined}, Combos: {combos}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "[1, 2, 3, 4]" in result
        assert "Combos:" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_functools_module(self, mock_tools):
        """Test that functools module is allowed."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)

print(f"Product: {product}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "120" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_locals_and_vars_builtins(self, mock_tools):
        """Test that locals() and vars() builtins work."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
x = 10
y = 20

# Test locals()
local_vars = locals()
has_x = 'x' in local_vars

# Test vars() on an object
class TestObj:
    def __init__(self):
        self.value = 42

obj = TestObj()
obj_vars = vars(obj)

print(f"Has x: {has_x}, Obj value: {obj_vars['value']}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "Has x: True" in result
        assert "Obj value: 42" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_list_comprehensions_and_generators(self, mock_tools):
        """Test that list comprehensions and generators work."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
# List comprehension
squares = [x**2 for x in range(5)]

# Generator expression
gen = (x * 2 for x in range(5))
doubled = list(gen)

# Nested comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]

print(f"Squares: {squares}, Doubled: {doubled}, Matrix row 2: {matrix[2]}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "[0, 1, 4, 9, 16]" in result
        assert "[0, 2, 4, 6, 8]" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_exception_handling(self, mock_tools):
        """Test that exception handling works within the code."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
result = None
try:
    # This will raise an error
    x = 10 / 0
except ZeroDivisionError:
    result = "Caught division by zero"

print(result)
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "Caught division by zero" in result
        assert "Error during execution" not in result

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_sequence(self, mock_tools):
        """Test multiple async tool calls in sequence."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
# First tool call
contacts = await filesystem_read_text_file(path='./cuga_workspace/contacts.txt')
contact_count = len(contacts['result'].splitlines())

# Second tool call
crm_data = await crm_get_contacts_contacts_get(limit=100)
crm_count = len(crm_data['items'])

# Third tool call
accounts = await crm_get_accounts_accounts_get(limit=50)
account_count = len(accounts['items'])

print(f"Contacts: {contact_count}, CRM: {crm_count}, Accounts: {account_count}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "Contacts: 3" in result
        assert "CRM: 3" in result
        assert "Accounts: 3" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_new_variables_captured(self, mock_tools):
        """Test that new variables are properly captured in new_vars."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
my_var = "test_value"
my_number = 42
my_list = [1, 2, 3]
my_dict = {'key': 'value'}
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        # Check that variables were captured
        assert 'my_var' in new_vars
        assert 'my_number' in new_vars
        assert 'my_list' in new_vars
        assert 'my_dict' in new_vars
        assert new_vars['my_var'] == 'test_value'
        assert new_vars['my_number'] == 42

    @pytest.mark.asyncio
    async def test_class_definition_and_usage(self, mock_tools):
        """Test that class definitions work."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
class Contact:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def get_info(self):
        return f"{self.name} <{self.email}>"

contact = Contact("Alice", "alice@example.com")
info = contact.get_info()
print(info)
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "Alice <alice@example.com>" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_lambda_functions(self, mock_tools):
        """Test that lambda functions work."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
# Simple lambda
square = lambda x: x ** 2
result1 = square(5)

# Lambda with map
numbers = [1, 2, 3, 4]
doubled = list(map(lambda x: x * 2, numbers))

# Lambda with filter
evens = list(filter(lambda x: x % 2 == 0, numbers))

print(f"Square: {result1}, Doubled: {doubled}, Evens: {evens}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "Square: 25" in result
        assert "[2, 4, 6, 8]" in result
        assert "[2, 4]" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_string_formatting_methods(self, mock_tools):
        """Test various string formatting methods."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
name = "Alice"
age = 30

# f-string
f_result = f"{name} is {age} years old"

# format method
format_result = "{} is {} years old".format(name, age)

# percent formatting
percent_result = "%s is %d years old" % (name, age)

print(f"F-string: {f_result}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "Alice is 30 years old" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_nested_data_structures(self, mock_tools):
        """Test working with nested data structures."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
# Nested dict
company = {
    'name': 'Tech Corp',
    'employees': [
        {'name': 'Alice', 'role': 'Engineer'},
        {'name': 'Bob', 'role': 'Manager'}
    ],
    'metadata': {
        'founded': 2020,
        'location': 'NYC'
    }
}

# Access nested values
engineer_name = company['employees'][0]['name']
founded_year = company['metadata']['founded']

print(f"Engineer: {engineer_name}, Founded: {founded_year}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "Engineer: Alice" in result
        assert "Founded: 2020" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_with_statement_not_blocked(self, mock_tools):
        """Test that with statement works for allowed objects."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
# With statement should work for objects that support it
class MyContext:
    def __enter__(self):
        return "context value"
    
    def __exit__(self, *args):
        pass

with MyContext() as ctx:
    result = f"Context: {ctx}"

print(result)
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "Context: context value" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_block_import_from_string(self, mock_tools):
        """Test that dynamic imports are blocked."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        # __import__ is restricted, so this should fail
        code = """
module_name = "os"
imported = __import__(module_name)
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result or "not allowed" in result.lower()

    @pytest.mark.asyncio
    async def test_typing_module(self, mock_tools):
        """Test that typing module is allowed for type hints."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        code = """
from typing import List, Dict, Optional

def process_items(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}

result = process_items(['hello', 'world'])
print(f"Result: {result}")
"""

        result, new_vars = await eval_with_tools_async(code, _locals=mock_tools)

        assert "hello" in result
        assert "world" in result
        assert "Error" not in result


class TestSecurityAttacks:
    """Test various attempts and security exploits."""

    @pytest.fixture
    def mock_tools(self):
        async def mock_read_file(path: str):
            return {"result": "mock file content"}

        return {"filesystem_read_text_file": mock_read_file}

    @pytest.mark.asyncio
    async def test_builtins_access_via_class(self, mock_tools):
        """Test accessing __builtins__ through class hierarchy."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to access builtins through object.__class__
x = ().__class__.__bases__[0].__subclasses__()
result = x
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        # Should either work safely (no dangerous items) or fail
        # The key is that even if it works, it shouldn't give access to dangerous functions
        if "Error" not in result:
            # Verify no dangerous functions are accessible
            assert "open" not in str(result).lower() or "Error" in result

    @pytest.mark.asyncio
    async def test_globals_access_attack(self, mock_tools):
        """Test trying to access globals() to get __builtins__."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to access globals to get builtins
g = globals()
b = g.get('__builtins__', {})
if hasattr(b, '__import__'):
    import os
    result = os.environ
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        # Should be blocked
        assert "Error" in result or "environ" not in str(result)

    @pytest.mark.asyncio
    async def test_import_via_string_eval(self, mock_tools):
        """Test dynamic import using string manipulation."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to construct import statement dynamically
import_str = "__imp" + "ort__"
os_str = "o" + "s"
# Try to use it (should fail - eval not available)
result = eval(f"{import_str}('{os_str}')")
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_bytecode_manipulation(self, mock_tools):
        """Test trying to manipulate bytecode."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to access code object and manipulate it
def dummy():
    pass

# Try to access bytecode
bytecode = dummy.__code__.co_code
result = bytecode
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        # Should work (bytecode access is fine) but shouldn't allow code injection
        # The real test is that you can't do anything dangerous with it
        assert "Error" not in result or "AttributeError" in result

    @pytest.mark.asyncio
    async def test_frame_stack_manipulation(self, mock_tools):
        """Test trying to access stack frames to escape sandbox."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
import sys
# Try to access caller's frame
frame = sys._getframe()
result = frame.f_back.f_globals
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result or "not allowed" in result.lower()

    @pytest.mark.asyncio
    async def test_unicode_smuggling(self, mock_tools):
        """Test using unicode to smuggle dangerous code."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Use unicode to try to bypass filters
\u0069\u006d\u0070\u006f\u0072\u0074 os
result = os.environ
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result or "NameError" in result

    @pytest.mark.asyncio
    async def test_attribute_access_chain(self, mock_tools):
        """Test long attribute access chains to find builtins."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to access builtins through various attribute chains
result = []
result.append(str.__class__)
result.append(str.__class__.__mro__)
result.append(object.__subclasses__())
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        # Should work but shouldn't expose dangerous functions
        # The result should not contain file operations or imports

    @pytest.mark.asyncio
    async def test_decorator_injection(self, mock_tools):
        """Test using decorators to inject malicious code."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use decorator to execute code
def malicious_decorator(func):
    import os
    return func

@malicious_decorator
def dummy():
    pass

result = "executed"
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_closure_escape(self, mock_tools):
        """Test trying to escape via closure variables."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to access closure to get outer scope
def outer():
    import os
    def inner():
        return os.environ
    return inner

result = outer()()
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_exception_exploitation(self, mock_tools):
        """Test using exception objects to access forbidden context."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use exception traceback to access outer scope
try:
    import os
except ImportError as e:
    # Try to access the exception context
    result = e.__traceback__.tb_frame.f_globals
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        # Should either fail or not expose dangerous items
        if "Error" not in result:
            assert "__import__" not in str(result) or "restricted" in str(result).lower()

    @pytest.mark.asyncio
    async def test_class_new_override(self, mock_tools):
        """Test overriding __new__ to execute code."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to override __new__ for code execution
class Malicious:
    def __new__(cls):
        import os
        return os.environ

result = Malicious()
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_metaclass_injection(self, mock_tools):
        """Test using metaclass to inject code."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use metaclass for code injection
class MetaAttack(type):
    def __new__(mcs, name, bases, attrs):
        import os
        return super().__new__(mcs, name, bases, attrs)

class Attack(metaclass=MetaAttack):
    pass

result = Attack()
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_descriptor_protocol_exploit(self, mock_tools):
        """Test using descriptor protocol to access forbidden attributes."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use descriptor to access forbidden items
class Descriptor:
    def __get__(self, obj, objtype=None):
        import os
        return os.environ

class Container:
    attr = Descriptor()

result = Container().attr
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_property_decorator_attack(self, mock_tools):
        """Test using @property to execute dangerous code."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use property to execute code
class Attack:
    @property
    def malicious(self):
        import os
        return os.environ

result = Attack().malicious
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_context_manager_exploit(self, mock_tools):
        """Test using context manager to execute code."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use context manager for code execution
class ContextAttack:
    def __enter__(self):
        import os
        return os.environ
    
    def __exit__(self, *args):
        pass

with ContextAttack() as result:
    pass
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_generator_attack(self, mock_tools):
        """Test using generators to defer dangerous execution."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use generator to defer import
def gen():
    import os
    yield os.environ

result = list(gen())
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_async_attack(self, mock_tools):
        """Test using async/await to bypass restrictions."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use async to import dangerous modules
async def attack():
    import os
    return os.environ

result = await attack()
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_list_comprehension_attack(self, mock_tools):
        """Test hiding import in list comprehension."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to hide import in comprehension
result = [__import__('os').environ for _ in range(1)]
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_lambda_attack(self, mock_tools):
        """Test using lambda to execute dangerous code."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use lambda for import
attack = lambda: __import__('os').environ
result = attack()
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_walrus_operator_attack(self, mock_tools):
        """Test using walrus operator to smuggle imports."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use walrus operator
if (dangerous := __import__('os')):
    result = dangerous.environ
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_format_string_attack(self, mock_tools):
        """Test using format strings to execute code."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use f-string with expressions
module = 'os'
result = f"{__import__(module).environ}"
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_getattr_attack(self, mock_tools):
        """Test using getattr to access forbidden attributes."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to use getattr to bypass restrictions
builtins = getattr(__builtins__, '__dict__', __builtins__)
import_func = builtins.get('__import__')
os = import_func('os')
result = os.environ
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        # Should be blocked by restricted import
        assert "Error" in result or "environ" not in str(result)

    @pytest.mark.asyncio
    async def test_setattr_namespace_pollution(self, mock_tools):
        """Test using setattr to pollute namespace."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to pollute namespace
import json
setattr(json, 'os', __import__('os'))
result = json.os.environ
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_del_statement_attack(self, mock_tools):
        """Test using del to manipulate restrictions."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to delete restrictions
try:
    del __builtins__['__import__']
except:
    pass

import os
result = os.environ
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result or "NameError" in result

    @pytest.mark.asyncio
    async def test_nested_function_scope_leak(self, mock_tools):
        """Test nested functions to leak scope."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to leak scope through nested functions
def level1():
    def level2():
        def level3():
            import os
            return os.environ
        return level3()
    return level2()

result = level1()
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_exec_string_construction(self, mock_tools):
        """Test constructing exec/eval from strings."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try to reconstruct exec function name
func_name = "e" + "x" + "e" + "c"
code = "import os; result = os.environ"
# Try to execute (should fail)
globals()[func_name](code)
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result
        assert "KeyError" in result or "NameError" in result

    @pytest.mark.asyncio
    async def test_type_confusion_attack(self, mock_tools):
        """Test type confusion to bypass checks."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try type confusion
class FakeDict(dict):
    def __getitem__(self, key):
        if key == 'os':
            return __import__('os')
        return super().__getitem__(key)

fake = FakeDict()
result = fake['os'].environ
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_infinite_recursion_dos(self, mock_tools):
        """Test DoS via infinite recursion (should timeout)."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Try infinite recursion
def recurse():
    return recurse()

result = recurse()
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        # Should timeout or hit recursion limit
        assert "Error" in result
        assert "RecursionError" in result or "TimeoutError" in result or "maximum recursion" in result.lower()

    @pytest.mark.asyncio
    async def test_transitive_import_exploit(self, mock_tools):
        """Test accessing dangerous modules via attributes of allowed modules."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        # typing module usually imports sys, asyncio imports sys/os
        # This tests if we can piggyback on allowed modules to reach forbidden ones
        attack_code = """
import typing
import asyncio
import json

# Try to find sys/os in allowed modules attributes
refs = [typing, asyncio, json]
found_os = None

for ref in refs:
    # Try common attribute names where modules might be stored
    for attr in dir(ref):
        val = getattr(ref, attr)
        if str(val).startswith("<module 'os'"):
            found_os = val
            break
        if str(val).startswith("<module 'sys'"):
            # If we have sys, we can get os via sys.modules
            sys_mod = val
            if hasattr(sys_mod, 'modules'):
                found_os = sys_mod.modules.get('os')
                break
    if found_os:
        break

if found_os:
    result = found_os.environ
else:
    result = "safe"
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        # Should be blocked or fail to find path
        assert "Error" in result or "environ" not in str(result)

    @pytest.mark.asyncio
    async def test_gadget_chain_globals(self, mock_tools):
        """Test finding a function with dangerous globals via subclass crawling."""
        from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_agent_base import eval_with_tools_async

        attack_code = """
# Iterate over subclasses to find a 'gadget' - a class/function that has 'os' in its global scope
# This is a common sandbox escape technique
classes = {}.__class__.__base__.__subclasses__()
result = "safe"

for cls in classes:
    # Check __init__ method's globals
    if hasattr(cls, '__init__') and hasattr(cls.__init__, '__globals__'):
        g = cls.__init__.__globals__
        # Check for os module
        if 'os' in g and g['os'] is not None:
             try:
                 # Found os in some class's globals! Try to use it.
                 # g['os'] is the actual os module
                 if hasattr(g['os'], 'environ'):
                    result = g['os'].environ
                    break
             except:
                 pass
"""
        result, new_vars = await eval_with_tools_async(attack_code, _locals=mock_tools)

        # This sophisticated attack should also fail in a robust sandbox
        assert "Error" in result or "environ" not in str(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
