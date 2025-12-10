#!/usr/bin/env python3
"""
Run the ngrok connectivity test inside an E2B sandbox.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox

# Load environment variables
load_dotenv()


async def run_code_async(sandbox, test_code):
    """Run code in sandbox asynchronously."""
    loop = asyncio.get_event_loop()
    execution = await loop.run_in_executor(None, sandbox.run_code, test_code)
    return execution


async def main():
    # Read the test script
    with open('test_e2b_ngrok.py', 'r') as f:
        test_code = f.read()

    print("=" * 80)
    print("USING EXISTING E2B SANDBOX TO TEST NGROK CONNECTIVITY (ASYNC)")
    print("=" * 80)
    print()

    # Get sandbox ID from environment or use default
    sandbox_id = os.getenv('E2B_SANDBOX_ID')

    try:
        if sandbox_id:
            print(f"Connecting to existing sandbox: {sandbox_id}")
            sandbox = Sandbox.connect(sandbox_id)
        else:
            # Create E2B sandbox with the cuga-langchain template
            print("Creating E2B sandbox with 'cuga-langchain' template...")
            sandbox = Sandbox.create("cuga-langchain")
            print("Sandbox created")

        try:
            print()
            print("Executing test script in sandbox asynchronously...")
            print("-" * 80)

            # Run the test code asynchronously
            execution = await run_code_async(sandbox, test_code)

            # Print all output
            if execution.logs.stdout:
                print("STDOUT:")
                for line in execution.logs.stdout:
                    print(line)

            if execution.logs.stderr:
                print("\nSTDERR:")
                for line in execution.logs.stderr:
                    print(line)

            if execution.error:
                print(f"\nEXECUTION ERROR: {execution.error}")

            print("-" * 80)
            print()

            if execution.error:
                print("❌ TEST FAILED IN E2B SANDBOX")
                sys.exit(1)
            else:
                print("✅ TEST COMPLETED IN E2B SANDBOX")
                sys.exit(0)

        finally:
            # Only close if we created it (not if we connected to existing)
            if not sandbox_id and sandbox:
                print("\nClosing sandbox...")
                try:
                    sandbox.kill()
                except Exception as e:
                    print(f"Warning: Could not close sandbox: {e}")

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
