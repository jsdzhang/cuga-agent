#!/usr/bin/env python3
"""
Test E2B sandbox connectivity to ngrok proxy.
This script runs inside an E2B sandbox and attempts to call the ngrok proxy endpoint.
"""

import json
import urllib.request
import urllib.error
import time


def test_ngrok_call():
    """Call the ngrok proxy endpoint from E2B sandbox."""

    url = "https://lieselotte-colligative-shabbily.ngrok-free.dev/functions/call"

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    payload = {
        "function_name": "digital_sales_get_my_accounts_my_accounts_get",
        "app_name": "digital_sales",
        "args": {},
    }

    print(f"[TEST] Starting request to {url}")
    print(f"[TEST] Payload: {json.dumps(payload, indent=2)}")
    print(f"[TEST] Headers: {json.dumps(headers, indent=2)}")

    start_time = time.time()

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')

    try:
        print(f"[TEST] Opening connection at {time.time() - start_time:.2f}s...")

        with urllib.request.urlopen(req, timeout=30) as response:
            elapsed = time.time() - start_time
            print(f"[TEST] Got response in {elapsed:.2f}s, status: {response.status}")

            response_data = response.read().decode('utf-8')
            print(f"[TEST] Body read, total time: {time.time() - start_time:.2f}s")

            try:
                result = json.loads(response_data)
                print("[TEST] Success! Response JSON:")
                print(json.dumps(result, indent=2))
                return result
            except Exception as parse_error:
                print(f"[TEST] JSON parse error: {parse_error}")
                print(f"[TEST] Raw response: {response_data}")
                return response_data

    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        print(f"[TEST] HTTP Error after {elapsed:.2f}s: {e.code} {e.reason}")
        error_body = ""
        try:
            error_body = e.read().decode('utf-8')
            print(f"[TEST] Error body: {error_body}")
        except Exception:
            pass
        raise

    except urllib.error.URLError as e:
        elapsed = time.time() - start_time
        print(f"[TEST] URL Error after {elapsed:.2f}s: {e.reason}")
        raise

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[TEST] Unexpected error after {elapsed:.2f}s: {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    print("=" * 80)
    print("E2B SANDBOX NGROK CONNECTIVITY TEST")
    print("=" * 80)

    try:
        result = test_ngrok_call()
        print("\n" + "=" * 80)
        print("TEST PASSED - Connection successful!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"TEST FAILED - {type(e).__name__}: {str(e)}")
        print("=" * 80)
        import traceback

        traceback.print_exc()
