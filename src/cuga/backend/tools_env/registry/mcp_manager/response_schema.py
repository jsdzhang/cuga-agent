import json
import yaml
from typing import List, Dict, Any
import httpx


def extract_api_ids(api_definitions: List[Dict[str, Any]]) -> List[str]:
    """
    Extract API operation IDs from a list of API definitions.

    Args:
        api_definitions: List of API definition dictionaries

    Returns:
        List of operation IDs
    """
    operation_ids = []
    for api_def in api_definitions:
        # Extract the 'name' part after mcplink prefix
        if 'description' in api_def:
            # Extract the operation ID from the description
            match = api_def['description'].split(" ")[0]

            operation_ids.append(match)

    return operation_ids


def load_openapi_spec(file_path: str) -> Dict[str, Any]:
    """
    Load an OpenAPI specification from a file.

    Args:
        file_path: Path to the OpenAPI JSON or YAML file

    Returns:
        OpenAPI specification as a dictionary
    """
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith(('.yaml', '.yml')):
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Please provide a JSON or YAML file.")
    except Exception as e:
        print(f"Error loading OpenAPI spec: {e}")
        return {}


async def fetch_openapi_spec(url: str) -> Dict[str, Any]:
    """
    Fetch an OpenAPI specification from a URL.

    Args:
        url: URL to the OpenAPI JSON or YAML file

    Returns:
        OpenAPI specification as a dictionary
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            if url.endswith('.json'):
                return response.json()
            elif url.endswith(('.yaml', '.yml')):
                return yaml.safe_load(response.text)
            else:
                # Try to determine from content type
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    return response.json()
                elif 'application/yaml' in content_type or 'text/yaml' in content_type:
                    return yaml.safe_load(response.text)
                else:
                    # Default to JSON
                    try:
                        return response.json()
                    except Exception:
                        return yaml.safe_load(response.text)
    except Exception as e:
        print(f"Error fetching OpenAPI spec: {e}")
        return {}


def resolve_ref(ref: str, openapi_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve a JSON reference ($ref) in an OpenAPI specification.

    Args:
        ref: Reference string (e.g., "#/components/schemas/Pet")
        openapi_spec: The complete OpenAPI specification

    Returns:
        The resolved schema
    """
    if not ref.startswith('#/'):
        print(f"External references not supported: {ref}")
        return {}

    # Remove the '#/' prefix and split the path
    path_parts = ref[2:].split('/')

    # Navigate through the OpenAPI spec to get the referenced object
    current = openapi_spec
    for part in path_parts:
        if part not in current:
            print(f"Reference part '{part}' not found in schema")
            return {}
        current = current[part]

    # If the resolved schema also has a $ref, resolve it recursively
    if isinstance(current, dict) and '$ref' in current:
        return resolve_ref(current['$ref'], openapi_spec)

    return current


def resolve_schema_references(schema: Dict[str, Any], openapi_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively resolve all references in a schema.

    Args:
        schema: The schema that may contain references
        openapi_spec: The complete OpenAPI specification

    Returns:
        Schema with all references resolved
    """
    if not schema:
        return {}

    # Create a copy of the schema to avoid modifying the original
    resolved_schema = schema.copy()

    # If the schema has a $ref, resolve it
    if '$ref' in resolved_schema:
        ref_schema = resolve_ref(resolved_schema['$ref'], openapi_spec)
        # Remove the $ref key
        resolved_schema.pop('$ref')
        # Update with the referenced schema
        resolved_schema.update(ref_schema)

    # Recursively resolve references in nested properties
    if 'properties' in resolved_schema:
        resolved_properties = {}
        for prop_name, prop_schema in resolved_schema['properties'].items():
            resolved_properties[prop_name] = resolve_schema_references(prop_schema, openapi_spec)
        resolved_schema['properties'] = resolved_properties

    # Resolve references in arrays
    if 'items' in resolved_schema and isinstance(resolved_schema['items'], dict):
        resolved_schema['items'] = resolve_schema_references(resolved_schema['items'], openapi_spec)

    # Resolve references in allOf, anyOf, oneOf
    for key in ['allOf', 'anyOf', 'oneOf']:
        if key in resolved_schema and isinstance(resolved_schema[key], list):
            resolved_schema[key] = [
                resolve_schema_references(item, openapi_spec) for item in resolved_schema[key]
            ]

    # Resolve references in additionalProperties
    if 'additionalProperties' in resolved_schema and isinstance(
        resolved_schema['additionalProperties'], dict
    ):
        resolved_schema['additionalProperties'] = resolve_schema_references(
            resolved_schema['additionalProperties'], openapi_spec
        )

    return resolved_schema


def extract_response_schema(openapi_spec: Dict[str, Any], operation_id: str) -> Dict[str, Any]:
    """
    Extract the response schema for a specific operation ID from an OpenAPI specification.
    Resolves all references in the schema.

    Args:
        openapi_spec: OpenAPI specification as a dictionary
        operation_id: Operation ID to extract the response schema for

    Returns:
        Response schema as a dictionary with all references resolved
    """
    schema = {}

    # Check if it's OpenAPI 3.x
    if 'openapi' in openapi_spec and openapi_spec['openapi'].startswith('3'):
        paths = openapi_spec.get('paths', {})

        # Iterate through paths and operations to find the matching operation ID
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if (
                    method in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']
                    and operation.get('operationId').lower() == operation_id.lower()
                ):
                    responses = operation.get('responses', {})

                    # Try to get 200 or 201 response first
                    for status_code in ['200', '201']:
                        if status_code in responses:
                            response = responses[status_code]
                            content = response.get('content', {})

                            # Check for application/json or */*
                            for content_type in ['application/json', '*/*']:
                                if content_type in content:
                                    schema = content[content_type].get('schema', {})
                                    return resolve_schema_references(schema, openapi_spec)

                    # If no 200/201 response, return the first response schema found
                    for status_code, response in responses.items():
                        content = response.get('content', {})
                        for content_type, content_schema in content.items():
                            schema = content_schema.get('schema', {})
                            return resolve_schema_references(schema, openapi_spec)

    # For OpenAPI 2.x (Swagger)
    elif 'swagger' in openapi_spec and openapi_spec['swagger'].startswith('2'):
        paths = openapi_spec.get('paths', {})

        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if (
                    method in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']
                    and operation.get('operationId') == operation_id
                ):
                    responses = operation.get('responses', {})

                    # Try to get 200 or 201 response first
                    for status_code in ['200', '201']:
                        if status_code in responses:
                            response = responses[status_code]
                            if 'schema' in response:
                                schema = response['schema']
                                return resolve_schema_references(schema, openapi_spec)

                    # If no 200/201 response, return the first response schema found
                    for status_code, response in responses.items():
                        if 'schema' in response:
                            schema = response['schema']
                            return resolve_schema_references(schema, openapi_spec)

    return resolve_schema_references(schema, openapi_spec)


def main(api_definitions, app_name):
    map = {}
    list_output = []
    # Example input from the prompt

    # Extract API operation IDs
    operation_ids = extract_api_ids(api_definitions)
    print("Extracted Operation IDs:")
    for op_id in operation_ids:
        print(f"  - {op_id}")

    with open(f'agent/api/api_schemas/{app_name}.json', 'r') as file:
        # Load the JSON data into a Python dictionary
        openapi_spec = json.load(file)

    if not openapi_spec:
        print("Failed to load OpenAPI spec")
        return

    # Extract response schemas for each operation ID
    print("\nResponse Schemas:")
    for operation_id in operation_ids:
        response_schema = extract_response_schema(openapi_spec, operation_id)
        print(f"\nOperation ID: {operation_id}")
        map[operation_id] = response_schema
        list_output.append(response_schema)
        print(f"Response Schema: {json.dumps(response_schema, indent=2)}")

    return list_output


if __name__ == "__main__":
    main()
