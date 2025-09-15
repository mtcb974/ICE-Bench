import ast
import re
from typing import Optional, List
import astunparse
import os
import textwrap

def parse_pytest_selector(selector: str) -> tuple:
    """Parse pytest selector, return (file path, class name, function name)"""
    parts = selector.split('::')
    if len(parts) == 2:
        return (parts[0], None, parts[1])
    elif len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    else:
        raise ValueError(f"Invalid pytest selector: {selector}")

def add_test_function_to_file(
    file_path: str,
    selector: str,
    new_test_code: str,
    insert_after: Optional[str] = None
) -> None:
    """
    Add new test functions to the test file
    
    :param file_path: Test file path
    :param selector: pytest selector, e.g. "path/to/test.py::TestClass::test_method"
    :param new_test_code: New test function's complete code string, can contain multiple test functions
    :param insert_after: Optional, specify which existing function to insert the new function after
    """
    # Parse selector
    new_test_code = textwrap.dedent(new_test_code)
    _, class_name, _ = parse_pytest_selector(selector)
    
    # if file_path != file_path_from_selector:
        # raise ValueError(f"Selector file path({file_path_from_selector}) does not match target file path({file_path})")
    
    # Read original file
    with open(file_path, 'r', encoding='utf-8') as f:
        original_source = f.read()
    
    tree = ast.parse(original_source)
    
    # Parse new test functions to AST node list
    new_test_nodes = ast.parse(new_test_code).body
    for node in new_test_nodes:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError("New test code must contain only function definitions")
    
    # Locate insertion position
    if class_name:
        # Add functions in class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                insert_position = _find_insert_position(node.body, insert_after)
                # Insert all new functions in order
                for i, new_node in enumerate(new_test_nodes):
                    node.body.insert(insert_position + i, new_node)
                break
        else:
            raise ValueError(f"Class {class_name} not found in file {file_path}")
    else:
        # Add functions at the top of the file
        insert_position = _find_insert_position(tree.body, insert_after)
        # Insert all new functions in order
        for i, new_node in enumerate(new_test_nodes):
            tree.body.insert(insert_position + i, new_node)
    
    # Generate new code and preserve original format (as much as possible)
    new_source = _unparse_with_preservation(tree, original_source)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_source)

def _find_insert_position(body_nodes, insert_after: Optional[str] = None) -> int:
    """Determine the position where the new node should be inserted"""
    if not insert_after:
        # If no insertion position is specified, find the last test function and insert after it
        last_test_pos = 0
        for i, node in enumerate(body_nodes):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith('test_'):
                    last_test_pos = i + 1
        return last_test_pos if last_test_pos > 0 else len(body_nodes)
    else:
        # Find the position after the specified function
        for i, node in enumerate(body_nodes):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == insert_after:
                return i + 1
        raise ValueError(f"Function {insert_after} not found for insertion reference")

def _unparse_with_preservation(tree, original_source: str) -> str:
    """
    Use astunparse to generate code, but try to preserve original format
    This can be extended to more complex format preservation logic
    """
    try:
        import astor  # More advanced code generator, can try to install: pip install astor
        return astor.to_source(tree)
    except ImportError:
        return astunparse.unparse(tree)

if __name__ == "__main__":
    add_test_function_to_file(
        file_path="test_example.py",
        selector="test_example.py::test_chords::setUp",
        new_test_code="""def test_new_function1():
    \"\"\"New added test function 1\"\"\"
    assert 1 + 1 == 2

def test_new_function2():
    \"\"\"New added test function 2\"\"\"
    assert 2 + 2 == 4
""",
        insert_after="setUp"  # Optional, specify which existing function to insert the new function after
    )
    
    # 示例2：在文件顶层添加多个新测试
    add_test_function_to_file(
        file_path="test_example.py",
        selector="test_example.py::load_tests",
        new_test_code="""def test_new_standalone1():
    \"\"\"Standalone test function 1\"\"\"
    assert True is not False

def test_new_standalone2():
    \"\"\"Standalone test function 2\"\"\"
    assert False is not True
"""
    )