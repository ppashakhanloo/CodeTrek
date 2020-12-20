import ast
import sys

input_code_file = sys.argv[1]
output_edges_file = sys.argv[2]


def node_str(node):
    if isinstance(node, ast.AST):
        fields = (
            f"{name}={node_str(val)}"
            for name, val in ast.iter_fields(node)
            if name not in ("left", "right")
        )
        rv = f"{node.__class__.__name__}({', '.join(fields)})"
        return rv
    else:
        return repr(node)


def node_kind(node):
    if isinstance(node, ast.AST):
        return node.__class__.__name__


def nodes(node):
    """
    Yields all the nodes in the ast sub-tree rooted at node in depth-first order.
    """
    yield node
    for _, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    yield from nodes(item)
        elif isinstance(value, ast.AST):
            yield from nodes(value)


def edges(node):
    """
    Yields all the edges in the ast sub-tree rooted at node.
    """
    for _, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    yield (node, item)
                    yield from edges(item)
        elif isinstance(value, ast.AST):
            yield (node, value)
            yield from edges(value)


with open(input_code_file, "r") as infile:
    root_node = ast.parse(infile.read())

    node_ids = {node: index for index, node in enumerate(nodes(root_node))}

    with open(output_edges_file, "w") as outfile:
        for src, dest in edges(root_node):
            outfile.write(f"{node_ids[src]} {node_ids[dest]}\n")
