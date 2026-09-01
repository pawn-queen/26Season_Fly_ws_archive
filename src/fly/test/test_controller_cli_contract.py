import ast
from pathlib import Path

import pytest


CONTROL_DIR = Path(__file__).resolve().parents[1] / 'control'


def _constructor_arg_names(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != 'OffboardControl':
            continue
        constructor = next(
            child
            for child in node.body
            if isinstance(child, ast.FunctionDef) and child.name == '__init__'
        )
        return {
            child.attr
            for child in ast.walk(constructor)
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == 'args'
            )
        }
    raise AssertionError('OffboardControl.__init__ was not found')


def _parser_destinations(tree):
    destinations = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument'
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith('--')
        ):
            continue
        explicit_dest = next(
            (
                keyword.value.value
                for keyword in node.keywords
                if (
                    keyword.arg == 'dest'
                    and isinstance(keyword.value, ast.Constant)
                )
            ),
            None,
        )
        destinations.add(
            explicit_dest or node.args[0].value[2:].replace('-', '_')
        )
    return destinations


def _literal_parser_defaults(tree):
    defaults = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument'
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        default_node = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == 'default'
            ),
            None,
        )
        if default_node is not None:
            try:
                defaults[node.args[0].value] = ast.literal_eval(default_node)
            except (TypeError, ValueError):
                continue
    return defaults


@pytest.mark.parametrize('relative_path', ['0821auto.py', 'sim/0707.py'])
def test_every_constructor_argument_is_declared_by_cli_parser(relative_path):
    """Prevent another Namespace attribute startup failure."""
    source_path = CONTROL_DIR / relative_path
    tree = ast.parse(source_path.read_text(encoding='utf-8'))

    missing = _constructor_arg_names(tree) - _parser_destinations(tree)

    assert not missing, f'{source_path.name} parser is missing: {sorted(missing)}'


@pytest.mark.parametrize(
    'option',
    [
        '--align-maxstep',
        '--alignment-altitude-threshold',
        '--first-align-maxtime',
        '--first-align-threshold',
        '--first-align-time-window',
        '--second-align-maxtime',
        '--second-align-threshold',
        '--second-align-time-window',
        '--target-anchor-hold-duration',
        '--target-confidence-window',
        '--target-observation-frame-id',
        '--target-pose-attitude-max-skew',
        '--target-pose-max-skew',
        '--target-timeout-duration',
    ],
)
def test_hardware_and_sim_alignment_defaults_match(option):
    """Keep simulation validation representative of hardware alignment."""
    defaults = []
    for relative_path in ('0821auto.py', 'sim/0707.py'):
        source_path = CONTROL_DIR / relative_path
        tree = ast.parse(source_path.read_text(encoding='utf-8'))
        defaults.append(_literal_parser_defaults(tree))

    assert option in defaults[0]
    assert option in defaults[1]
    assert defaults[0][option] == defaults[1][option]
