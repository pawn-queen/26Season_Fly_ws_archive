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


def _literal_parser_choices(tree):
    choices = {}
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
        for keyword in node.keywords:
            if keyword.arg == 'choices':
                choices[node.args[0].value] = ast.literal_eval(keyword.value)
    return choices


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
        '--first-align-threshold',
        '--first-align-time-window',
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


@pytest.mark.parametrize(
    'relative_path, first_timeout, second_timeout',
    [('0821auto.py', 12.0, 8.0), ('sim/0707.py', 15.0, 10.0)],
)
def test_alignment_timeout_defaults_allow_environment_tuning(
    relative_path, first_timeout, second_timeout
):
    """Simulation keeps its longer alignment budget while sharing the flow."""
    tree = ast.parse((CONTROL_DIR / relative_path).read_text(encoding='utf-8'))
    defaults = _literal_parser_defaults(tree)

    assert defaults['--first-align-maxtime'] == first_timeout
    assert defaults['--second-align-maxtime'] == second_timeout


@pytest.mark.parametrize('relative_path', ['0821auto.py', 'sim/0707.py'])
@pytest.mark.parametrize(
    'option, expected_choices, expected_default',
    [
        ('--enable-smooth-transit', ('true', 'false'), 'false'),
        ('--target-anchor-mode', ('max-confidence', 'top25'), 'max-confidence'),
    ],
)
def test_transit_and_anchor_options_share_cli_contract(
    relative_path, option, expected_choices, expected_default
):
    tree = ast.parse((CONTROL_DIR / relative_path).read_text(encoding='utf-8'))

    assert _literal_parser_choices(tree)[option] == expected_choices
    assert _literal_parser_defaults(tree)[option] == expected_default
