import itertools
import unittest.mock as mock

import pytest

import ecole


@pytest.mark.parametrize("done", (True, False))
@pytest.mark.parametrize("cste", (True, 1, "hello"))
def test_ConstantFunction(model, done, cste):
    """Always return the same constant."""
    cste_func = ecole.data.ConstantFunction(cste)
    cste_func.before_reset(model)
    pytest.helpers.advance_to_stage(model, ecole.scip.Stage.Solving)
    data = cste_func.extract(model, done)
    assert data == cste


@pytest.mark.parametrize("done", (True, False))
def test_VectorFunction(model, done):
    """Dispach calls and pack the result in a list."""
    data_func1, data_func2 = mock.MagicMock(), mock.MagicMock()
    tuple_data_func = ecole.data.VectorFunction(data_func1, data_func2)

    tuple_data_func.before_reset(model)
    data_func1.before_reset.assert_called_once_with(model)
    data_func2.before_reset.assert_called_once_with(model)

    pytest.helpers.advance_to_stage(model, ecole.scip.Stage.Solving)
    data_func1.extract.return_value = "something"
    data_func2.extract.return_value = "else"
    data = tuple_data_func.extract(model, done)
    assert data == ["something", "else"]


@pytest.mark.parametrize("done", (True, False))
def test_MapFunction(model, done):
    """Dispach calls and pack the result in a dict."""
    data_func1, data_func2 = mock.MagicMock(), mock.MagicMock()
    dict_data_func = ecole.data.MapFunction(name1=data_func1, name2=data_func2)

    dict_data_func.before_reset(model)
    data_func1.before_reset.assert_called_once_with(model)
    data_func2.before_reset.assert_called_once_with(model)

    pytest.helpers.advance_to_stage(model, ecole.scip.Stage.Solving)
    data_func1.extract.return_value = "something"
    data_func2.extract.return_value = "else"
    data = dict_data_func.extract(model, done)
    assert data == {"name1": "something", "name2": "else"}


@pytest.mark.parametrize("done", (True, False))
@pytest.mark.parametrize("wall", (True, False))
def test_TimedFunction(model, done, wall):
    """Time a given data function."""
    data_func = mock.MagicMock()
    time_data_func = ecole.data.TimedFunction(data_func, wall=wall)

    time_data_func.before_reset(model)
    data_func.before_reset.assert_called_once_with(model)

    pytest.helpers.advance_to_stage(model, ecole.scip.Stage.Solving)
    time = time_data_func.extract(model, done)
    assert time > 0


def test_parse_None():
    """None is parsed as NoneFunction."""
    assert isinstance(ecole.data.parse(None, mock.MagicMock()), ecole.data.NoneFunction)


def test_parse_default():
    """Default return default."""
    default_func = mock.MagicMock()
    assert ecole.data.parse(ecole.Default, default_func) == default_func


def test_parse_default_str():
    """Default return default."""
    default_func = mock.MagicMock()
    assert ecole.data.parse("Default", default_func) == default_func


def test_parse_self_reference():
    """Default can not be used in the default function."""
    with pytest.raises(ValueError):
        ecole.data.parse(ecole.Default, ecole.Default)


def test_parse_number():
    """Number return ConstantFunction."""
    assert isinstance(ecole.data.parse(1, mock.MagicMock()), ecole.data.ConstantFunction)


def test_parse_tuple():
    """Tuple is parsed as VectorFunction."""
    aggregate = (mock.MagicMock(), mock.MagicMock())
    assert isinstance(ecole.data.parse(aggregate, mock.MagicMock()), ecole.data.VectorFunction)


def test_parse_dict():
    """Dict is parsed as MapFunction."""
    aggregate = {"name1": mock.MagicMock(), "name2": mock.MagicMock()}
    assert isinstance(ecole.data.parse(aggregate, mock.MagicMock()), ecole.data.MapFunction)


def test_parse_recursive(model):
    """Parsing is recursive."""
    aggregate = {
        "name1": mock.MagicMock(),
        "name2": (mock.MagicMock(), None, 1),
        "name3": "Default",
    }
    default_func = mock.MagicMock()
    default_func.extract.return_value == mock.MagicMock()
    func = ecole.data.parse(aggregate, default_func)
    # Using the extract method to inspect the recusive parsing since Vector, Map, Constant functions are private.
    data = func.extract(model, False)
    assert isinstance(data, dict)
    assert isinstance(data["name2"], list)
    assert data["name2"][1] is None
    assert data["name2"][2] == 1
    assert data["name3"] == default_func.extract.return_value


def test_parse_recursive_default(model):
    """Default function is parsed as well."""
    aggregate = {
        "name1": mock.MagicMock(),
        "name2": (mock.MagicMock(), None, 1),
    }
    func = ecole.data.parse("Default", aggregate)
    # Using the extract method to inspect the recusive parsing since Vector, Map, Constant functions are private.
    data = func.extract(model, False)
    assert isinstance(data, dict)
    assert isinstance(data["name2"], list)
    assert data["name2"][1] is None
    assert data["name2"][2] == 1
