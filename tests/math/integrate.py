from conspire.math.integrate import (
    BogackiShampine,
    DormandPrince,
    Verner8,
    Verner9,
)
from pytest import raises
import numpy as np
import pytest

ABS_TOL = 1e-8
DEFAULT_ABS_TOL = 1e-12
DEFAULT_REL_TOL = 1e-12
DEFAULT_DT_BETA = 0.9
DEFAULT_DT_CUT = 0.5
DEFAULT_DT_MIN = 1e-12

# dt_expn defaults to each method's order.
INTEGRATORS = [
    ("BogackiShampine", BogackiShampine, 3.0),
    ("DormandPrince", DormandPrince, 5.0),
    ("Verner8", Verner8, 8.0),
    ("Verner9", Verner9, 9.0),
]


@pytest.fixture(params=INTEGRATORS, ids=[name for name, _, _ in INTEGRATORS])
def integrator_case(request):
    return request.param


@pytest.fixture
def integrator(integrator_case):
    _, cls, _ = integrator_case
    return cls()


def test_default_tolerances(integrator, integrator_case):
    _, _, dt_expn = integrator_case
    assert integrator.abs_tol == DEFAULT_ABS_TOL
    assert integrator.rel_tol == DEFAULT_REL_TOL
    assert integrator.dt_beta == DEFAULT_DT_BETA
    assert integrator.dt_expn == dt_expn
    assert integrator.dt_cut == DEFAULT_DT_CUT
    assert integrator.dt_min == DEFAULT_DT_MIN


def test_kwargs_override(integrator_case):
    _, cls, dt_expn = integrator_case
    integrator = cls(abs_tol=1e-6, rel_tol=1e-7, dt_beta=0.8)
    assert integrator.abs_tol == 1e-6
    assert integrator.rel_tol == 1e-7
    assert integrator.dt_beta == 0.8
    assert integrator.dt_expn == dt_expn


def test_exponential_decay(integrator):
    time, solution, function = integrator.integrate(
        lambda _, y: [-y[0]], [0.0, 0.8], [1.0]
    )
    for t, y, dydt in zip(time, solution, function):
        assert np.abs(y[0] - np.exp(-t)) < ABS_TOL
        assert np.abs(dydt[0] + y[0]) < ABS_TOL


def test_dydt_eq_2yt(integrator):
    time, solution, _ = integrator.integrate(
        lambda t, y: [2.0 * y[0] * t], [0.0, 1.0], [1.0]
    )
    for t, y in zip(time, solution):
        assert np.abs(y[0] - np.exp(t**2)) < ABS_TOL


def test_dydt_eq_cos_t(integrator):
    time, solution, function = integrator.integrate(
        lambda t, _: [np.cos(t)], [0.0, 1.0], [0.0]
    )
    for t, y, dydt in zip(time, solution, function):
        assert np.abs(y[0] - np.sin(t)) < ABS_TOL
        assert np.abs(dydt[0] - np.cos(t)) < ABS_TOL


def test_vector_ode(integrator):
    time, solution, _ = integrator.integrate(
        lambda _, y: list(y), [0.0, 1.0], [1.0, 1.0, 1.0]
    )
    for t, y in zip(time, solution):
        for y_n in y:
            assert np.abs(y_n - np.exp(t)) < ABS_TOL


def test_eval_times(integrator):
    times = list(np.linspace(0.0, 1.0, 50))
    time, solution, _ = integrator.integrate(lambda t, _: [np.cos(t)], times, [0.0])
    assert len(time) == len(times)
    for t_requested, t, y in zip(times, time, solution):
        assert t == t_requested
        assert np.abs(y[0] - np.sin(t)) < ABS_TOL


def test_time_length_less_than_two(integrator):
    with raises(TypeError):
        integrator.integrate(lambda _, y: [-y[0]], [0.0], [1.0])


def test_initial_time_not_less_than_final_time(integrator):
    with raises(TypeError):
        integrator.integrate(lambda _, y: [-y[0]], [1.0, 0.0], [1.0])
