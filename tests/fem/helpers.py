from conspire.constitutive.solid import elastic, hyperelastic
from conspire.fem import Block
from conspire.units import Energy, Force, ForcePerLength, Length, Stress
import numpy as np

_BS = {"bulk_modulus": 13, "shear_modulus": 3}


def _models(module, cases):
    return [
        (name, getattr(module, name), params, shear) for name, params, shear in cases
    ]


ELASTIC_MODELS = _models(
    elastic,
    [
        (name, {**_BS, **extra}, 3)
        for name, extra in [
            ("AlmansiHamelEulerian", {}),
            ("AlmansiHamelLagrangian", {}),
            ("BazantItskovEulerian", {"exponent": 1.3}),
            ("BazantItskovLagrangian", {"exponent": 1.3}),
            ("Hencky", {}),
            ("SaintVenantKirchhoff", {}),
            ("SethHillEulerian", {"exponent": 1.3}),
            ("SethHillLagrangian", {"exponent": 1.3}),
        ]
    ],
)

HYPERELASTIC_MODELS = _models(
    hyperelastic,
    [
        ("ArrudaBoyce", {**_BS, "number_of_links": 8}, 3),
        ("BlatzKo", {**_BS, "mixing_parameter": 0.35}, 3),
        (
            "Carroll",
            {
                "bulk_modulus": 13,
                "linear_modulus": 1.2,
                "quartic_modulus": 1e-4,
                "second_invariant_modulus": 0.9,
            },
            None,
        ),
        ("Fung", {**_BS, "extra_modulus": 1, "exponent": 1}, 3),
        ("Gent", {**_BS, "extensibility": 23}, 3),
        ("Hencky", _BS, 3),
        ("Isihara", {**_BS, "extra_modulus": 1.2, "quadratic_modulus": 1e-2}, 3),
        ("MooneyRivlin", {**_BS, "extra_modulus": 1}, 3),
        ("NeoHookean", _BS, 3),
        (
            "Ogden",
            {
                "bulk_modulus": 13,
                "shear_moduli": [2.3, 1.1e-2, -2e-3],
                "exponents": [1.3, 5.0, -2.0],
            },
            None,
        ),
        ("SaintVenantKirchhoff", _BS, 3),
        (
            "Yeoh",
            {"bulk_modulus": 13, "shear_moduli": [3, -1, 3e-1, -1e-3, 1e-5]},
            3,
        ),
    ],
)


def _convert(key, value):
    if key.endswith(("modulus", "moduli")):
        return Stress.pascals(np.asarray(value, dtype=float))
    return value


class InPascals:
    """A model that takes stresses in pascals and returns them in pascals."""

    def __init__(self, model):
        self.raw = model

    def __getattr__(self, name):
        attribute = getattr(self.raw, name)
        if callable(attribute):
            return lambda *args: self._unwrap(attribute(*args))
        return self._unwrap(attribute)

    @staticmethod
    def _unwrap(value):
        return value.in_pascals() if isinstance(value, Stress) else value


class InSI:
    """A block that takes lengths in meters and returns SI energies and forces."""

    def __init__(self, block):
        self._block = block

    def __getattr__(self, name):
        method = getattr(self._block, name)
        return lambda coordinates: self._unwrap(method(Length.meters(coordinates)))

    @staticmethod
    def _unwrap(value):
        if isinstance(value, Energy):
            return value.in_joules()
        if isinstance(value, Force):
            return value.in_newtons()
        if isinstance(value, ForcePerLength):
            return value.in_newtons_per_meter()
        return value


def build(cls, params):
    return InPascals(cls(*[_convert(key, value) for key, value in params.items()]))


def make_block(model, connectivity, reference_coordinates):
    return InSI(Block(model.raw, connectivity, Length.meters(reference_coordinates)))


def assert_nodal_forces_zero(block, reference_coordinates):
    assert np.all(block.nodal_forces(reference_coordinates) == 0.0)


def assert_nodal_forces_finite_difference(
    block, reference_coordinates, deformed_coordinates, epsilon
):
    forces = block.nodal_forces(deformed_coordinates)
    for a in range(len(reference_coordinates)):
        for i in range(3):
            deformed_coordinates[a, i] += epsilon / 2
            d_helmholtz = block.helmholtz_free_energy(deformed_coordinates)
            deformed_coordinates[a, i] -= epsilon
            d_helmholtz -= block.helmholtz_free_energy(deformed_coordinates)
            assert np.abs(forces[a, i] - d_helmholtz / epsilon) < epsilon
            deformed_coordinates[a, i] += epsilon / 2


def assert_nodal_stiffnesses_finite_difference(
    block, reference_coordinates, deformed_coordinates, epsilon
):
    tan = block.nodal_stiffnesses(deformed_coordinates)
    for a in range(len(reference_coordinates)):
        for b in range(len(reference_coordinates)):
            for i in range(3):
                for j in range(3):
                    deformed_coordinates[b, j] += epsilon / 2
                    d_force = block.nodal_forces(deformed_coordinates)[a, i]
                    deformed_coordinates[b, j] -= epsilon
                    d_force -= block.nodal_forces(deformed_coordinates)[a, i]
                    assert np.abs(tan[a, b, i, j] - d_force / epsilon) < epsilon
                    deformed_coordinates[b, j] += epsilon / 2
