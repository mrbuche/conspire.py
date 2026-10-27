from conspire.fem import Block
from conspire.units import Energy, Force, ForcePerLength, Length, Stress
import numpy as np


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


def build(cls, bulk_modulus, shear_modulus, extra):
    parameters = [
        Stress.pascals(value) if key.endswith("modulus") else value
        for key, value in extra.items()
    ]
    return InPascals(
        cls(Stress.pascals(bulk_modulus), Stress.pascals(shear_modulus), *parameters)
    )


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
