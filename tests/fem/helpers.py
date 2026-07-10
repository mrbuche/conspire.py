import numpy as np


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
