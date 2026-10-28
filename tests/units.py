from conspire.constitutive.solid.hyperelastic import NeoHookean
from conspire.units import Energy, Force, ForcePerLength, Length, Stress
import numpy as np
import pytest


def test_scale_round_trip_scalar():
    assert Stress.megapascals(2.0).in_pascals() == 2e6
    assert Stress.megapascals(2.0).in_kilopascals() == 2e3
    assert Length.millimeters(5.0).in_meters() == 5e-3
    assert Force.kilonewtons(1.0).in_newtons() == 1e3
    assert Energy.kilojoules(1.0).in_joules() == 1e3
    assert ForcePerLength.newtons_per_meter(7.0).in_newtons_per_meter() == 7.0


def test_scale_crosses_units_consistently():
    assert Length.inches(1.0).in_millimeters() == pytest.approx(25.4)
    assert Stress.psi(1.0).in_pascals() == pytest.approx(6894.757293168361)


def test_array_round_trip_preserves_shape_and_values():
    coordinates = np.arange(12.0).reshape(4, 3)
    length = Length.millimeters(coordinates)
    assert np.allclose(length.in_meters(), coordinates * 1e-3)
    assert length.in_millimeters().shape == (4, 3)


def test_scalar_reads_as_float_and_array_as_ndarray():
    assert isinstance(Stress.pascals(1.0).in_pascals(), float)
    assert isinstance(Stress.pascals(np.ones(3)).in_pascals(), np.ndarray)


def test_repr_names_the_constructor():
    assert repr(Stress.pascals(13.0)) == "Stress.pascals(13)"


def test_parameter_must_be_a_single_quantity():
    with pytest.raises(ValueError, match="expected a single Stress"):
        NeoHookean(Stress.pascals(np.ones(3)), Stress.pascals(1.0))


def test_model_parameter_reads_back_in_any_scale():
    model = NeoHookean(Stress.megapascals(13.0), Stress.megapascals(3.0))
    assert model.bulk_modulus.in_pascals() == 13e6
    assert model.shear_modulus.in_kilopascals() == 3e3


def test_stress_output_is_a_quantity():
    model = NeoHookean(Stress.pascals(13.0), Stress.pascals(3.0))
    stress = model.cauchy_stress(np.eye(3))
    assert isinstance(stress, Stress)
    assert np.all(stress.in_pascals() == 0.0)
    assert np.all(stress.in_megapascals() == 0.0)
