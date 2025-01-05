from conspire.constitutive.solid.hyperelastic import NeoHookean
import numpy as np


zero = np.zeros((3, 3))
identity = np.eye(3)


def test_cauchy_stress_zero():
    assert (NeoHookean(13, 3).cauchy_stress(identity) == zero).all()
