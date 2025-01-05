from conspire.constitutive.solid.hyperelastic import Gent
import numpy as np


zero = np.zeros((3, 3))
identity = np.eye(3)


def test_cauchy_stress_zero():
    assert (Gent(13, 3, 23).cauchy_stress(identity) == zero).all()
