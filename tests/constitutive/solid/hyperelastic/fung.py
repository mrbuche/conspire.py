from conspire.constitutive.solid.hyperelastic import Fung
import numpy as np


zero = np.zeros((3, 3))
identity = np.eye(3)


def test_cauchy_stress_zero():
    assert (Fung(13, 3, 1.2, 1.1).cauchy_stress(identity) == zero).all()
