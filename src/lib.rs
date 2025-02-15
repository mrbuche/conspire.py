mod constitutive;
mod math;

use ::conspire::constitutive::ConstitutiveError;
use ndarray::ShapeError;
use numpy::FromVecError;
use pyo3::{exceptions::PyTypeError, prelude::*};

/// [![stable](https://img.shields.io/badge/docs-stable-blue)](https://conspire.readthedocs.io/en/stable)
/// [![latest](https://img.shields.io/badge/docs-latest-blue)](https://conspire.readthedocs.io/en/latest)
/// [![license](https://img.shields.io/github/license/mrbuche/conspire.py?color=blue)](https://github.com/mrbuche/conspire.py?tab=GPL-3.0-1-ov-file#GPL-3.0-1-ov-file)
/// [![release](https://img.shields.io/pypi/v/conspire?color=blue&label=release)](https://pypi.org/project/conspire)
///
/// The Python interface to [conspire](https://mrbuche.github.io/conspire).
/// <hr>
/// - [math](conspire/math.html) - Mathematics library.
/// - [constitutive](conspire/constitutive.html) - Constitutive model library.
#[pymodule]
fn conspire(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_math = PyModule::new(py, "math")?;
    let submodule_constitutive = PyModule::new(py, "constitutive")?;
    submodule_math.setattr(
        "__doc__",
        "Mathematics library.\n\n - [special](math/special.html) - Special functions.",
    )?;
    submodule_constitutive.setattr(
        "__doc__",
        "Constitutive model library.\n\n - [solid](constitutive/solid.html) - Solid constitutive models.",
    )?;
    m.add_submodule(&submodule_math)?;
    m.add_submodule(&submodule_constitutive)?;
    math::register_module(py, &submodule_math)?;
    constitutive::register_module(py, &submodule_constitutive)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.math", submodule_math)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive", submodule_constitutive)
}

pub struct PyErrGlue {
    message: String,
}

impl From<PyErrGlue> for PyErr {
    fn from(error: PyErrGlue) -> Self {
        PyTypeError::new_err(error.message)
    }
}

impl From<ConstitutiveError> for PyErrGlue {
    fn from(error: ConstitutiveError) -> Self {
        PyErrGlue {
            message: format!("{:?}\x1B[A", error),
        }
    }
}

impl From<ShapeError> for PyErrGlue {
    fn from(error: ShapeError) -> Self {
        PyErrGlue {
            message: error.to_string(),
        }
    }
}

impl From<FromVecError> for PyErrGlue {
    fn from(error: FromVecError) -> Self {
        PyErrGlue {
            message: error.to_string(),
        }
    }
}
