mod constitutive;
mod math;
mod physics;
mod units;

use ::conspire::{constitutive::ConstitutiveError, math::integrate::IntegrationError};
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
/// - [physics](conspire/physics.html) - Physics library.
/// - [constitutive](conspire/constitutive.html) - Constitutive model library.
/// - [units](conspire/units.html) - Units and quantities.
#[pymodule]
fn conspire(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_math = PyModule::new(py, "math")?;
    let submodule_physics = PyModule::new(py, "physics")?;
    let submodule_constitutive = PyModule::new(py, "constitutive")?;
    let submodule_units = PyModule::new(py, "units")?;
    submodule_math.setattr(
        "__doc__",
        "Mathematics library.\n\n - [integrate](math/integrate.html) - Integration and ODEs.\n - [special](math/special.html) - Special functions.",
    )?;
    submodule_physics.setattr(
        "__doc__",
        "Physics library.\n\n - [molecular](physics/molecular.html) - Molecular physics models.",
    )?;
    submodule_constitutive.setattr(
        "__doc__",
        "Constitutive model library.\n\n - [solid](constitutive/solid.html) - Solid constitutive models.",
    )?;
    submodule_units.setattr("__doc__", "Units and quantities.")?;
    m.add_submodule(&submodule_math)?;
    m.add_submodule(&submodule_physics)?;
    m.add_submodule(&submodule_constitutive)?;
    m.add_submodule(&submodule_units)?;
    math::register_module(py, &submodule_math)?;
    physics::register_module(py, &submodule_physics)?;
    constitutive::register_module(py, &submodule_constitutive)?;
    units::register_module(&submodule_units)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.math", submodule_math)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.physics", submodule_physics)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive", submodule_constitutive)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.units", submodule_units)
}

struct PyErrGlue {
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
            message: format!("{error:?}\x1B[A"),
        }
    }
}

impl From<IntegrationError> for PyErrGlue {
    fn from(error: IntegrationError) -> Self {
        PyErrGlue {
            message: format!("{error:?}\x1B[A"),
        }
    }
}

impl From<PyErr> for PyErrGlue {
    fn from(error: PyErr) -> Self {
        PyErrGlue {
            message: error.to_string(),
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

// macro_rules! replace_expr {
//     ($_t:tt $sub:expr) => {
//         $sub
//     };
// }
// pub(crate) use replace_expr;

// macro_rules! count_tts {
//     ($($tts:tt)*) => {0usize $(+ replace_expr!($tts 1usize))*};
// }
// pub(crate) use count_tts;
