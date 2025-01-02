mod constitutive;

use ::conspire::constitutive::ConstitutiveError;
use pyo3::{exceptions::PyTypeError, prelude::*};

#[pymodule]
fn conspire(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    constitutive::register_module(py, m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive", m)
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
