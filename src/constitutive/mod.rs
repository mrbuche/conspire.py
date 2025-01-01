mod solid;

use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    solid::register_module(py, m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive.solid", m)
}
