mod constitutive;

use pyo3::prelude::*;

#[pymodule]
fn conspire(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    constitutive::register_module(py, m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive", m)
}
