mod elastic;
mod hyperelastic;

use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    elastic::register_module(m)?;
    hyperelastic::register_module(m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive.solid.elastic", m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive.solid.hyperelastic", m)
}
