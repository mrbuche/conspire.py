mod single_chain;

use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_single_chain = PyModule::new(py, "single_chain")?;
    m.add_submodule(&submodule_single_chain)?;
    single_chain::register_module(py, &submodule_single_chain)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.physics.single_chain", submodule_single_chain)
}
