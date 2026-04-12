mod single_chain;

use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_single_chain = PyModule::new(py, "single_chain")?;
    submodule_single_chain.setattr("__doc__", "Single-chain models of polymer physics.")?;
    m.add_submodule(&submodule_single_chain)?;
    single_chain::register_module(&submodule_single_chain)?;
    py.import("sys")?.getattr("modules")?.set_item(
        "conspire.physics.molecular.single_chain",
        submodule_single_chain,
    )
}
