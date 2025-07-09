pub mod elastic;
pub mod hyperelastic;

use conspire::constitutive::solid::{
    elastic::doc::DOC as ELASTIC, hyperelastic::doc::DOC as HYPERELASTIC,
};
use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_elastic = PyModule::new(py, "elastic")?;
    let submodule_hyperelastic = PyModule::new(py, "hyperelastic")?;
    submodule_elastic.setattr(
        "__doc__",
        ELASTIC.replace("```math", "$$").replace("```", "$$"),
    )?;
    submodule_hyperelastic.setattr(
        "__doc__",
        HYPERELASTIC.replace("```math", "$$").replace("```", "$$"),
    )?;
    m.add_submodule(&submodule_elastic)?;
    m.add_submodule(&submodule_hyperelastic)?;
    elastic::register_module(&submodule_elastic)?;
    hyperelastic::register_module(&submodule_hyperelastic)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive.solid.elastic", submodule_elastic)?;
    py.import("sys")?.getattr("modules")?.set_item(
        "conspire.constitutive.solid.hyperelastic",
        submodule_hyperelastic,
    )
}
