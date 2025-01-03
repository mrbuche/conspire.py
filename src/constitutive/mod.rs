mod solid;

use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_solid = PyModule::new(py, "solid")?;
    submodule_solid.setattr(
        "__doc__",
        "Solid constitutive models.\n\n - [elastic](solid/elastic.html) - Elastic constitutive models.\n - [hyperelastic](solid/hyperelastic.html) - Hyperelastic constitutive models.",
    )?;
    m.add_submodule(&submodule_solid)?;
    solid::register_module(py, &submodule_solid)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive.solid", submodule_solid)
}
