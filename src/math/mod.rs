mod special;

use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_special = PyModule::new(py, "special")?;
    submodule_special.setattr("__doc__", "Special functions.\n\n")?;
    m.add_submodule(&submodule_special)?;
    special::register_module(&submodule_special)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.math.special", submodule_special)
}
