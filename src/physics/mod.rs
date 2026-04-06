mod molecular;

use conspire::physics::{BOLTZMANN_CONSTANT, ROOM_TEMPERATURE};
use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_molecular = PyModule::new(py, "molecular")?;
    m.add_submodule(&submodule_molecular)?;
    molecular::register_module(py, &submodule_molecular)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.physics.molecular", submodule_molecular)?;
    m.add("BOLTZMANN_CONSTANT", BOLTZMANN_CONSTANT)?;
    m.add("ROOM_TEMPERATURE", ROOM_TEMPERATURE)
}
