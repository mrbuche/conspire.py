mod integrate;
mod special;

use pyo3::{PyClass, prelude::*};

use conspire::math::TensorRank2;

struct Foo<const D: usize, const I: usize, const J: usize>(TensorRank2<D, I, J>);

// impl<const D: usize, const I: usize, const J: usize> PyClass for Foo<D, I, J> {

// }

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_integrate = PyModule::new(py, "integrate")?;
    let submodule_special = PyModule::new(py, "special")?;
    submodule_integrate.setattr("__doc__", "Integration and ODEs.\n\n")?;
    submodule_special.setattr("__doc__", "Special functions.\n\n")?;
    m.add_submodule(&submodule_integrate)?;
    m.add_submodule(&submodule_special)?;
    integrate::register_module(&submodule_integrate)?;
    special::register_module(&submodule_special)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.math.integrate", submodule_integrate)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.math.special", submodule_special)
}
