mod gent;

use pyo3::prelude::*;

use gent::Gent;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Gent>()
}
