mod almansi_hamel;

use pyo3::prelude::*;

use almansi_hamel::AlmansiHamel;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlmansiHamel>()
}
