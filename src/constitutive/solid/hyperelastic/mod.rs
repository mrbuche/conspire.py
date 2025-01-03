mod fung;
mod gent;

use pyo3::prelude::*;

use fung::Fung;
use gent::Gent;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Fung>()?;
    m.add_class::<Gent>()
}
