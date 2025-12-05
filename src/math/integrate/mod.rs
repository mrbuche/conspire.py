pub mod explicit;

use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<explicit::bogacki_shampine::BogackiShampine>()?;
    m.add_class::<explicit::dormand_prince::DormandPrince>()?;
    m.add_class::<explicit::verner_8::Verner8>()?;
    m.add_class::<explicit::verner_9::Verner9>()
}
