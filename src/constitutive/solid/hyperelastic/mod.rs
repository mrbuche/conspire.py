mod arruda_boyce;
mod fung;
mod gent;
mod mooney_rivlin;
mod neo_hookean;
mod saint_venant_kirchhoff;
// mod yeoh;

use pyo3::prelude::*;

use arruda_boyce::ArrudaBoyce;
use fung::Fung;
use gent::Gent;
use mooney_rivlin::MooneyRivlin;
use neo_hookean::NeoHookean;
use saint_venant_kirchhoff::SaintVenantKirchhoff;
// use yeoh::Yeoh;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrudaBoyce>()?;
    m.add_class::<Fung>()?;
    m.add_class::<Gent>()?;
    m.add_class::<MooneyRivlin>()?;
    m.add_class::<NeoHookean>()?;
    m.add_class::<SaintVenantKirchhoff>()
    // m.add_class::<Yeoh>()
}
