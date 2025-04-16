mod block;

use crate::PyErrGlue;
use crate::constitutive::solid::{
    elastic::AlmansiHamel,
    hyperelastic::{ArrudaBoyce, Fung, Gent, MooneyRivlin, NeoHookean, SaintVenantKirchhoff},
};
use block::{elastic::ElasticBlock, hyperelastic::HyperelasticBlock};
use conspire::{fem::Connectivity, mechanics::Scalar};
use numpy::PyArray2;
use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Block>()
}

macro_rules! call_method {
    ($model: ident, $py: ident, $name: literal, $nodal_coordinates: ident) => {
        Ok($model
            .call_method1($py, $name, ($nodal_coordinates,))
            .unwrap()
            .extract($py)
            .unwrap())
    };
}
pub(crate) use call_method;

/// Finite element block.
#[pyclass]
enum Block {
    ElasticBlock(Py<ElasticBlock>),
    HyperelasticBlock(Py<HyperelasticBlock>),
}

#[derive(FromPyObject)]
enum Model {
    AlmansiHamel(Py<AlmansiHamel>),
    ArrudaBoyce(Py<ArrudaBoyce>),
    Gent(Py<Gent>),
    Fung(Py<Fung>),
    MooneyRivlin(Py<MooneyRivlin>),
    NeoHookean(Py<NeoHookean>),
    SaintVenantKirchhoff(Py<SaintVenantKirchhoff>),
}

#[pymethods]
impl Block {
    #[new]
    fn new(
        py: Python,
        model: Model,
        connectivity: Connectivity<4>,
        reference_nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Self, PyErr> {
        match model {
            Model::AlmansiHamel(model) => {
                let bulk_modulus: Scalar = model.getattr(py, "bulk_modulus")?.extract(py)?;
                let shear_modulus: Scalar = model.getattr(py, "shear_modulus")?.extract(py)?;
                let block = block::elastic::AlmansiHamel::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::ElasticBlock(Py::new(
                    py,
                    ElasticBlock::AlmansiHamel(Py::new(py, block)?),
                )?))
            }
            Model::ArrudaBoyce(model) => {
                let bulk_modulus: Scalar = model.getattr(py, "bulk_modulus")?.extract(py)?;
                let shear_modulus: Scalar = model.getattr(py, "shear_modulus")?.extract(py)?;
                let number_of_links: Scalar = model.getattr(py, "number_of_links")?.extract(py)?;
                let block = block::hyperelastic::ArrudaBoyce::new(
                    bulk_modulus,
                    shear_modulus,
                    number_of_links,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::HyperelasticBlock(Py::new(
                    py,
                    HyperelasticBlock::ArrudaBoyce(Py::new(py, block)?),
                )?))
            }
            Model::Fung(model) => {
                let bulk_modulus: Scalar = model.getattr(py, "bulk_modulus")?.extract(py)?;
                let shear_modulus: Scalar = model.getattr(py, "shear_modulus")?.extract(py)?;
                let extra_modulus: Scalar = model.getattr(py, "extra_modulus")?.extract(py)?;
                let exponent: Scalar = model.getattr(py, "exponent")?.extract(py)?;
                let block = block::hyperelastic::Fung::new(
                    bulk_modulus,
                    shear_modulus,
                    extra_modulus,
                    exponent,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::HyperelasticBlock(Py::new(
                    py,
                    HyperelasticBlock::Fung(Py::new(py, block)?),
                )?))
            }
            Model::Gent(model) => {
                let bulk_modulus: Scalar = model.getattr(py, "bulk_modulus")?.extract(py)?;
                let shear_modulus: Scalar = model.getattr(py, "shear_modulus")?.extract(py)?;
                let extensibility: Scalar = model.getattr(py, "extensibility")?.extract(py)?;
                let block = block::hyperelastic::Gent::new(
                    bulk_modulus,
                    shear_modulus,
                    extensibility,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::HyperelasticBlock(Py::new(
                    py,
                    HyperelasticBlock::Gent(Py::new(py, block)?),
                )?))
            }
            Model::MooneyRivlin(model) => {
                let bulk_modulus: Scalar = model.getattr(py, "bulk_modulus")?.extract(py)?;
                let shear_modulus: Scalar = model.getattr(py, "shear_modulus")?.extract(py)?;
                let extra_modulus: Scalar = model.getattr(py, "extra_modulus")?.extract(py)?;
                let block = block::hyperelastic::MooneyRivlin::new(
                    bulk_modulus,
                    shear_modulus,
                    extra_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::HyperelasticBlock(Py::new(
                    py,
                    HyperelasticBlock::MooneyRivlin(Py::new(py, block)?),
                )?))
            }
            Model::NeoHookean(model) => {
                let bulk_modulus: Scalar = model.getattr(py, "bulk_modulus")?.extract(py)?;
                let shear_modulus: Scalar = model.getattr(py, "shear_modulus")?.extract(py)?;
                let block = block::hyperelastic::NeoHookean::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::HyperelasticBlock(Py::new(
                    py,
                    HyperelasticBlock::NeoHookean(Py::new(py, block)?),
                )?))
            }
            Model::SaintVenantKirchhoff(model) => {
                let bulk_modulus: Scalar = model.getattr(py, "bulk_modulus")?.extract(py)?;
                let shear_modulus: Scalar = model.getattr(py, "shear_modulus")?.extract(py)?;
                let block = block::hyperelastic::SaintVenantKirchhoff::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::HyperelasticBlock(Py::new(
                    py,
                    HyperelasticBlock::SaintVenantKirchhoff(Py::new(py, block)?),
                )?))
            }
        }
    }
    /// $$
    /// A = \int_\Omega a\,dV
    /// $$
    fn helmholtz_free_energy(
        &self,
        py: Python,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Scalar, PyErrGlue> {
        match self {
            Self::ElasticBlock(_) => Err(PyErrGlue::new(
                "The Helmholtz free energy density is undefined for elastic constitutive models.",
            )),
            Self::HyperelasticBlock(block) => {
                call_method!(block, py, "helmholtz_free_energy", nodal_coordinates)
            }
        }
    }
    /// $$
    /// \mathbf{f}_a = \frac{\partial A}{\partial\mathbf{x}_a}
    /// $$
    fn nodal_forces<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        match self {
            Self::ElasticBlock(block) => call_method!(block, py, "nodal_forces", nodal_coordinates),
            Self::HyperelasticBlock(block) => {
                call_method!(block, py, "nodal_forces", nodal_coordinates)
            }
        }
    }
}
