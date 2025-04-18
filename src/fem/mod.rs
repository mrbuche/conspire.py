mod block;

use crate::PyErrGlue;
use crate::constitutive::solid::{
    elastic::AlmansiHamel,
    hyperelastic::{ArrudaBoyce, Fung, Gent, MooneyRivlin, NeoHookean, SaintVenantKirchhoff},
};
use block::{elastic::ElasticBlock, hyperelastic::HyperelasticBlock};
use conspire::{fem::Connectivity, mechanics::Scalar};
use numpy::{PyArray2, PyArray4};
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

macro_rules! block_inner {
    ($py: ident, $model: ident, $type: ident, $block: ident, $name: ident, $connectivity: ident, $reference_nodal_coordinates: ident, $($parameter: expr),+ $(,)?) => {
        Ok(Self::$block(Py::new(
            $py,
            $block::$name(Py::new(
                $py,
                block::$type::$name::new(
                    $($model.getattr($py, stringify!($parameter))?.extract($py)?),+,
                    $connectivity,
                    $reference_nodal_coordinates,
                )
            )?)
        )?))
    }
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
            Model::AlmansiHamel(model) => block_inner!(
                py,
                model,
                elastic,
                ElasticBlock,
                AlmansiHamel,
                connectivity,
                reference_nodal_coordinates,
                bulk_modulus,
                shear_modulus,
            ),
            Model::ArrudaBoyce(model) => block_inner!(
                py,
                model,
                hyperelastic,
                HyperelasticBlock,
                ArrudaBoyce,
                connectivity,
                reference_nodal_coordinates,
                bulk_modulus,
                shear_modulus,
                number_of_links,
            ),
            Model::Fung(model) => block_inner!(
                py,
                model,
                hyperelastic,
                HyperelasticBlock,
                Fung,
                connectivity,
                reference_nodal_coordinates,
                bulk_modulus,
                shear_modulus,
                extra_modulus,
                exponent,
            ),
            Model::Gent(model) => block_inner!(
                py,
                model,
                hyperelastic,
                HyperelasticBlock,
                Gent,
                connectivity,
                reference_nodal_coordinates,
                bulk_modulus,
                shear_modulus,
                extensibility,
            ),
            Model::MooneyRivlin(model) => block_inner!(
                py,
                model,
                hyperelastic,
                HyperelasticBlock,
                MooneyRivlin,
                connectivity,
                reference_nodal_coordinates,
                bulk_modulus,
                shear_modulus,
                extra_modulus,
            ),
            Model::NeoHookean(model) => block_inner!(
                py,
                model,
                hyperelastic,
                HyperelasticBlock,
                NeoHookean,
                connectivity,
                reference_nodal_coordinates,
                bulk_modulus,
                shear_modulus,
            ),
            Model::SaintVenantKirchhoff(model) => block_inner!(
                py,
                model,
                hyperelastic,
                HyperelasticBlock,
                SaintVenantKirchhoff,
                connectivity,
                reference_nodal_coordinates,
                bulk_modulus,
                shear_modulus,
            ),
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
    /// $$
    /// \mathbf{K}_{ab} = \frac{\partial\mathbf{f}_a}{\partial\mathbf{x}_b}
    /// $$
    fn nodal_stiffnesses<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
        match self {
            Self::ElasticBlock(block) => {
                call_method!(block, py, "nodal_stiffnesses", nodal_coordinates)
            }
            Self::HyperelasticBlock(block) => {
                call_method!(block, py, "nodal_stiffnesses", nodal_coordinates)
            }
        }
    }
}
