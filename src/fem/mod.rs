mod block;

use crate::PyErrGlue;
use crate::constitutive::solid::{
    elastic::AlmansiHamelEulerian,
    hyperelastic::{
        ArrudaBoyce, Fung, Gent, Hencky, MooneyRivlin, NeoHookean, SaintVenantKirchhoff,
    },
};
use crate::units::{PyEnergy, PyForce, PyForcePerLength, PyLength};
use block::{elastic::ElasticBlock, hyperelastic::HyperelasticBlock};
use conspire::{fem::solid::NodalStiffnessesSolid, math::Tensor, mechanics::Scalar};
use ndarray::{ArrayD, Ix2, IxDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

pub fn coordinates(length: &PyLength) -> PyResult<Vec<[Scalar; 3]>> {
    match length.array().view().into_dimensionality::<Ix2>() {
        Ok(array) if array.ncols() == 3 => Ok(array
            .rows()
            .into_iter()
            .map(|node| [node[0], node[1], node[2]])
            .collect()),
        _ => Err(PyValueError::new_err(
            "expected nodal coordinates with shape (nodes, 3)",
        )),
    }
}

pub fn stiffnesses(nodes: usize, stiffnesses: &NodalStiffnessesSolid<3>) -> PyForcePerLength {
    let mut array = ArrayD::<Scalar>::zeros(IxDyn(&[nodes, nodes, 3, 3]));
    stiffnesses.iter().enumerate().for_each(|(a, row)| {
        row.entries().for_each(|(b, entry)| {
            (0..3).for_each(|i| {
                (0..3).for_each(|j| array[IxDyn(&[a, b, i, j])] = entry[i][j].value())
            })
        })
    });
    PyForcePerLength::from_array(array)
}

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
    AlmansiHamelEulerian(Py<AlmansiHamelEulerian>),
    ArrudaBoyce(Py<ArrudaBoyce>),
    Gent(Py<Gent>),
    Hencky(Py<Hencky>),
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
                )?
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
        connectivity: Vec<[usize; 4]>,
        reference_nodal_coordinates: PyLength,
    ) -> Result<Self, PyErr> {
        match model {
            Model::AlmansiHamelEulerian(model) => block_inner!(
                py,
                model,
                elastic,
                ElasticBlock,
                AlmansiHamelEulerian,
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
            Model::Hencky(model) => block_inner!(
                py,
                model,
                hyperelastic,
                HyperelasticBlock,
                Hencky,
                connectivity,
                reference_nodal_coordinates,
                bulk_modulus,
                shear_modulus,
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
        nodal_coordinates: PyLength,
    ) -> Result<PyEnergy, PyErrGlue> {
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
    fn nodal_forces(&self, py: Python, nodal_coordinates: PyLength) -> Result<PyForce, PyErrGlue> {
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
    fn nodal_stiffnesses(
        &self,
        py: Python,
        nodal_coordinates: PyLength,
    ) -> Result<PyForcePerLength, PyErrGlue> {
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
