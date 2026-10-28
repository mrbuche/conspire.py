mod block;

use crate::PyErrGlue;
use crate::units::{PyEnergy, PyForce, PyForcePerLength, PyLength};
use block::{
    elastic::{ElasticBlock, ElasticModel},
    hyperelastic::{HyperelasticBlock, HyperelasticModel},
};
use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Block>()
}

macro_rules! call_method {
    ($model: ident, $py: ident, $name: literal, $nodal_coordinates: ident) => {
        Ok($model
            .call_method1($py, $name, ($nodal_coordinates,))?
            .extract($py)
            .map_err(PyErr::from)?)
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
enum Model<'py> {
    Elastic(ElasticModel<'py>),
    Hyperelastic(HyperelasticModel<'py>),
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
            Model::Elastic(model) => Ok(Self::ElasticBlock(Py::new(
                py,
                ElasticBlock::new(py, model, connectivity, reference_nodal_coordinates)?,
            )?)),
            Model::Hyperelastic(model) => Ok(Self::HyperelasticBlock(Py::new(
                py,
                HyperelasticBlock::new(py, model, connectivity, reference_nodal_coordinates)?,
            )?)),
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
