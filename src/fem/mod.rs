mod solid;

use crate::PyErrGlue;
use crate::units::{PyEnergy, PyForce, PyForcePerLength, PyLength};
use pyo3::prelude::*;
use solid::{
    elastic::{ElasticConstitutive, ElasticModel},
    hyperelastic::{HyperelasticConstitutive, HyperelasticModel},
};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()
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

/// Finite element model.
#[pyclass]
enum Model {
    Elastic(Py<ElasticModel>),
    Hyperelastic(Py<HyperelasticModel>),
}

#[derive(FromPyObject)]
enum Constitutive<'py> {
    Elastic(ElasticConstitutive<'py>),
    Hyperelastic(HyperelasticConstitutive<'py>),
}

#[pymethods]
impl Model {
    #[new]
    fn new(
        py: Python,
        constitutive_model: Constitutive,
        connectivity: Vec<[usize; 4]>,
        reference_nodal_coordinates: PyLength,
    ) -> Result<Self, PyErr> {
        match constitutive_model {
            Constitutive::Elastic(model) => Ok(Self::Elastic(Py::new(
                py,
                ElasticModel::new(py, model, connectivity, reference_nodal_coordinates)?,
            )?)),
            Constitutive::Hyperelastic(model) => Ok(Self::Hyperelastic(Py::new(
                py,
                HyperelasticModel::new(py, model, connectivity, reference_nodal_coordinates)?,
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
            Self::Elastic(_) => Err(PyErrGlue::new(
                "The Helmholtz free energy density is undefined for elastic constitutive models.",
            )),
            Self::Hyperelastic(model) => {
                call_method!(model, py, "helmholtz_free_energy", nodal_coordinates)
            }
        }
    }
    /// $$
    /// \mathbf{f}_a = \frac{\partial A}{\partial\mathbf{x}_a}
    /// $$
    fn nodal_forces(&self, py: Python, nodal_coordinates: PyLength) -> Result<PyForce, PyErrGlue> {
        match self {
            Self::Elastic(model) => call_method!(model, py, "nodal_forces", nodal_coordinates),
            Self::Hyperelastic(model) => call_method!(model, py, "nodal_forces", nodal_coordinates),
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
            Self::Elastic(model) => call_method!(model, py, "nodal_stiffnesses", nodal_coordinates),
            Self::Hyperelastic(model) => {
                call_method!(model, py, "nodal_stiffnesses", nodal_coordinates)
            }
        }
    }
}
