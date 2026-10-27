use crate::{
    PyErrGlue,
    constitutive::solid::elastic as constitutive,
    fem::call_method,
    units::{PyForce, PyForcePerLength, PyLength},
};
use conspire::fem::{
    NodalCoordinates, NodalReferenceCoordinates,
    block::{Block, element::linear::Tetrahedron as LinearTetrahedron},
    solid::elastic::ElasticElements,
};
use pyo3::prelude::*;

pub const G: usize = 1;
pub const M: usize = 3;
pub const N: usize = 4;
pub const P: usize = N;

#[pyclass]
pub enum ElasticBlock {
    AlmansiHamelEulerian(Py<AlmansiHamelEulerian>),
}

#[derive(FromPyObject)]
enum ElasticModel<'py> {
    AlmansiHamelEulerian(Bound<'py, constitutive::AlmansiHamelEulerian>),
}

#[pymethods]
impl ElasticBlock {
    #[new]
    fn new(
        py: Python,
        model: ElasticModel,
        connectivity: Vec<[usize; N]>,
        reference_nodal_coordinates: PyLength,
    ) -> Result<Self, PyErr> {
        match model {
            ElasticModel::AlmansiHamelEulerian(model) => {
                let block = AlmansiHamelEulerian::new(
                    model.getattr("bulk_modulus")?.extract()?,
                    model.getattr("shear_modulus")?.extract()?,
                    connectivity,
                    reference_nodal_coordinates,
                )?;
                Ok(Self::AlmansiHamelEulerian(Py::new(py, block)?))
            }
        }
    }
    fn nodal_forces(&self, py: Python, nodal_coordinates: PyLength) -> Result<PyForce, PyErrGlue> {
        match self {
            Self::AlmansiHamelEulerian(model) => {
                call_method!(model, py, "nodal_forces", nodal_coordinates)
            }
        }
    }
    fn nodal_stiffnesses(
        &self,
        py: Python,
        nodal_coordinates: PyLength,
    ) -> Result<PyForcePerLength, PyErrGlue> {
        match self {
            Self::AlmansiHamelEulerian(model) => {
                call_method!(model, py, "nodal_stiffnesses", nodal_coordinates)
            }
        }
    }
}

#[pyclass(unsendable)]
pub struct AlmansiHamelEulerian {
    block: Block<
        conspire::constitutive::solid::elastic::AlmansiHamelEulerian,
        LinearTetrahedron<G>,
        G,
        M,
        N,
        P,
    >,
}

#[pymethods]
impl AlmansiHamelEulerian {
    #[new]
    pub fn new(
        bulk_modulus: crate::units::PyStress,
        shear_modulus: crate::units::PyStress,
        connectivity: Vec<[usize; N]>,
        reference_nodal_coordinates: PyLength,
    ) -> PyResult<Self> {
        Ok(Self {
            block: Block::from((
                conspire::constitutive::solid::elastic::AlmansiHamelEulerian {
                    bulk_modulus: bulk_modulus.try_into()?,
                    shear_modulus: shear_modulus.try_into()?,
                },
                connectivity,
                &NodalReferenceCoordinates::try_from(&reference_nodal_coordinates)?,
            )),
        })
    }
    fn nodal_forces(&self, nodal_coordinates: PyLength) -> Result<PyForce, PyErrGlue> {
        Ok(self
            .block
            .nodal_forces(&NodalCoordinates::try_from(&nodal_coordinates)?)?
            .into())
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: PyLength,
    ) -> Result<PyForcePerLength, PyErrGlue> {
        Ok((&self
            .block
            .nodal_stiffnesses(&NodalCoordinates::try_from(&nodal_coordinates)?)?)
            .into())
    }
}
