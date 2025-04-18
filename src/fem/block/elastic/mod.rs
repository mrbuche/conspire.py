use crate::{PyErrGlue, constitutive::solid::elastic as constitutive, fem::call_method};
use conspire::{
    fem::{
        Connectivity, ElasticFiniteElementBlock, ElementBlock, FiniteElementBlock,
        LinearTetrahedron, NodalCoordinatesBlock, ReferenceNodalCoordinatesBlock,
    },
    math::TensorVec,
    mechanics::Scalar,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

const N: usize = 4;

#[pyclass]
pub enum ElasticBlock {
    AlmansiHamel(Py<AlmansiHamel>),
}

#[derive(FromPyObject)]
enum ElasticModel<'py> {
    AlmansiHamel(Bound<'py, constitutive::AlmansiHamel>),
}

#[pymethods]
impl ElasticBlock {
    #[new]
    fn new(
        py: Python,
        model: ElasticModel,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Self, PyErr> {
        match model {
            ElasticModel::AlmansiHamel(model) => {
                let bulk_modulus: Scalar = model.getattr("bulk_modulus")?.extract()?;
                let shear_modulus: Scalar = model.getattr("shear_modulus")?.extract()?;
                let block = AlmansiHamel::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::AlmansiHamel(Py::new(py, block)?))
            }
        }
    }
    fn nodal_forces<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        match self {
            Self::AlmansiHamel(model) => call_method!(model, py, "nodal_forces", nodal_coordinates),
        }
    }
    fn nodal_stiffnesses<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
        match self {
            Self::AlmansiHamel(model) => {
                call_method!(model, py, "nodal_stiffnesses", nodal_coordinates)
            }
        }
    }
}

#[pyclass]
pub struct AlmansiHamel {
    block: ElementBlock<
        LinearTetrahedron<conspire::constitutive::solid::elastic::AlmansiHamel<[Scalar; 2]>>,
        N,
    >,
}

#[pymethods]
impl AlmansiHamel {
    #[new]
    pub fn new(
        bulk_modulus: Scalar,
        shear_modulus: Scalar,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Self {
        Self {
            block: ElementBlock::new(
                [bulk_modulus, shear_modulus],
                connectivity,
                ReferenceNodalCoordinatesBlock::new(&reference_nodal_coordinates),
            ),
        }
    }
    fn nodal_forces<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        let forces: Vec<Vec<Scalar>> = self
            .block
            .nodal_forces(&NodalCoordinatesBlock::new(&nodal_coordinates))?
            .into();
        Ok(PyArray2::from_vec2(py, &forces)?)
    }
    fn nodal_stiffnesses<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
        let stiffnesses: Vec<Scalar> = self
            .block
            .nodal_stiffnesses(&NodalCoordinatesBlock::new(&nodal_coordinates))?
            .into();
        let nodes = nodal_coordinates.len();
        Ok(PyArray4::from_array(
            py,
            &Array::from_shape_vec((nodes, nodes, 3, 3), stiffnesses)?,
        ))
    }
}
