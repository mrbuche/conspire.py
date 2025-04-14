use super::PyErrGlue;
use conspire::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Connectivity, ElasticFiniteElementBlock, ElementBlock, FiniteElementBlock,
        HyperelasticFiniteElementBlock, LinearTetrahedron, NodalCoordinatesBlock,
        ReferenceNodalCoordinatesBlock,
    },
    math::TensorVec,
    mechanics::Scalar,
};
use numpy::PyArray2;
use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeoHookeanBlock>()
}

const N: usize = 4;
type Parameters = [Scalar; 2];
type Model = NeoHookean<Parameters>;

/// ???
#[pyclass]
pub struct NeoHookeanBlock {
    block: ElementBlock<LinearTetrahedron<Model>, N>,
}

#[pymethods]
impl NeoHookeanBlock {
    #[new]
    fn new(
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
    /// ???
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Scalar, PyErrGlue> {
        Ok(self
            .block
            .helmholtz_free_energy(&NodalCoordinatesBlock::new(&nodal_coordinates))?)
    }
    /// ???
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
    // /// ???
    // fn nodal_stiffnesses<'py>(&self, py: Python<'py>, nodal_coordinates: Vec<[Scalar; 3]>) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {}
}
