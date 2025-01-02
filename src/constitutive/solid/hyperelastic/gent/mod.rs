use crate::PyErrGlue;
use conspire::{
    constitutive::{
        solid::{elastic::Elastic, hyperelastic::Gent as GentConspire},
        Constitutive,
    },
    math::TensorArray,
    // mechanics::DeformationGradient,
};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct Gent {
    bulk_modulus: f64,
    shear_modulus: f64,
    extensibility: f64,
}

#[pymethods]
impl Gent {
    #[new]
    fn new(bulk_modulus: f64, shear_modulus: f64, extensibility: f64) -> Self {
        Self {
            bulk_modulus,
            shear_modulus,
            extensibility,
        }
    }
    fn helmholtz_free_energy_density<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: PyReadonlyArray2<f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let f = deformation_gradient
            .as_array()
            .outer_iter()
            .map(|entry| entry.iter().copied().collect())
            .collect();
        Ok(PyArray2::from_vec2(
            py,
            &GentConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extensibility])
                .calculate_cauchy_stress(&f)?
                .as_array()
                .iter()
                .map(|entry| entry.to_vec())
                .collect::<Vec<Vec<f64>>>()
                .to_vec(),
        )
        .unwrap())
    }
}
