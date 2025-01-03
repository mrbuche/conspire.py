use crate::PyErrGlue;
use conspire::constitutive::{
    solid::elastic::{AlmansiHamel as AlmansiHamelConspire, Elastic},
    Constitutive,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

#[pyclass]
pub struct AlmansiHamel {
    bulk_modulus: f64,
    shear_modulus: f64,
}

#[pymethods]
impl AlmansiHamel {
    #[new]
    fn new(bulk_modulus: f64, shear_modulus: f64) -> Self {
        Self {
            bulk_modulus,
            shear_modulus,
        }
    }
    fn cauchy_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .calculate_cauchy_stress(&deformation_gradient.into())?
                .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    fn cauchy_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .calculate_cauchy_tangent_stiffness(&deformation_gradient.into())?
                .into();
        Ok(PyArray4::from_array(
            py,
            &Array::from_shape_vec(
                (3, 3, 3, 3),
                cauchy_tangent_stiffness
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect(),
            )?,
        ))
    }
    fn first_piola_kirchoff_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .calculate_first_piola_kirchoff_stress(&deformation_gradient.into())?
                .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    fn first_piola_kirchoff_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .calculate_first_piola_kirchoff_tangent_stiffness(&deformation_gradient.into())?
                .into();
        Ok(PyArray4::from_array(
            py,
            &Array::from_shape_vec(
                (3, 3, 3, 3),
                cauchy_tangent_stiffness
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect(),
            )?,
        ))
    }
    fn second_piola_kirchoff_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .calculate_second_piola_kirchoff_stress(&deformation_gradient.into())?
                .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    fn second_piola_kirchoff_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .calculate_second_piola_kirchoff_tangent_stiffness(&deformation_gradient.into())?
                .into();
        Ok(PyArray4::from_array(
            py,
            &Array::from_shape_vec(
                (3, 3, 3, 3),
                cauchy_tangent_stiffness
                    .into_iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .collect(),
            )?,
        ))
    }
}
