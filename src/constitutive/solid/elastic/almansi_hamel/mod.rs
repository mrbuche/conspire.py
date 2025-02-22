use crate::PyErrGlue;
use conspire::constitutive::{
    solid::elastic::{AlmansiHamel as AlmansiHamelConspire, Elastic},
    Constitutive,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

/// The Almansi-Hamel elastic constitutive model.
///
/// **Parameters**
/// - The bulk modulus $\kappa$.
/// - The shear modulus $\mu$.
///
/// **External variables**
/// - The deformation gradient $\mathbf{F}$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Almansi-Hamel strain measure is given by $\mathbf{e}=\tfrac{1}{2}(\mathbf{1}-\mathbf{B}^{-1})$.
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
    /// $$
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{2\mu}{J}\,\mathbf{e}' + \frac{\kappa}{J}\,\mathrm{tr}(\mathbf{e})\mathbf{1}
    /// $$
    fn cauchy_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .cauchy_stress(&deformation_gradient.into())?
                .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    /// $$
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu}{J}\left[B_{jk}^{-1}F_{iL}^{-T} + B_{ik}^{-1}F_{jL}^{-T} - \frac{2}{3}\,\delta_{ij}B_{km}^{-1}F_{mL}^{-T} - 2e_{ij}'F_{kL}^{-T}\right] + \frac{\kappa}{J}\left[\delta_{ij}B_{km}^{-1}F_{mL}^{-T} - \mathrm{tr}(\mathbf{e})\delta_{ij}F_{kL}^{-T}\right]
    /// $$
    fn cauchy_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .cauchy_tangent_stiffness(&deformation_gradient.into())?
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
    /// $$
    /// \mathbf{P} = J\boldsymbol{\sigma}\cdot\mathbf{F}^{-T}
    /// $$
    fn first_piola_kirchhoff_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .first_piola_kirchhoff_stress(&deformation_gradient.into())?
                .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    /// $$
    /// \mathcal{C}_{iJkL} = \frac{\partial P_{iJ}}{\partial F_{kL}} = J \mathcal{T}_{iskL} F_{sJ}^{-T} + P_{iJ} F_{kL}^{-T} - P_{iL} F_{kJ}^{-T}
    /// $$
    fn first_piola_kirchhoff_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .first_piola_kirchhoff_tangent_stiffness(&deformation_gradient.into())?
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
    /// $$
    /// \mathbf{S} = \mathbf{F}^{-1}\cdot\mathbf{P}
    /// $$
    fn second_piola_kirchhoff_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .second_piola_kirchhoff_stress(&deformation_gradient.into())?
                .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    /// $$
    /// \mathcal{G}_{IJkL} = \frac{\partial S_{IJ}}{\partial F_{kL}} = \mathcal{C}_{mJkL}F_{mI}^{-T} - S_{LJ}F_{kI}^{-T} = J \mathcal{T}_{mnkL} F_{mI}^{-T} F_{nJ}^{-T} + S_{IJ} F_{kL}^{-T} - S_{IL} F_{kJ}^{-T} -S_{LJ} F_{kI}^{-T}
    /// $$
    fn second_piola_kirchhoff_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> =
            AlmansiHamelConspire::new(&[self.bulk_modulus, self.shear_modulus])
                .second_piola_kirchhoff_tangent_stiffness(&deformation_gradient.into())?
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
