use crate::PyErrGlue;
use conspire::constitutive::{
    solid::{
        elastic::Elastic,
        hyperelastic::{Fung as FungConspire, Hyperelastic},
    },
    Constitutive,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

/// The Fung hyperelastic constitutive model.[^fung]
///
/// [^fung]: Y.C. Fung, [Am. J. Physiol. **213**, 1532 (1967)](https://doi.org/10.1152/ajplegacy.1967.213.6.1532).
///
/// **Parameters**
/// - The bulk modulus $\kappa$.
/// - The shear modulus $\mu$.
/// - The extra modulus $\mu_m$.
/// - The exponent $c$.
///
/// **External variables**
/// - The deformation gradient $\mathbf{F}$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Fung model reduces to the [Neo-Hookean model](#NeoHookean) when $\mu_m\to 0$ or $c\to 0$.
#[pyclass]
pub struct Fung {
    bulk_modulus: f64,
    shear_modulus: f64,
    extra_modulus: f64,
    exponent: f64,
}

#[pymethods]
impl Fung {
    #[new]
    fn new(bulk_modulus: f64, shear_modulus: f64, extra_modulus: f64, exponent: f64) -> Self {
        Self {
            bulk_modulus,
            shear_modulus,
            extra_modulus,
            exponent,
        }
    }
    /// $$
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{1}{J}\left[\mu + \mu_m\left(e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)\right]{\mathbf{B}^* }' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// $$
    fn cauchy_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> = FungConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.extra_modulus,
            self.exponent,
        ])
        .cauchy_stress(&deformation_gradient.into())?
        .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    /// $$
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{1}{J^{5/3}}\left[\mu + \mu_m\left(e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)\right]\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL} - \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) + \frac{2c\mu_m}{J^{7/3}}\,e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]}B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// $$
    fn cauchy_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> = FungConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.extra_modulus,
            self.exponent,
        ])
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
        let cauchy_stress: Vec<Vec<f64>> = FungConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.extra_modulus,
            self.exponent,
        ])
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
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> = FungConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.extra_modulus,
            self.exponent,
        ])
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
        let cauchy_stress: Vec<Vec<f64>> = FungConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.extra_modulus,
            self.exponent,
        ])
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
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> = FungConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.extra_modulus,
            self.exponent,
        ])
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
    /// $$
    /// a(\mathbf{F}) = \frac{\mu - \mu_m}{2}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right] + \frac{\mu_m}{2c}\left(e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)
    /// $$
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<f64, PyErrGlue> {
        Ok(FungConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.extra_modulus,
            self.exponent,
        ])
        .helmholtz_free_energy_density(&deformation_gradient.into())?)
    }
}
