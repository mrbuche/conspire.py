use crate::PyErrGlue;
use conspire::constitutive::{
    solid::{
        elastic::Elastic,
        hyperelastic::{Hyperelastic, MooneyRivlin as MooneyRivlinConspire},
    },
    Constitutive,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

/// The Mooney-Rivlin hyperelastic constitutive model.[^mooneyrivlin1]<sup>,</sup>[^mooneyrivlin2]
///
/// [^mooneyrivlin1]: M. Mooney, [J. Appl. Phys. **11**, 582 (1940)](https://doi.org/10.1063/1.1712836).
/// [^mooneyrivlin2]: R.S. Rivlin, [Philos. Trans. R. Soc. London, Ser. A **241**, 379 (1948)](https://doi.org/10.1098/rsta.1948.0024).
///
/// **Parameters**
/// - The bulk modulus $\kappa$.
/// - The shear modulus $\mu$.
/// - The extra modulus $\mu_m$.
///
/// **External variables**
/// - The deformation gradient $\mathbf{F}$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Mooney-Rivlin model reduces to the [Neo-Hookean model](#NeoHookean) when $\mu_m\to 0$.
#[pyclass]
pub struct MooneyRivlin {
    bulk_modulus: f64,
    shear_modulus: f64,
    extra_modulus: f64,
}

#[pymethods]
impl MooneyRivlin {
    #[new]
    fn new(bulk_modulus: f64, shear_modulus: f64, extra_modulus: f64) -> Self {
        Self {
            bulk_modulus,
            shear_modulus,
            extra_modulus,
        }
    }
    /// $$
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{\mu - \mu_m}{J}\, {\mathbf{B}^* }' - \frac{\mu_m}{J}\left(\mathbf{B}^{* -1}\right)' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// $$
    fn cauchy_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> =
            MooneyRivlinConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extra_modulus])
                .cauchy_stress(&deformation_gradient.into())?
                .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    /// $$
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu-\mu_m}{J^{5/3}}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL}- \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) - \frac{\mu_m}{J}\left[ \frac{2}{3}\,B_{ij}^{* -1}F_{kL}^{-T} - B_{ik}^{* -1}F_{jL}^{-T} - B_{ik}^{* -1}F_{iL}^{-T} + \frac{2}{3}\,\delta_{ij}\left(B_{km}^{* -1}\right)'F_{mL}^{-T} - \left(B_{ij}^{* -1}\right)'F_{kL}^{-T} \right] + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// $$
    fn cauchy_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> =
            MooneyRivlinConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extra_modulus])
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
            MooneyRivlinConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extra_modulus])
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
            MooneyRivlinConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extra_modulus])
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
            MooneyRivlinConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extra_modulus])
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
            MooneyRivlinConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extra_modulus])
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
    /// a(\mathbf{F}) = \frac{\mu - \mu_m}{2}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right] + \frac{\mu_m}{2}\left[I_2(\mathbf{B}^*) - 3\right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// $$
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<f64, PyErrGlue> {
        Ok(
            MooneyRivlinConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extra_modulus])
                .helmholtz_free_energy_density(&deformation_gradient.into())?,
        )
    }
}
