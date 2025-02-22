use crate::PyErrGlue;
use conspire::constitutive::{
    solid::{
        elastic::Elastic,
        hyperelastic::{Hyperelastic, Yeoh as YeohConspire},
    },
    Constitutive,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

/// The Yeoh hyperelastic constitutive model.[^yeoh]
///
/// [^yeoh]: O.H. Yeoh, [Rubber Chem. Technol. **66**, 754 (1993)](https://doi.org/10.5254/1.3538343).
///
/// **Parameters**
/// - The bulk modulus $\kappa$.
/// - The shear modulus $\mu$.
/// - The extra moduli $\mu_n$ for $n=2\ldots N$.
///
/// **External variables**
/// - The deformation gradient $\mathbf{F}$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Yeoh model reduces to the [Neo-Hookean model](#NeoHookean) when $\mu_n\to 0$ for $n=2\ldots N$.
#[pyclass]
pub struct Yeoh {
    bulk_modulus: f64,
    shear_modulus: f64,
    extra_moduli: &[f64],
}

#[pymethods]
impl Yeoh {
    #[new]
    fn new(bulk_modulus: f64, shear_modulus: f64, extensibility: f64) -> Self {
        Self {
            bulk_modulus,
            shear_modulus,
            extensibility,
        }
    }
    /// $$
    /// \boldsymbol{\sigma}(\mathbf{F}) = \sum_{n=1}^N \frac{n\mu_n}{J}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^{n-1}\,{\mathbf{B}^*}' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// $$
    fn cauchy_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> =
            YeohConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extensibility])
                .cauchy_stress(&deformation_gradient.into())?
                .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    /// $$
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \sum_{n=1}^N \frac{n\mu_n}{J^{5/3}}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^{n-1}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL}- \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) + \sum_{n=2}^N \frac{2n(n-1)\mu_n}{J^{7/3}}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^{n-2}B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// $$
    fn cauchy_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> =
            YeohConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extensibility])
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
            YeohConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extensibility])
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
            YeohConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extensibility])
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
            YeohConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extensibility])
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
            YeohConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extensibility])
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
    /// a(\mathbf{F}) = \sum_{n=1}^N \frac{\mu_n}{2}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^n + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// $$
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<f64, PyErrGlue> {
        Ok(
            YeohConspire::new(&[self.bulk_modulus, self.shear_modulus, self.extensibility])
                .helmholtz_free_energy_density(&deformation_gradient.into())?,
        )
    }
}
