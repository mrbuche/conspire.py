use crate::PyErrGlue;
use conspire::{
    constitutive::{
        Constitutive,
        solid::{
            Solid,
            elastic::Elastic,
            hyperelastic::{Gent as GentConspire, Hyperelastic},
        },
    },
    mechanics::Scalar,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;
use std::fmt::{self, Display, Formatter};

/// The Gent hyperelastic constitutive model.[^gent]
///
/// [^gent]: A.N. Gent, [Rubber Chem. Technol. **69**, 59 (1996)](https://doi.org/10.5254/1.3538357).
///
/// **Parameters**
/// - The bulk modulus $\kappa$.
/// - The shear modulus $\mu$.
/// - The extensibility $J_m$.
///
/// **External variables**
/// - The deformation gradient $\mathbf{F}$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Gent model reduces to the [Neo-Hookean model](#NeoHookean) when $J_m\to\infty$.
#[pyclass(str)]
pub struct Gent {
    model: GentConspire<[Scalar; 3]>,
}

impl Display for Gent {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gent(bulk_modulus={}, shear_modulus={}, extensibility={})",
            self.model.bulk_modulus(),
            self.model.shear_modulus(),
            self.model.extensibility(),
        )
    }
}

#[pymethods]
impl Gent {
    #[new]
    fn new(bulk_modulus: Scalar, shear_modulus: Scalar, extensibility: Scalar) -> Self {
        Self {
            model: GentConspire::new([bulk_modulus, shear_modulus, extensibility]),
        }
    }
    /// $$
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{J^{-1}\mu J_m {\mathbf{B}^* }'}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3} + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// $$
    fn cauchy_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<Scalar>>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<Scalar>> = self
            .model
            .cauchy_stress(&deformation_gradient.into())?
            .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    /// $$
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{J^{-5/3}\mu J_m}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3}\Bigg[ \delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL} + \frac{2{B_{ij}^* }' F_{kL}}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3} - \left(\frac{5}{3} + \frac{2}{3}\frac{\mathrm{tr}(\mathbf{B}^* )}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3}\right) J^{2/3} {B_{ij}^* }' F_{kL}^{-T} \Bigg] + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// $$
    fn cauchy_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<Scalar>>,
    ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Scalar> = self
            .model
            .cauchy_tangent_stiffness(&deformation_gradient.into())?
            .into();
        Ok(PyArray4::from_array(
            py,
            &Array::from_shape_vec((3, 3, 3, 3), cauchy_tangent_stiffness)?,
        ))
    }
    /// $$
    /// \mathbf{P} = J\boldsymbol{\sigma}\cdot\mathbf{F}^{-T}
    /// $$
    fn first_piola_kirchhoff_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<Scalar>>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<Scalar>> = self
            .model
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
        deformation_gradient: Vec<Vec<Scalar>>,
    ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Scalar> = self
            .model
            .first_piola_kirchhoff_tangent_stiffness(&deformation_gradient.into())?
            .into();
        Ok(PyArray4::from_array(
            py,
            &Array::from_shape_vec((3, 3, 3, 3), cauchy_tangent_stiffness)?,
        ))
    }
    /// $$
    /// \mathbf{S} = \mathbf{F}^{-1}\cdot\mathbf{P}
    /// $$
    fn second_piola_kirchhoff_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<Scalar>>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<Scalar>> = self
            .model
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
        deformation_gradient: Vec<Vec<Scalar>>,
    ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Scalar> = self
            .model
            .second_piola_kirchhoff_tangent_stiffness(&deformation_gradient.into())?
            .into();
        Ok(PyArray4::from_array(
            py,
            &Array::from_shape_vec((3, 3, 3, 3), cauchy_tangent_stiffness)?,
        ))
    }
    /// $$
    /// a(\mathbf{F}) = -\frac{\mu J_m}{2}\,\ln\left[1 - \frac{\mathrm{tr}(\mathbf{B}^* ) - 3}{J_m}\right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// $$
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: Vec<Vec<Scalar>>,
    ) -> Result<Scalar, PyErrGlue> {
        Ok(self
            .model
            .helmholtz_free_energy_density(&deformation_gradient.into())?)
    }
}
