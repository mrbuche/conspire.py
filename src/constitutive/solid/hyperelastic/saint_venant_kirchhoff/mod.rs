use crate::PyErrGlue;
use conspire::{
    constitutive::{
        Constitutive,
        solid::{
            Solid,
            elastic::Elastic,
            hyperelastic::{Hyperelastic, SaintVenantKirchhoff as SaintVenantKirchhoffConspire},
        },
    },
    mechanics::Scalar,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;
use std::fmt::{self, Display, Formatter};

/// The Saint Venant-Kirchhoff hyperelastic constitutive model.
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
/// - The Green-Saint Venant strain measure is given by $\mathbf{E}=\tfrac{1}{2}(\mathbf{C}-\mathbf{1})$.
#[pyclass(str)]
pub struct SaintVenantKirchhoff {
    model: SaintVenantKirchhoffConspire<[Scalar; 2]>,
}

impl Display for SaintVenantKirchhoff {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SaintVenantKirchhoff(bulk_modulus={}, shear_modulus={})",
            self.model.bulk_modulus(),
            self.model.shear_modulus()
        )
    }
}

#[pymethods]
impl SaintVenantKirchhoff {
    #[new]
    fn new(bulk_modulus: Scalar, shear_modulus: Scalar) -> Self {
        Self {
            model: SaintVenantKirchhoffConspire::new([bulk_modulus, shear_modulus]),
        }
    }
    #[getter]
    pub fn bulk_modulus(&self) -> &Scalar {
        self.model.bulk_modulus()
    }
    #[getter]
    pub fn shear_modulus(&self) -> &Scalar {
        self.model.shear_modulus()
    }
    /// $$
    /// \boldsymbol{\sigma} = J^{-1}\mathbf{P}\cdot\mathbf{F}^T
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
    /// \mathcal{T}_{ijkL} = \frac{\partial\sigma_{ij}}{\partial F_{kL}} = J^{-1} \mathcal{G}_{MNkL} F_{iM} F_{jN} - \sigma_{ij} F_{kL}^{-T} + \left(\delta_{jk}\sigma_{is} + \delta_{ik}\sigma_{js}\right)F_{sL}^{-T}
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
    /// \mathbf{S}(\mathbf{F}) = 2\mu\mathbf{E}' + \kappa\,\mathrm{tr}(\mathbf{E})\mathbf{1}
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
    /// \mathcal{G}_{IJkL}(\mathbf{F}) = \mu\,\delta_{JL}F_{kI} + \mu\,\delta_{IL}F_{kJ} + \left(\kappa - \frac{2}{3}\,\mu\right)\delta_{IJ}F_{kL}
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
    /// a(\mathbf{F}) = \mu\,\mathrm{tr}(\mathbf{E}^2) + \frac{1}{2}\left(\kappa - \frac{2}{3}\,\mu\right)\mathrm{tr}(\mathbf{E})^2
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
