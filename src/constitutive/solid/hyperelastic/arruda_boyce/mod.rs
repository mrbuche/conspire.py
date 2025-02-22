use crate::PyErrGlue;
use conspire::constitutive::{
    solid::{
        elastic::Elastic,
        hyperelastic::{ArrudaBoyce as ArrudaBoyceConspire, Hyperelastic},
    },
    Constitutive,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

/// The Arruda-Boyce hyperelastic constitutive model.[^arrudaboyce]
///
/// [^arrudaboyce]: E.M. Arruda and M.C. Boyce, [J. Mech. Phys. Solids **41**, 389 (1993)](https://doi.org/10.1016/0022-5096(93)90013-6).
///
/// **Parameters**
/// - The bulk modulus $\kappa$.
/// - The shear modulus $\mu$.
/// - The number of links $N_b$.
///
/// **External variables**
/// - The deformation gradient $\mathbf{F}$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The nondimensional end-to-end length per link of the chains is $\gamma=\sqrt{\mathrm{tr}(\mathbf{B}^*)/3N_b}$.
/// - The nondimensional force is given by the inverse Langevin function as $\eta=\mathcal{L}^{-1}(\gamma)$.
/// - The initial values are given by $\gamma_0=\sqrt{1/3N_b}$ and $\eta_0=\mathcal{L}^{-1}(\gamma_0)$.
/// - The Arruda-Boyce model reduces to the [Neo-Hookean model](#NeoHookean) when $N_b\to\infty$.
#[pyclass]
pub struct ArrudaBoyce {
    bulk_modulus: f64,
    shear_modulus: f64,
    number_of_links: f64,
}

#[pymethods]
impl ArrudaBoyce {
    #[new]
    fn new(bulk_modulus: f64, shear_modulus: f64, number_of_links: f64) -> Self {
        Self {
            bulk_modulus,
            shear_modulus,
            number_of_links,
        }
    }
    /// $$
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{\mu\gamma_0\eta}{J\gamma\eta_0}\,{\mathbf{B}^*}' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// $$
    fn cauchy_stress<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyErrGlue> {
        let cauchy_stress: Vec<Vec<f64>> = ArrudaBoyceConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.number_of_links,
        ])
        .cauchy_stress(&deformation_gradient.into())?
        .into();
        Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
    }
    /// $$
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu\gamma_0\eta}{J^{5/3}\gamma\eta_0}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL}- \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) + \frac{\mu\gamma_0\eta}{3J^{7/3}N_b\gamma^2\eta_0}\left(\frac{1}{\eta\mathcal{L}'(\eta)} - \frac{1}{\gamma}\right)B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// $$
    fn cauchy_tangent_stiffness<'py>(
        &self,
        py: Python<'py>,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<Bound<'py, PyArray4<f64>>, PyErrGlue> {
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> = ArrudaBoyceConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.number_of_links,
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
        let cauchy_stress: Vec<Vec<f64>> = ArrudaBoyceConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.number_of_links,
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
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> = ArrudaBoyceConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.number_of_links,
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
        let cauchy_stress: Vec<Vec<f64>> = ArrudaBoyceConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.number_of_links,
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
        let cauchy_tangent_stiffness: Vec<Vec<Vec<Vec<f64>>>> = ArrudaBoyceConspire::new(&[
            self.bulk_modulus,
            self.shear_modulus,
            self.number_of_links,
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
    /// a(\mathbf{F}) = \frac{3\mu N_b\gamma_0}{\eta_0}\left[\gamma\eta - \gamma_0\eta_0 - \ln\left(\frac{\eta_0\sinh\eta}{\eta\sinh\eta_0}\right) \right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// $$
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: Vec<Vec<f64>>,
    ) -> Result<f64, PyErrGlue> {
        Ok(
            ArrudaBoyceConspire::new(&[
                self.bulk_modulus,
                self.shear_modulus,
                self.number_of_links,
            ])
            .helmholtz_free_energy_density(&deformation_gradient.into())?,
        )
    }
}
