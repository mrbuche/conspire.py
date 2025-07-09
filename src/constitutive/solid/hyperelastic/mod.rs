mod arruda_boyce;
mod fung;
mod gent;
mod mooney_rivlin;
mod neo_hookean;
mod saint_venant_kirchhoff;

use pyo3::prelude::*;

pub use arruda_boyce::ArrudaBoyce;
pub use fung::Fung;
pub use gent::Gent;
pub use mooney_rivlin::MooneyRivlin;
pub use neo_hookean::NeoHookean;
pub use saint_venant_kirchhoff::SaintVenantKirchhoff;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrudaBoyce>()?;
    m.add_class::<Fung>()?;
    m.add_class::<Gent>()?;
    m.add_class::<MooneyRivlin>()?;
    m.add_class::<NeoHookean>()?;
    m.add_class::<SaintVenantKirchhoff>()
}

macro_rules! hyperelastic {
    ($model: ident, $name: literal, $($parameter: ident),+ $(,)?) => {
        use crate::{PyErrGlue, count_tts, replace_expr, constitutive::solid::elastic::shared};
        use conspire::{
            constitutive::{
                Constitutive,
                solid::{Solid, elastic::Elastic, hyperelastic::{Hyperelastic, $model as Inner}},
            },
            mechanics::Scalar,
        };
        use ndarray::Array;
        use numpy::{PyArray2, PyArray4};
        use pyo3::prelude::*;
        shared!($model, $name, $($parameter),+);
        #[pymethods]
        impl $model {
            #[new]
            fn new($($parameter: Scalar),+) -> Self {
                Self {
                    model: Inner::new([$($parameter),+]),
                }
            }
            $(
                /// @private
                #[getter]
                pub fn $parameter(&self) -> &Scalar {
                    self.model.$parameter()
                }
            )+
            fn helmholtz_free_energy_density(
                &self,
                deformation_gradient: Vec<Vec<Scalar>>,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(self
                    .model
                    .helmholtz_free_energy_density(&deformation_gradient.into())?)
            }
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
        }
    }
}
pub(crate) use hyperelastic;
