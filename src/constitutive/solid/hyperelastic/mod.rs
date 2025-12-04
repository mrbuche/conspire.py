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
    ($model: ident, $($parameter: ident),+ $(,)?) => {
        use crate::{PyErrGlue, constitutive::solid::elastic::shared};
        use conspire::{
            constitutive::{
                solid::{Solid, elastic::Elastic, hyperelastic::{Hyperelastic, $model as Inner}},
            },
            mechanics::Scalar,
        };
        use ndarray::Array;
        use numpy::{PyArray2, PyArray4};
        use pyo3::prelude::*;
        shared!($model, $($parameter),+);
        #[pymethods]
        impl $model {
            #[new]
            fn new($($parameter: Scalar),+) -> Self {
                Self (
                    Inner {
                        $($parameter),+
                    }
                )
            }
            $(
                /// @private
                #[getter]
                pub fn $parameter(&self) -> Scalar {
                    self.0.$parameter()
                }
            )+
            #[doc = include_str!("helmholtz_free_energy_density.md")]
            fn helmholtz_free_energy_density(
                &self,
                deformation_gradient: Vec<Vec<Scalar>>,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(self
                    .0
                    .helmholtz_free_energy_density(&deformation_gradient.into())?)
            }
            #[doc = include_str!("cauchy_stress.md")]
            fn cauchy_stress<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: Vec<Vec<Scalar>>,
            ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
                let cauchy_stress: Vec<Vec<Scalar>> = self
                    .0
                    .cauchy_stress(&deformation_gradient.into())?
                    .into();
                Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
            }
            #[doc = include_str!("cauchy_tangent_stiffness.md")]
            fn cauchy_tangent_stiffness<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: Vec<Vec<Scalar>>,
            ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
                Ok(PyArray4::from_array(
                    py,
                    &Array::from_shape_vec(
                        (3, 3, 3, 3),
                        self.0
                            .cauchy_tangent_stiffness(
                                &deformation_gradient.into()
                            )?.into()
                    )?,
                ))
            }
            #[doc = include_str!("first_piola_kirchhoff_stress.md")]
            fn first_piola_kirchhoff_stress<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: Vec<Vec<Scalar>>,
            ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
                let cauchy_stress: Vec<Vec<Scalar>> = self
                    .0
                    .first_piola_kirchhoff_stress(&deformation_gradient.into())?
                    .into();
                Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
            }
            #[doc = include_str!("first_piola_kirchhoff_tangent_stiffness.md")]
            fn first_piola_kirchhoff_tangent_stiffness<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: Vec<Vec<Scalar>>,
            ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
                Ok(PyArray4::from_array(
                    py,
                    &Array::from_shape_vec(
                        (3, 3, 3, 3),
                        self.0
                            .first_piola_kirchhoff_tangent_stiffness(
                                &deformation_gradient.into()
                            )?.into()
                    )?,
                ))
            }
            #[doc = include_str!("second_piola_kirchhoff_stress.md")]
            fn second_piola_kirchhoff_stress<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: Vec<Vec<Scalar>>,
            ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
                let cauchy_stress: Vec<Vec<Scalar>> = self
                    .0
                    .second_piola_kirchhoff_stress(&deformation_gradient.into())?
                    .into();
                Ok(PyArray2::from_vec2(py, &cauchy_stress)?)
            }
            #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
            fn second_piola_kirchhoff_tangent_stiffness<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: Vec<Vec<Scalar>>,
            ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
                Ok(PyArray4::from_array(
                    py,
                    &Array::from_shape_vec(
                        (3, 3, 3, 3),
                        self.0
                            .second_piola_kirchhoff_tangent_stiffness(
                                &deformation_gradient.into()
                            )?.into()
                    )?,
                ))
            }
        }
    }
}
pub(crate) use hyperelastic;
