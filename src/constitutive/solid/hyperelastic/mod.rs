mod arruda_boyce;
mod fung;
mod gent;
mod hencky;
mod mooney_rivlin;
mod neo_hookean;
mod saint_venant_kirchhoff;

use pyo3::prelude::*;

pub use arruda_boyce::ArrudaBoyce;
pub use fung::Fung;
pub use gent::Gent;
pub use hencky::Hencky;
pub use mooney_rivlin::MooneyRivlin;
pub use neo_hookean::NeoHookean;
pub use saint_venant_kirchhoff::SaintVenantKirchhoff;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrudaBoyce>()?;
    m.add_class::<Fung>()?;
    m.add_class::<Gent>()?;
    m.add_class::<Hencky>()?;
    m.add_class::<MooneyRivlin>()?;
    m.add_class::<NeoHookean>()?;
    m.add_class::<SaintVenantKirchhoff>()
}

macro_rules! hyperelastic {
    ($model: ident, $($parameter: ident : $ty: ty),+ $(,)?) => {
        use crate::{
            PyErrGlue,
            math::{PyTensorRank2, PyTensorRank4},
            units::{Parameter, PyStress},
            constitutive::solid::elastic::shared
        };
        use conspire::{
            constitutive::{
                solid::{
                    Solid,
                    elastic::Elastic,
                    hyperelastic::{Hyperelastic, $model as Inner}
                },
            },
            mechanics::Scalar,
        };
        use pyo3::prelude::*;
        shared!($model, $($parameter: $ty),+);
        #[pymethods]
        impl $model {
            #[new]
            fn new($($parameter: $ty),+) -> PyResult<Self> {
                Ok(Self (
                    Inner {
                        $($parameter: $parameter.into_inner()?),+
                    }
                ))
            }
            $(
                /// @private
                #[getter]
                pub fn $parameter(&self) -> $ty {
                    <$ty>::from_inner(self.0.$parameter())
                }
            )+
            #[doc = include_str!("helmholtz_free_energy_density.md")]
            fn helmholtz_free_energy_density(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(PyStress::from_scalar(
                    self.0.helmholtz_free_energy_density(
                        &deformation_gradient.into()
                    )?
                ))
            }
            #[doc = include_str!("cauchy_stress.md")]
            fn cauchy_stress(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(PyStress::from_array(PyTensorRank2::from(self.0.cauchy_stress(
                        &deformation_gradient.into()
                    )?).into_array()?))
            }
            #[doc = include_str!("cauchy_tangent_stiffness.md")]
            fn cauchy_tangent_stiffness(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(PyStress::from_array(PyTensorRank4::from(self.0.cauchy_tangent_stiffness(
                        &deformation_gradient.into()
                    )?).into_array()?))
            }
            #[doc = include_str!("first_piola_kirchhoff_stress.md")]
            fn first_piola_kirchhoff_stress(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(PyStress::from_array(PyTensorRank2::from(self.0.first_piola_kirchhoff_stress(
                        &deformation_gradient.into()
                    )?).into_array()?))
            }
            #[doc = include_str!("first_piola_kirchhoff_tangent_stiffness.md")]
            fn first_piola_kirchhoff_tangent_stiffness(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(PyStress::from_array(PyTensorRank4::from(self.0.first_piola_kirchhoff_tangent_stiffness(
                        &deformation_gradient.into()
                    )?).into_array()?))
            }
            #[doc = include_str!("second_piola_kirchhoff_stress.md")]
            fn second_piola_kirchhoff_stress(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(PyStress::from_array(PyTensorRank2::from(self.0.second_piola_kirchhoff_stress(
                        &deformation_gradient.into()
                    )?).into_array()?))
            }
            #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
            fn second_piola_kirchhoff_tangent_stiffness(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(PyStress::from_array(PyTensorRank4::from(self.0.second_piola_kirchhoff_tangent_stiffness(
                        &deformation_gradient.into()
                    )?).into_array()?))
            }
        }
    }
}
pub(crate) use hyperelastic;
