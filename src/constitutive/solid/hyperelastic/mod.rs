mod arruda_boyce;
mod blatz_ko;
mod carroll;
mod fung;
mod gent;
mod hencky;
mod isihara;
mod mooney_rivlin;
mod neo_hookean;
mod ogden;
mod saint_venant_kirchhoff;
mod yeoh;

use pyo3::prelude::*;

pub use arruda_boyce::ArrudaBoyce;
pub use blatz_ko::BlatzKo;
pub use carroll::Carroll;
pub use fung::Fung;
pub use gent::Gent;
pub use hencky::Hencky;
pub use isihara::Isihara;
pub use mooney_rivlin::MooneyRivlin;
pub use neo_hookean::NeoHookean;
pub use ogden::Ogden;
pub use saint_venant_kirchhoff::SaintVenantKirchhoff;
pub use yeoh::Yeoh;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrudaBoyce>()?;
    m.add_class::<BlatzKo>()?;
    m.add_class::<Carroll>()?;
    m.add_class::<Fung>()?;
    m.add_class::<Gent>()?;
    m.add_class::<Hencky>()?;
    m.add_class::<Isihara>()?;
    m.add_class::<MooneyRivlin>()?;
    m.add_class::<NeoHookean>()?;
    m.add_class::<Ogden>()?;
    m.add_class::<SaintVenantKirchhoff>()?;
    m.add_class::<Yeoh>()
}

macro_rules! hyperelastic {
    ($model: ident, $($parameter: ident : $ty: ty),+ $(,)?) => {
        use crate::{
            PyErrGlue,
            units::PyStress,
            constitutive::solid::elastic::shared
        };
        use conspire::{
            constitutive::{
                solid::{
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
                        $($parameter: $parameter.try_into()?),+
                    }
                ))
            }
            $(
                /// @private
                #[getter]
                pub fn $parameter(&self) -> $ty {
                    self.0.$parameter.clone().into()
                }
            )+
            #[doc = include_str!("helmholtz_free_energy_density.md")]
            fn helmholtz_free_energy_density(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(self.0.helmholtz_free_energy_density(
                        &deformation_gradient.into()
                    )?.into())
            }
            #[doc = include_str!("cauchy_stress.md")]
            fn cauchy_stress(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(self.0.cauchy_stress(
                        &deformation_gradient.into()
                    )?.into())
            }
            #[doc = include_str!("cauchy_tangent_stiffness.md")]
            fn cauchy_tangent_stiffness(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(self.0.cauchy_tangent_stiffness(
                        &deformation_gradient.into()
                    )?.into())
            }
            #[doc = include_str!("first_piola_kirchhoff_stress.md")]
            fn first_piola_kirchhoff_stress(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(self.0.first_piola_kirchhoff_stress(
                        &deformation_gradient.into()
                    )?.into())
            }
            #[doc = include_str!("first_piola_kirchhoff_tangent_stiffness.md")]
            fn first_piola_kirchhoff_tangent_stiffness(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(self.0.first_piola_kirchhoff_tangent_stiffness(
                        &deformation_gradient.into()
                    )?.into())
            }
            #[doc = include_str!("second_piola_kirchhoff_stress.md")]
            fn second_piola_kirchhoff_stress(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(self.0.second_piola_kirchhoff_stress(
                        &deformation_gradient.into()
                    )?.into())
            }
            #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
            fn second_piola_kirchhoff_tangent_stiffness(
                &self,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<PyStress, PyErrGlue> {
                Ok(self.0.second_piola_kirchhoff_tangent_stiffness(
                        &deformation_gradient.into()
                    )?.into())
            }
        }
    }
}
pub(crate) use hyperelastic;
