mod almansi_hamel_eulerian;
mod almansi_hamel_lagrangian;
mod bazant_itskov_eulerian;
mod bazant_itskov_lagrangian;
mod hencky;
mod saint_venant_kirchhoff;
mod seth_hill_eulerian;
mod seth_hill_lagrangian;

use pyo3::prelude::*;

pub use almansi_hamel_eulerian::AlmansiHamelEulerian;
pub use almansi_hamel_lagrangian::AlmansiHamelLagrangian;
pub use bazant_itskov_eulerian::BazantItskovEulerian;
pub use bazant_itskov_lagrangian::BazantItskovLagrangian;
pub use hencky::Hencky;
pub use saint_venant_kirchhoff::SaintVenantKirchhoff;
pub use seth_hill_eulerian::SethHillEulerian;
pub use seth_hill_lagrangian::SethHillLagrangian;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlmansiHamelEulerian>()?;
    m.add_class::<AlmansiHamelLagrangian>()?;
    m.add_class::<BazantItskovEulerian>()?;
    m.add_class::<BazantItskovLagrangian>()?;
    m.add_class::<Hencky>()?;
    m.add_class::<SaintVenantKirchhoff>()?;
    m.add_class::<SethHillEulerian>()?;
    m.add_class::<SethHillLagrangian>()
}

macro_rules! shared {
    ($model: ident, $($parameter: ident : $ty: ty),+ $(,)?) => {
        #[doc = include_str!("doc.md")]
        #[pyclass(str)]
        pub struct $model (Inner);
        use std::fmt::{self, Display, Formatter};
        impl Display for $model {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let args = format!(concat!($(stringify!($parameter), "={:?}, "),+), $(self.0.$parameter),+);
                let args = args.strip_suffix(", ").unwrap();
                write!( f, "{}({})", stringify!($model), args)
            }
        }
    }
}
pub(crate) use shared;

macro_rules! elastic {
    ($model: ident, $($parameter: ident : $ty: ty),+ $(,)?) => {
        use crate::{
            PyErrGlue,
            units::PyStress,
            constitutive::solid::elastic::shared
        };
        use conspire::{
            constitutive::{
                solid::{
                    elastic::{Elastic, $model as Inner}
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
    };
}
pub(crate) use elastic;
