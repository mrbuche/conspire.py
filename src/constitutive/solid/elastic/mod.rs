mod almansi_hamel_eulerian;

use pyo3::prelude::*;

pub use almansi_hamel_eulerian::AlmansiHamelEulerian;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlmansiHamelEulerian>()
}

macro_rules! shared {
    ($model: ident, $($parameter: ident : $ty: ty),+ $(,)?) => {
        #[doc = include_str!("doc.md")]
        #[pyclass(str)]
        pub struct $model (Inner);
        use std::fmt::{self, Display, Formatter};
        impl Display for $model {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let args = format!(concat!($(stringify!($parameter), "={:?}, "),+), $(self.0.$parameter()),+);
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
                    Solid,
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
                    self.0.$parameter().into()
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
