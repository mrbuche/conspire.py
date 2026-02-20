mod almansi_hamel;

use pyo3::prelude::*;

pub use almansi_hamel::AlmansiHamel;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlmansiHamel>()
}

macro_rules! shared {
    ($model: ident, $($parameter: ident),+ $(,)?) => {
        #[doc = include_str!("doc.md")]
        #[pyclass(str)]
        pub struct $model (Inner);
        use std::fmt::{self, Display, Formatter};
        impl Display for $model {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let args = format!(concat!($(stringify!($parameter), "={}, "),+), $(self.0.$parameter()),+);
                let args = args.strip_suffix(", ").unwrap();
                write!( f, "{}({})", stringify!($model), args)
            }
        }
    }
}
pub(crate) use shared;

macro_rules! elastic {
    ($model: ident, $($parameter: ident),+ $(,)?) => {
        use crate::{
            PyErrGlue,
            math::{PyTensorRank2, PyTensorRank4},
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
            #[doc = include_str!("cauchy_stress.md")]
            fn cauchy_stress<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
                PyTensorRank2::from(
                    self.0.cauchy_stress(
                        &deformation_gradient.into()
                    )?
                ).into_pyarray(py)
            }
            #[doc = include_str!("cauchy_tangent_stiffness.md")]
            fn cauchy_tangent_stiffness<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
                PyTensorRank4::from(
                    self.0.cauchy_tangent_stiffness(
                        &deformation_gradient.into()
                    )?
                ).into_pyarray(py)
            }
            #[doc = include_str!("first_piola_kirchhoff_stress.md")]
            fn first_piola_kirchhoff_stress<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
                PyTensorRank2::from(
                    self.0.first_piola_kirchhoff_stress(
                        &deformation_gradient.into()
                    )?
                ).into_pyarray(py)
            }
            #[doc = include_str!("first_piola_kirchhoff_tangent_stiffness.md")]
            fn first_piola_kirchhoff_tangent_stiffness<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
                PyTensorRank4::from(
                    self.0.first_piola_kirchhoff_tangent_stiffness(
                        &deformation_gradient.into()
                    )?
                ).into_pyarray(py)
            }
            #[doc = include_str!("second_piola_kirchhoff_stress.md")]
            fn second_piola_kirchhoff_stress<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
                PyTensorRank2::from(
                    self.0.second_piola_kirchhoff_stress(
                        &deformation_gradient.into()
                    )?
                ).into_pyarray(py)
            }
            #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
            fn second_piola_kirchhoff_tangent_stiffness<'py>(
                &self,
                py: Python<'py>,
                deformation_gradient: [[Scalar; 3]; 3],
            ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
                PyTensorRank4::from(
                    self.0.second_piola_kirchhoff_tangent_stiffness(
                        &deformation_gradient.into()
                    )?
                ).into_pyarray(py)
            }
        }
    };
}
pub(crate) use elastic;
