mod almansi_hamel;

use pyo3::prelude::*;

pub use almansi_hamel::AlmansiHamel;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlmansiHamel>()
}

macro_rules! shared {
    ($model: ident, $name: literal, $($parameter: ident),+ $(,)?) => {
        #[pyclass(str)]
        pub struct $model {
            model: Inner<[Scalar; count_tts!($($parameter)?)]>,
        }
        use std::fmt::{self, Display, Formatter};
        impl Display for $model {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let args = format!(concat!($(stringify!($parameter), "={}, "),+), $(self.model.$parameter()),+);
                let args = args.strip_suffix(", ").unwrap();
                write!( f, "{}({})", $name, args)
            }
        }
    }
}
pub(crate) use shared;

macro_rules! elastic {
    ($model: ident, $name: literal, $($parameter: ident),+ $(,)?) => {
        use crate::{PyErrGlue, count_tts, replace_expr, constitutive::solid::elastic::shared};
        use conspire::{
            constitutive::{
                Constitutive,
                solid::{Solid, elastic::{Elastic, $model as Inner}},
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
            /// @private
            #[getter]
            pub fn bulk_modulus(&self) -> &Scalar {
                self.model.bulk_modulus()
            }
            /// @private
            #[getter]
            pub fn shear_modulus(&self) -> &Scalar {
                self.model.shear_modulus()
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
    };
}
pub(crate) use elastic;
