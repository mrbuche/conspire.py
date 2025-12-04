pub mod bogacki_shampine;
pub mod dormand_prince;
pub mod verner_8;
pub mod verner_9;

macro_rules! explicit {
    ($method: ident) => {
        use crate::PyErrGlue;
        use conspire::math::{
            Matrix, Scalar, Vector,
            integrate::{self, Explicit},
        };
        use pyo3::{
            prelude::*,
            types::{PyDict, PyFunction},
        };
        #[doc = include_str!("doc.md")]
        #[pyclass]
        pub struct $method(integrate::$method);
        #[pymethods]
        impl $method {
            #[new]
            #[pyo3(signature = (**kwargs))]
            fn new(kwargs: Option<&Bound<'_, PyDict>>) -> Result<Self, PyErr> {
                let mut integrator = integrate::$method::default();
                if let Some(args) = kwargs {
                    args.into_iter().try_for_each(|(name, value)| {
                        match name.extract()? {
                            "abs_tol" => integrator.abs_tol = value.extract()?,
                            "rel_tol" => integrator.rel_tol = value.extract()?,
                            "dt_beta" => integrator.dt_beta = value.extract()?,
                            "dt_expn" => integrator.dt_expn = value.extract()?,
                            "dt_cut" => integrator.dt_cut = value.extract()?,
                            "dt_min" => integrator.dt_min = value.extract()?,
                            _ => (),
                        };
                        Ok::<(), PyErr>(())
                    })?
                }
                Ok(Self(integrator))
            }
            /// @private
            #[getter]
            pub fn abs_tol(&self) -> Scalar {
                self.0.abs_tol
            }
            /// @private
            #[getter]
            pub fn rel_tol(&self) -> Scalar {
                self.0.rel_tol
            }
            /// @private
            #[getter]
            pub fn dt_beta(&self) -> Scalar {
                self.0.dt_beta
            }
            /// @private
            #[getter]
            pub fn dt_expn(&self) -> Scalar {
                self.0.dt_expn
            }
            /// @private
            #[getter]
            pub fn dt_cut(&self) -> Scalar {
                self.0.dt_cut
            }
            /// @private
            #[getter]
            pub fn dt_min(&self) -> Scalar {
                self.0.dt_min
            }
            #[doc = include_str!("../doc.md")]
            fn integrate(
                &self,
                py: Python,
                function: Py<PyFunction>,
                time: Vec<Scalar>,
                initial_condition: Vec<Scalar>,
            ) -> Result<(Vec<Scalar>, Vec<Vec<Scalar>>, Vec<Vec<Scalar>>), PyErrGlue> {
                let (t, y, dydt): (Vector, Matrix, Matrix) = self.0.integrate(
                    |t: Scalar, y: &Vector| {
                        function
                            .call1(py, (t, Vec::from(y.clone())))
                            .map_err(|e| e.to_string())
                            .and_then(|val| {
                                val.extract::<Vec<_>>(py)
                                    .map(Vector::from)
                                    .map_err(|e| e.to_string())
                            })
                    },
                    &time,
                    Vector::from(initial_condition),
                )?;
                Ok((t.into(), y.into(), dydt.into()))
            }
        }
    };
}
pub(crate) use explicit;
