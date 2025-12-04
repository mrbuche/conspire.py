use conspire::math::{
    Scalar,Vector,
    integrate::{self, Explicit, IntegrationError},
};
use pyo3::{prelude::*, types::{PyFunction, PyDict}};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BogackiShampine>()
}

/// (import doc.md)
#[pyclass]
pub struct BogackiShampine(integrate::BogackiShampine);

#[pymethods]
impl BogackiShampine {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> Result<Self, PyErr> {
        let mut integrator = integrate::BogackiShampine::default();
        if let Some(args) = kwargs {
            args.into_iter().try_for_each(|(name, value)| {
                match name.extract()? {
                    "abs_tol" => integrator.abs_tol = value.extract()?,
                    "rel_tol" => integrator.rel_tol = value.extract()?,
                    "dt_beta" => integrator.dt_beta = value.extract()?,
                    "dt_expn" => integrator.dt_expn = value.extract()?,
                    // "dt_cut" => integrator.dt_cut = value.extract()?,
                    // "dt_min" => integrator.dt_min = value.extract()?,
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
    /// ???
    fn integrate(&self, function: Py<PyFunction>, time: Vec<Scalar>, initial_condition: Scalar) -> Vec<Scalar> {
        let (foo, _, _): (Vector, Vector, Vector) = self.0.integrate(
            |t: Scalar, y: &Scalar| {
                match Python::attach(|py| {
                    function.call1(py, (t, *y))
                        .map_err(|e| e.to_string())
                        .and_then(|val| val.extract::<Scalar>(py).map_err(|e| e.to_string()))
                }) {
                    Ok(val) => Ok(val),
                    Err(e) => panic!(),
                }
            },
            &time, initial_condition).unwrap();
        todo!()
        // foo.into()
    //     let mut func = function;
    //     let result = self.0.integrate(
    //         |t: Scalar, y: &Scalar| -> Result<Scalar, String> {
    //             Python::with_gil(|py| {
    //                 func.call1(py, (t, *y))
    //                     .map_err(|e| e.to_string())
    //                     .and_then(|val| val.extract::<Scalar>(py).map_err(|e| e.to_string()))
    //             })
    //         },
    //         &time,
    //         initial_condition,
    //     );
    //     result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    // }
    }
}
