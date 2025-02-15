use conspire::math::special;
use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(langevin, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_langevin, m)?)
}

/// Returns the Langevin function.
///
/// $$
/// \mathcal{L}(x) = \coth(x) - x^{-1}
/// $$
#[pyfunction]
pub fn langevin(x: f64) -> f64 {
    special::langevin(x)
}

/// Returns the inverse Langevin function.
///
/// $$
/// x = \mathcal{L}^{-1}(y)
/// $$
#[pyfunction]
pub fn inverse_langevin(y: f64) -> f64 {
    special::inverse_langevin(y)
}
