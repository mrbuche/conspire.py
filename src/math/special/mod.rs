use conspire::math::{Vector, special};
use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lambert_w, m)?)?;
    m.add_function(wrap_pyfunction!(langevin, m)?)?;
    m.add_function(wrap_pyfunction!(langevin_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_langevin, m)?)?;
    m.add_function(wrap_pyfunction!(rosenbrock, m)?)?;
    m.add_function(wrap_pyfunction!(sinhc, m)?)
}

/// Returns the Lambert W function.
///
/// $$
/// y = W_0(x)
/// $$
#[pyfunction]
pub fn lambert_w(x: f64) -> f64 {
    special::lambert_w(x)
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

/// Returns the derivative of the Langevin function.
///
/// $$
/// \mathcal{L}'(x) = x^{-2} - \sinh^{-2}(x)
/// $$
#[pyfunction]
pub fn langevin_derivative(x: f64) -> f64 {
    special::langevin_derivative(x)
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

/// Returns the Rosenbrock function.
///
/// $$
/// f(\mathbf{x}) = \sum_{i=1}^{N-1} \left[\left(a - x_i\right)^2 + b\left(x_{i+1} - x_i^2\right)^2\right]
/// $$
#[pyfunction]
pub fn rosenbrock(x: Vec<f64>, a: f64, b: f64) -> f64 {
    special::rosenbrock(&Vector::from(x), a, b)
}

/// Returns the hyperbolic sinc function.
///
/// $$
/// \mathrm{sinhc}(x) = \frac{\sinh(x)}{x}
/// $$
#[pyfunction]
pub fn sinhc(x: f64) -> f64 {
    special::sinhc(x)
}
