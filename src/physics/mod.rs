use crate::PyErrGlue;
use conspire::{
    math::Scalar,
    physics::molecular::single_chain::{
        // ArbitraryPotentialFreelyJointedChain as Ufjc,
        Ensemble,
        // ExtensibleFreelyJointedChain as Efjc,
        FreelyJointedChain as Fjc,
        MonteCarloInextensible,
        // IdealChain as Ideal,
        SingleChainError,
        SquareWellFreelyJointedChain as Swfjc,
        Thermodynamics,
    },
};
use numpy::PyArray1;
use pyo3::prelude::*;

pub fn register_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<ArbitraryPotentialFreelyJointedChain>()?;
    // m.add_class::<ExtensibleFreelyJointedChain>()?;
    m.add_class::<FreelyJointedChain>()?;
    // m.add_class::<IdealChain>()?;
    m.add_class::<SquareWellFreelyJointedChain>()
}

impl From<SingleChainError> for PyErrGlue {
    fn from(error: SingleChainError) -> Self {
        PyErrGlue {
            message: error.to_string(),
        }
    }
}

macro_rules! single_chain {
    ($model:ident, $inner:ident, $($parameter: ident),+ $(,)?) => {
        #[pyclass]
        pub struct $model($inner);
        #[pymethods]
        impl $model {
            #[new]
            fn new(
                number_of_links: u8,
                $($parameter: Scalar),+,
                ensemble: String,
                temperature: Scalar,
            ) -> Self {
                let ensemble = match ensemble.as_str() {
                    "isometric" => Ensemble::Isometric(temperature),
                    "isotensional" => Ensemble::Isotensional(temperature),
                    _ => panic!(),
                };
                Self($inner {
                    number_of_links,
                    $($parameter),+,
                    ensemble,
                })
            }
            /// @private
            #[getter]
            pub fn number_of_links(&self) -> u8 {
                self.0.number_of_links
            }
            $(
                /// @private
                #[getter]
                pub fn $parameter(&self) -> Scalar {
                    self.0.$parameter
                }
            )+
            fn nondimensional_helmholtz_free_energy(
                &self,
                nondimensional_extension: Scalar,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(self
                    .0
                    .nondimensional_helmholtz_free_energy(nondimensional_extension)?)
            }
            fn nondimensional_helmholtz_free_energy_per_link(
                &self,
                nondimensional_extension: Scalar,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(self
                    .0
                    .nondimensional_helmholtz_free_energy_per_link(nondimensional_extension)?)
            }
            fn nondimensional_force(&self, nondimensional_extension: Scalar) -> Result<Scalar, PyErrGlue> {
                Ok(self.0.nondimensional_force(nondimensional_extension)?)
            }
            fn nondimensional_stiffness(
                &self,
                nondimensional_extension: Scalar,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(self.0.nondimensional_stiffness(nondimensional_extension)?)
            }
            fn nondimensional_radial_distribution(
                &self,
                nondimensional_extension: Scalar,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(Thermodynamics::nondimensional_radial_distribution(&self.0, nondimensional_extension)?)
            }
            fn nondimensional_radial_distribution_monte_carlo<'py>(
                &self,
                py: Python<'py>,
                nondimensional_force: Scalar,
                num_bins: usize,
                num_samples: usize,
                num_threads: usize,
            ) -> (Bound<'py, PyArray1<Scalar>>, Bound<'py, PyArray1<Scalar>>) {
                let (g, p) = MonteCarloInextensible::nondimensional_radial_distribution(
                    &self.0,
                    nondimensional_force,
                    num_bins,
                    num_samples,
                    num_threads,
                );
                (PyArray1::from_vec(py, Vec::from(g)), PyArray1::from_vec(py, Vec::from(p)))
            }
            fn nondimensional_longitudinal_distribution_monte_carlo<'py>(
                &self,
                py: Python<'py>,
                nondimensional_force: Scalar,
                num_bins: usize,
                num_samples: usize,
                num_threads: usize,
            ) -> (Bound<'py, PyArray1<Scalar>>, Bound<'py, PyArray1<Scalar>>) {
                let (g, p) = MonteCarloInextensible::nondimensional_longitudinal_distribution(
                    &self.0,
                    nondimensional_force,
                    num_bins,
                    num_samples,
                    num_threads,
                );
                (PyArray1::from_vec(py, Vec::from(g)), PyArray1::from_vec(py, Vec::from(p)))
            }
            fn nondimensional_transverse_distribution_monte_carlo<'py>(
                &self,
                py: Python<'py>,
                nondimensional_force: Scalar,
                num_bins: usize,
                num_samples: usize,
                num_threads: usize,
            ) -> (Bound<'py, PyArray1<Scalar>>, Bound<'py, PyArray1<Scalar>>) {
                let (g, p) = MonteCarloInextensible::nondimensional_transverse_distribution(
                    &self.0,
                    nondimensional_force,
                    num_bins,
                    num_samples,
                    num_threads,
                );
                (PyArray1::from_vec(py, Vec::from(g)), PyArray1::from_vec(py, Vec::from(p)))
            }
            fn nondimensional_spherical_distribution(
                &self,
                nondimensional_extension: Scalar,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(self
                    .0
                    .nondimensional_spherical_distribution(nondimensional_extension)?)
            }
            fn nondimensional_gibbs_free_energy(
                &self,
                nondimensional_force: Scalar,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(Thermodynamics::nondimensional_gibbs_free_energy(&self.0, nondimensional_force)?)
            }
            fn nondimensional_gibbs_free_energy_per_link(
                &self,
                nondimensional_force: Scalar,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(Thermodynamics::nondimensional_gibbs_free_energy_per_link(&self.0, nondimensional_force)?)
            }
            fn nondimensional_extension(&self, nondimensional_force: Scalar) -> Result<Scalar, PyErrGlue> {
                Ok(Thermodynamics::nondimensional_extension(&self.0, nondimensional_force)?)
            }
            fn nondimensional_compliance(&self, nondimensional_force: Scalar) -> Result<Scalar, PyErrGlue> {
                Ok(Thermodynamics::nondimensional_compliance(&self.0, nondimensional_force)?)
            }
        }
    };
}

// single_chain!(
//     ExtensibleFreelyJointedChain,
//     Efjc,
//     link_length,
//     link_stiffness
// );
// single_chain!(IdealChain, Ideal, link_length);
single_chain!(FreelyJointedChain, Fjc, link_length);
single_chain!(SquareWellFreelyJointedChain, Swfjc, link_length, well_width);
