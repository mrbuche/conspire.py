use crate::PyErrGlue;
use conspire::{
    math::Scalar,
    physics::molecular::{
        potential::Harmonic,
        single_chain::{
            ArbitraryPotentialFreelyJointedChain as Ufjc,
            Ensemble,
            ExtensibleFreelyJointedChain as Efjc,
            FreelyJointedChain as Fjc,
            MonteCarloInextensible,
            // IdealChain as Ideal,
            SingleChainError,
            SquareWellFreelyJointedChain as Swfjc,
            Thermodynamics,
            ThermodynamicsExtensible,
        },
    },
};
use numpy::{FromVecError, PyArray1, PyArray2};
use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Potential>()?;
    m.add_class::<ArbitraryPotentialFreelyJointedChain>()?;
    m.add_class::<ExtensibleFreelyJointedChain>()?;
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
            fn cosine_powers_monte_carlo<'py>(
                &self,
                py: Python<'py>,
                nondimensional_force: Scalar,
                num_powers: usize,
                num_samples: usize,
                num_threads: usize,
            ) -> Result<Bound<'py, PyArray2<Scalar>>, FromVecError> {
                PyArray2::from_vec2(py, &Vec::from(MonteCarloInextensible::cosine_powers(
                    &self.0,
                    nondimensional_force,
                    num_powers,
                    num_samples,
                    num_threads,
                )))
            }
            fn nondimensional_angular_distribution_monte_carlo<'py>(
                &self,
                py: Python<'py>,
                nondimensional_force: Scalar,
                num_bins: usize,
                num_samples: usize,
                num_threads: usize,
            ) -> (Bound<'py, PyArray1<Scalar>>, Bound<'py, PyArray1<Scalar>>) {
                let (g, p) = MonteCarloInextensible::nondimensional_angular_distribution(
                    &self.0,
                    nondimensional_force,
                    num_bins,
                    num_samples,
                    num_threads,
                );
                (PyArray1::from_vec(py, Vec::from(g)), PyArray1::from_vec(py, Vec::from(p)))
            }
            fn nondimensional_lateral_distribution_monte_carlo<'py>(
                &self,
                py: Python<'py>,
                nondimensional_force: Scalar,
                num_bins: usize,
                num_samples: usize,
                num_threads: usize,
            ) -> (Bound<'py, PyArray1<Scalar>>, Bound<'py, PyArray1<Scalar>>) {
                let (g, p) = MonteCarloInextensible::nondimensional_lateral_distribution(
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

#[pyclass]
pub struct ExtensibleFreelyJointedChain(Efjc);

#[pymethods]
impl ExtensibleFreelyJointedChain {
    #[new]
    fn new(
        number_of_links: u8,
        link_length: Scalar,
        link_stiffness: Scalar,
        ensemble: String,
        temperature: Scalar,
    ) -> Self {
        let ensemble = match ensemble.as_str() {
            "isometric" => Ensemble::Isometric(temperature),
            "isotensional" => Ensemble::Isotensional(temperature),
            _ => panic!(),
        };
        Self(Efjc {
            number_of_links,
            link_length,
            link_stiffness,
            ensemble,
        })
    }
    /// @private
    #[getter]
    pub fn number_of_links(&self) -> u8 {
        self.0.number_of_links
    }
    /// @private
    #[getter]
    pub fn link_length(&self) -> Scalar {
        self.0.link_length
    }
    /// @private
    #[getter]
    pub fn link_stiffness(&self) -> Scalar {
        self.0.link_stiffness
    }
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
        Ok(Thermodynamics::nondimensional_radial_distribution(
            &self.0,
            nondimensional_extension,
        )?)
    }
    fn nondimensional_link_energy_average(
        &self,
        nondimensional_value: Scalar,
    ) -> Result<Scalar, PyErrGlue> {
        Ok(ThermodynamicsExtensible::nondimensional_link_energy_average(
            &self.0,
            nondimensional_value,
        )?)
    }
    fn nondimensional_link_energy_variance(
        &self,
        nondimensional_value: Scalar,
    ) -> Result<Scalar, PyErrGlue> {
        Ok(
            ThermodynamicsExtensible::nondimensional_link_energy_variance(
                &self.0,
                nondimensional_value,
            )?,
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
enum Potential {
    Harmonic {
        rest_length: Scalar,
        stiffness: Scalar,
    },
    // Morse {
    //     rest_length: Scalar,
    //     depth: Scalar,
    //     parameter: Scalar,
    // },
}

#[pyclass]
pub struct ArbitraryPotentialFreelyJointedChain {
    number_of_links: u8,
    potential: Potential,
    ensemble: Ensemble,
}

#[pymethods]
impl ArbitraryPotentialFreelyJointedChain {
    #[new]
    fn new(
        number_of_links: u8,
        potential: Potential,
        ensemble: String,
        temperature: Scalar,
    ) -> Self {
        let ensemble = match ensemble.as_str() {
            "isometric" => Ensemble::Isometric(temperature),
            "isotensional" => Ensemble::Isotensional(temperature),
            _ => panic!(),
        };
        Self {
            number_of_links,
            potential,
            ensemble,
        }
    }
    /// @private
    #[getter]
    pub fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
    fn nondimensional_link_energy_average(
        &self,
        nondimensional_value: Scalar,
    ) -> Result<Scalar, PyErrGlue> {
        match self.potential.clone() {
            Potential::Harmonic {
                rest_length,
                stiffness,
            } => Ok(ThermodynamicsExtensible::nondimensional_link_energy_average(
                &Ufjc {
                    number_of_links: self.number_of_links,
                    link_potential: Harmonic {
                        rest_length,
                        stiffness,
                    },
                    ensemble: self.ensemble,
                },
                nondimensional_value,
            )?),
        }
    }
    fn nondimensional_link_energy_variance(
        &self,
        nondimensional_value: Scalar,
    ) -> Result<Scalar, PyErrGlue> {
        match self.potential.clone() {
            Potential::Harmonic {
                rest_length,
                stiffness,
            } => Ok(ThermodynamicsExtensible::nondimensional_link_energy_variance(
                &Ufjc {
                    number_of_links: self.number_of_links,
                    link_potential: Harmonic {
                        rest_length,
                        stiffness,
                    },
                    ensemble: self.ensemble,
                },
                nondimensional_value,
            )?),
        }
    }
}
