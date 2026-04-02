use crate::PyErrGlue;
use conspire::{
    math::Scalar,
    physics::molecular::single_chain::{
        Ensemble, FreelyJointedChain as Inner, SingleChainError, Thermodynamics,
    },
};
use pyo3::prelude::*;

pub fn register_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FreelyJointedChain>()
}

impl From<SingleChainError> for PyErrGlue {
    fn from(error: SingleChainError) -> Self {
        PyErrGlue {
            message: error.to_string(),
        }
    }
}

#[pyclass]
pub struct FreelyJointedChain(Inner);

#[pymethods]
impl FreelyJointedChain {
    #[new]
    fn new(
        ensemble: String,
        link_length: Scalar,
        number_of_links: u8,
        temperature: Scalar,
    ) -> Self {
        let ensemble = match ensemble.as_str() {
            "isometric" => Ensemble::Isometric(temperature),
            "isotensional" => Ensemble::Isotensional(temperature),
            _ => panic!(),
        };
        Self(Inner {
            link_length,
            number_of_links,
            ensemble,
        })
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
        Ok(self
            .0
            .nondimensional_radial_distribution(nondimensional_extension)?)
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
        Ok(self
            .0
            .nondimensional_gibbs_free_energy(nondimensional_force)?)
    }
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, PyErrGlue> {
        Ok(self
            .0
            .nondimensional_gibbs_free_energy_per_link(nondimensional_force)?)
    }
    fn nondimensional_extension(&self, nondimensional_force: Scalar) -> Result<Scalar, PyErrGlue> {
        Ok(self.0.nondimensional_extension(nondimensional_force)?)
    }
    fn nondimensional_compliance(&self, nondimensional_force: Scalar) -> Result<Scalar, PyErrGlue> {
        Ok(self.0.nondimensional_compliance(nondimensional_force)?)
    }
}
