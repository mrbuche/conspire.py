use crate::{
    PyErrGlue,
    constitutive::solid::hyperelastic as constitutive,
    fem::{
        call_method,
        model::elastic::{G, M, N, P},
    },
    units::{PyEnergy, PyForce, PyForcePerLength, PyLength, PyStress},
};
use conspire::{
    fem::{
        Model as FemModel, NodalReferenceCoordinates,
        block::{Block, element::linear::Tetrahedron},
        solid::{elastic::ElasticElements, hyperelastic::HyperelasticElements},
    },
    mechanics::Scalar,
};
use pyo3::prelude::*;

macro_rules! hyperelastic_models {
    ($($name: ident [$($parameter: ident : $ty: ty),+]);+ $(;)?) => {
        #[pyclass]
        pub enum HyperelasticModel {
            $($name(Py<$name>)),+
        }

        #[derive(FromPyObject)]
        pub enum HyperelasticConstitutive<'py> {
            $($name(Bound<'py, constitutive::$name>)),+
        }

        #[pymethods]
        impl HyperelasticModel {
            #[new]
            pub fn new(
                py: Python,
                constitutive_model: HyperelasticConstitutive,
                connectivity: Vec<[usize; N]>,
                reference_nodal_coordinates: PyLength,
            ) -> Result<Self, PyErr> {
                match constitutive_model {
                    $(HyperelasticConstitutive::$name(model) => Ok(Self::$name(Py::new(
                        py,
                        $name::new(
                            $(model.getattr(stringify!($parameter))?.extract()?),+,
                            connectivity,
                            reference_nodal_coordinates,
                        )?,
                    )?))),+
                }
            }
            fn helmholtz_free_energy(
                &self,
                py: Python,
                nodal_coordinates: PyLength,
            ) -> Result<PyEnergy, PyErrGlue> {
                match self {
                    $(Self::$name(model) => call_method!(model, py, "helmholtz_free_energy", nodal_coordinates)),+
                }
            }
            fn nodal_forces(
                &self,
                py: Python,
                nodal_coordinates: PyLength,
            ) -> Result<PyForce, PyErrGlue> {
                match self {
                    $(Self::$name(model) => call_method!(model, py, "nodal_forces", nodal_coordinates)),+
                }
            }
            fn nodal_stiffnesses(
                &self,
                py: Python,
                nodal_coordinates: PyLength,
            ) -> Result<PyForcePerLength, PyErrGlue> {
                match self {
                    $(Self::$name(model) => call_method!(model, py, "nodal_stiffnesses", nodal_coordinates)),+
                }
            }
        }

        $(
            #[pyclass(unsendable)]
            pub struct $name {
                model: FemModel<
                    Block<
                        conspire::constitutive::solid::hyperelastic::$name,
                        Tetrahedron<G>,
                        G,
                        M,
                        N,
                        P,
                    >,
                    M,
                >,
            }
            #[pymethods]
            impl $name {
                #[new]
                pub fn new(
                    $($parameter: $ty),+,
                    connectivity: Vec<[usize; N]>,
                    reference_nodal_coordinates: PyLength,
                ) -> PyResult<Self> {
                    let coordinates = NodalReferenceCoordinates::try_from(&reference_nodal_coordinates)?;
                    let block = (
                        conspire::constitutive::solid::hyperelastic::$name {
                            $($parameter: $parameter.try_into()?),+
                        },
                        connectivity,
                        &coordinates,
                    )
                        .into();
                    Ok(Self {
                        model: (block, coordinates).into(),
                    })
                }
                fn helmholtz_free_energy(
                    &self,
                    nodal_coordinates: PyLength,
                ) -> Result<PyEnergy, PyErrGlue> {
                    Ok(self
                        .model
                        .helmholtz_free_energy(&(&nodal_coordinates).try_into()?)?
                        .into())
                }
                fn nodal_forces(&self, nodal_coordinates: PyLength) -> Result<PyForce, PyErrGlue> {
                    Ok(self
                        .model
                        .nodal_forces(&(&nodal_coordinates).try_into()?)?
                        .into())
                }
                fn nodal_stiffnesses(
                    &self,
                    nodal_coordinates: PyLength,
                ) -> Result<PyForcePerLength, PyErrGlue> {
                    Ok((&self
                        .model
                        .nodal_stiffnesses(&(&nodal_coordinates).try_into()?)?)
                        .into())
                }
            }
        )+
    };
}

hyperelastic_models!(
    ArrudaBoyce [bulk_modulus: PyStress, shear_modulus: PyStress, number_of_links: Scalar];
    BlatzKo [bulk_modulus: PyStress, shear_modulus: PyStress, mixing_parameter: Scalar];
    Carroll [bulk_modulus: PyStress, linear_modulus: PyStress, quartic_modulus: PyStress, second_invariant_modulus: PyStress];
    Fung [bulk_modulus: PyStress, shear_modulus: PyStress, extra_modulus: PyStress, exponent: Scalar];
    Gent [bulk_modulus: PyStress, shear_modulus: PyStress, extensibility: Scalar];
    Hencky [bulk_modulus: PyStress, shear_modulus: PyStress];
    Isihara [bulk_modulus: PyStress, shear_modulus: PyStress, extra_modulus: PyStress, quadratic_modulus: PyStress];
    MooneyRivlin [bulk_modulus: PyStress, shear_modulus: PyStress, extra_modulus: PyStress];
    NeoHookean [bulk_modulus: PyStress, shear_modulus: PyStress];
    Ogden [bulk_modulus: PyStress, shear_moduli: PyStress, exponents: Vec<Scalar>];
    SaintVenantKirchhoff [bulk_modulus: PyStress, shear_modulus: PyStress];
    Yeoh [bulk_modulus: PyStress, shear_moduli: PyStress];
);
