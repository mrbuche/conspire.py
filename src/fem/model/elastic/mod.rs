use crate::{
    PyErrGlue,
    constitutive::solid::elastic as constitutive,
    fem::call_method,
    units::{PyForce, PyForcePerLength, PyLength, PyStress},
};
use conspire::{
    fem::{
        Model as FemModel, NodalReferenceCoordinates,
        block::{Block, element::linear::Tetrahedron},
        solid::elastic::ElasticElements,
    },
    mechanics::Scalar,
};
use pyo3::prelude::*;

pub const G: usize = 1;
pub const M: usize = 3;
pub const N: usize = 4;
pub const P: usize = N;

macro_rules! elastic_models {
    ($($name: ident [$($parameter: ident : $ty: ty),+]);+ $(;)?) => {
        #[pyclass]
        pub enum ElasticModel {
            $($name(Py<$name>)),+
        }

        #[derive(FromPyObject)]
        pub enum ElasticConstitutive<'py> {
            $($name(Bound<'py, constitutive::$name>)),+
        }

        #[pymethods]
        impl ElasticModel {
            #[new]
            pub fn new(
                py: Python,
                constitutive_model: ElasticConstitutive,
                connectivity: Vec<[usize; N]>,
                reference_nodal_coordinates: PyLength,
            ) -> Result<Self, PyErr> {
                match constitutive_model {
                    $(ElasticConstitutive::$name(model) => Ok(Self::$name(Py::new(
                        py,
                        $name::new(
                            $(model.getattr(stringify!($parameter))?.extract()?),+,
                            connectivity,
                            reference_nodal_coordinates,
                        )?,
                    )?))),+
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
                        conspire::constitutive::solid::elastic::$name,
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
                        conspire::constitutive::solid::elastic::$name {
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

elastic_models!(
    AlmansiHamelEulerian [bulk_modulus: PyStress, shear_modulus: PyStress];
    AlmansiHamelLagrangian [bulk_modulus: PyStress, shear_modulus: PyStress];
    BazantItskovEulerian [bulk_modulus: PyStress, shear_modulus: PyStress, exponent: Scalar];
    BazantItskovLagrangian [bulk_modulus: PyStress, shear_modulus: PyStress, exponent: Scalar];
    Hencky [bulk_modulus: PyStress, shear_modulus: PyStress];
    SaintVenantKirchhoff [bulk_modulus: PyStress, shear_modulus: PyStress];
    SethHillEulerian [bulk_modulus: PyStress, shear_modulus: PyStress, exponent: Scalar];
    SethHillLagrangian [bulk_modulus: PyStress, shear_modulus: PyStress, exponent: Scalar];
);
