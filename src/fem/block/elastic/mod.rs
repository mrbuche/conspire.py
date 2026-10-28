use crate::{
    PyErrGlue,
    constitutive::solid::elastic as constitutive,
    fem::call_method,
    units::{PyForce, PyForcePerLength, PyLength, PyStress},
};
use conspire::{
    fem::{
        block::{Block, element::linear::Tetrahedron as LinearTetrahedron},
        solid::elastic::ElasticElements,
    },
    mechanics::Scalar,
};
use pyo3::prelude::*;

pub const G: usize = 1;
pub const M: usize = 3;
pub const N: usize = 4;
pub const P: usize = N;

macro_rules! elastic_blocks {
    ($($name: ident [$($parameter: ident : $ty: ty),+]);+ $(;)?) => {
        #[pyclass]
        pub enum ElasticBlock {
            $($name(Py<$name>)),+
        }

        #[derive(FromPyObject)]
        pub enum ElasticModel<'py> {
            $($name(Bound<'py, constitutive::$name>)),+
        }

        #[pymethods]
        impl ElasticBlock {
            #[new]
            pub fn new(
                py: Python,
                model: ElasticModel,
                connectivity: Vec<[usize; N]>,
                reference_nodal_coordinates: PyLength,
            ) -> Result<Self, PyErr> {
                match model {
                    $(ElasticModel::$name(model) => Ok(Self::$name(Py::new(
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
                block: Block<
                    conspire::constitutive::solid::elastic::$name,
                    LinearTetrahedron<G>,
                    G,
                    M,
                    N,
                    P,
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
                    Ok(Self {
                        block: (
                            conspire::constitutive::solid::elastic::$name {
                                $($parameter: $parameter.try_into()?),+
                            },
                            connectivity,
                            &(&reference_nodal_coordinates).try_into()?,
                        )
                            .into(),
                    })
                }
                fn nodal_forces(&self, nodal_coordinates: PyLength) -> Result<PyForce, PyErrGlue> {
                    Ok(self
                        .block
                        .nodal_forces(&(&nodal_coordinates).try_into()?)?
                        .into())
                }
                fn nodal_stiffnesses(
                    &self,
                    nodal_coordinates: PyLength,
                ) -> Result<PyForcePerLength, PyErrGlue> {
                    Ok((&self
                        .block
                        .nodal_stiffnesses(&(&nodal_coordinates).try_into()?)?)
                        .into())
                }
            }
        )+
    };
}

elastic_blocks!(
    AlmansiHamelEulerian [bulk_modulus: PyStress, shear_modulus: PyStress];
    AlmansiHamelLagrangian [bulk_modulus: PyStress, shear_modulus: PyStress];
    BazantItskovEulerian [bulk_modulus: PyStress, shear_modulus: PyStress, exponent: Scalar];
    BazantItskovLagrangian [bulk_modulus: PyStress, shear_modulus: PyStress, exponent: Scalar];
    Hencky [bulk_modulus: PyStress, shear_modulus: PyStress];
    SaintVenantKirchhoff [bulk_modulus: PyStress, shear_modulus: PyStress];
    SethHillEulerian [bulk_modulus: PyStress, shear_modulus: PyStress, exponent: Scalar];
    SethHillLagrangian [bulk_modulus: PyStress, shear_modulus: PyStress, exponent: Scalar];
);
