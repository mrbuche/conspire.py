use crate::{
    PyErrGlue,
    constitutive::solid::hyperelastic as constitutive,
    fem::{
        block::elastic::{G, M, N, P},
        call_method,
    },
};
use conspire::{
    fem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::{
            Block, Connectivity,
            element::linear::Tetrahedron as LinearTetrahedron,
            solid::{
                elastic::ElasticFiniteElementBlock, hyperelastic::HyperelasticFiniteElementBlock,
            },
        },
    },
    mechanics::Scalar,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

#[pyclass]
pub enum HyperelasticBlock {
    ArrudaBoyce(Py<ArrudaBoyce>),
    Fung(Py<Fung>),
    Gent(Py<Gent>),
    Hencky(Py<Hencky>),
    MooneyRivlin(Py<MooneyRivlin>),
    NeoHookean(Py<NeoHookean>),
    SaintVenantKirchhoff(Py<SaintVenantKirchhoff>),
}

#[derive(FromPyObject)]
enum HyperelasticModel<'py> {
    ArrudaBoyce(Bound<'py, constitutive::ArrudaBoyce>),
    Fung(Bound<'py, constitutive::Fung>),
    Gent(Bound<'py, constitutive::Gent>),
    Hencky(Bound<'py, constitutive::Hencky>),
    MooneyRivlin(Bound<'py, constitutive::MooneyRivlin>),
    NeoHookean(Bound<'py, constitutive::NeoHookean>),
    SaintVenantKirchhoff(Bound<'py, constitutive::SaintVenantKirchhoff>),
}

macro_rules! match_model {
    ($self: ident, $py: ident, $name: literal, $nodal_coordinates: ident) => {
        match $self {
            Self::ArrudaBoyce(model) => call_method!(model, $py, $name, $nodal_coordinates),
            Self::Fung(model) => call_method!(model, $py, $name, $nodal_coordinates),
            Self::Gent(model) => call_method!(model, $py, $name, $nodal_coordinates),
            Self::Hencky(model) => call_method!(model, $py, $name, $nodal_coordinates),
            Self::MooneyRivlin(model) => call_method!(model, $py, $name, $nodal_coordinates),
            Self::NeoHookean(model) => call_method!(model, $py, $name, $nodal_coordinates),
            Self::SaintVenantKirchhoff(model) => {
                call_method!(model, $py, $name, $nodal_coordinates)
            }
        }
    };
}

macro_rules! hyperelastic_block_inner {
    ($py: ident, $model: ident, $name: ident, $connectivity: ident, $reference_nodal_coordinates: ident, $($parameter: expr),+ $(,)?) => {
        Ok(Self::$name(Py::new($py, $name::new(
            $($model.getattr(stringify!($parameter))?.extract()?),+,
            $connectivity,
            $reference_nodal_coordinates,
        ))?))
    }
}

macro_rules! hyperelastic_block {
    ($py: ident, $model: ident, $connectivity: ident, $reference_nodal_coordinates: ident, $($name: ident, [$($parameter: expr),+]),+ $(,)?) => {
        match $model {
            $(HyperelasticModel::$name(model) => hyperelastic_block_inner!($py, model, $name, $connectivity, $reference_nodal_coordinates, $($parameter),+)),+
        }
    }
}

#[pymethods]
impl HyperelasticBlock {
    #[new]
    fn new(
        py: Python,
        model: HyperelasticModel,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Self, PyErr> {
        hyperelastic_block!(
            py,
            model,
            connectivity,
            reference_nodal_coordinates,
            ArrudaBoyce,
            [bulk_modulus, shear_modulus, number_of_links],
            Fung,
            [bulk_modulus, shear_modulus, extra_modulus, exponent],
            Gent,
            [bulk_modulus, shear_modulus, extensibility],
            Hencky,
            [bulk_modulus, shear_modulus],
            MooneyRivlin,
            [bulk_modulus, shear_modulus, extra_modulus],
            NeoHookean,
            [bulk_modulus, shear_modulus],
            SaintVenantKirchhoff,
            [bulk_modulus, shear_modulus],
        )
    }
    fn helmholtz_free_energy(
        &self,
        py: Python,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Scalar, PyErrGlue> {
        match_model!(self, py, "helmholtz_free_energy", nodal_coordinates)
    }
    fn nodal_forces<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        match_model!(self, py, "nodal_forces", nodal_coordinates)
    }
    fn nodal_stiffnesses<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
        match_model!(self, py, "nodal_stiffnesses", nodal_coordinates)
    }
}

macro_rules! hyperelastic {
    ($element: ident, $n: literal, $model: ident, $($parameter: ident),+ $(,)?) => {
        #[pyclass]
        pub struct $model {
            block: Block<conspire::constitutive::solid::hyperelastic::$model, $element, G, M, $n, P>,
        }
        #[pymethods]
        impl $model {
            #[new]
            pub fn new(
                $($parameter: Scalar),+,
                connectivity: Connectivity<$n>,
                reference_nodal_coordinates: Vec<[Scalar; 3]>,
            ) -> Self {
                Self {
                    block: Block::from((
                        conspire::constitutive::solid::hyperelastic::$model {
                            $($parameter),+
                        },
                        connectivity,
                        NodalReferenceCoordinates::from(reference_nodal_coordinates),
                    )),
                }
            }
            fn helmholtz_free_energy(
                &self,
                nodal_coordinates: Vec<[Scalar; 3]>,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(self
                    .block
                    .helmholtz_free_energy(&NodalCoordinates::from(nodal_coordinates))?)
            }
            fn nodal_forces<'py>(
                &self,
                py: Python<'py>,
                nodal_coordinates: Vec<[Scalar; 3]>,
            ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
                let forces: Vec<Vec<Scalar>> = self
                    .block
                    .nodal_forces(&NodalCoordinates::from(nodal_coordinates))?
                    .into();
                Ok(PyArray2::from_vec2(py, &forces)?)
            }
            fn nodal_stiffnesses<'py>(
                &self,
                py: Python<'py>,
                nodal_coordinates: Vec<[Scalar; 3]>,
            ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
                let nodes = nodal_coordinates.len();
                Ok(PyArray4::from_owned_array(
                    py,
                    Array::from_shape_vec(
                        (nodes, nodes, 3, 3),
                        self.block
                            .nodal_stiffnesses(&NodalCoordinates::from(nodal_coordinates))?
                            .into(),
                    )?,
                ))
            }
        }
    };
}

hyperelastic!(
    LinearTetrahedron,
    4,
    ArrudaBoyce,
    bulk_modulus,
    shear_modulus,
    number_of_links,
);
hyperelastic!(
    LinearTetrahedron,
    4,
    Fung,
    bulk_modulus,
    shear_modulus,
    extra_modulus,
    exponent,
);
hyperelastic!(
    LinearTetrahedron,
    4,
    Gent,
    bulk_modulus,
    shear_modulus,
    extensibility,
);
hyperelastic!(LinearTetrahedron, 4, Hencky, bulk_modulus, shear_modulus,);
hyperelastic!(
    LinearTetrahedron,
    4,
    MooneyRivlin,
    bulk_modulus,
    shear_modulus,
    extra_modulus,
);
hyperelastic!(
    LinearTetrahedron,
    4,
    NeoHookean,
    bulk_modulus,
    shear_modulus,
);
hyperelastic!(
    LinearTetrahedron,
    4,
    SaintVenantKirchhoff,
    bulk_modulus,
    shear_modulus,
);
