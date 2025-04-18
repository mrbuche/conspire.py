use crate::{
    PyErrGlue, constitutive::solid::hyperelastic as constitutive, count_tts, fem::call_method,
    replace_expr,
};
use conspire::{
    fem::{
        Connectivity, ElasticFiniteElementBlock, ElementBlock, FiniteElementBlock,
        HyperelasticFiniteElementBlock, LinearTetrahedron, NodalCoordinatesBlock,
        ReferenceNodalCoordinatesBlock,
    },
    math::TensorVec,
    mechanics::Scalar,
};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

const N: usize = 4;

#[pyclass]
pub enum HyperelasticBlock {
    ArrudaBoyce(Py<ArrudaBoyce>),
    Fung(Py<Fung>),
    Gent(Py<Gent>),
    MooneyRivlin(Py<MooneyRivlin>),
    NeoHookean(Py<NeoHookean>),
    SaintVenantKirchhoff(Py<SaintVenantKirchhoff>),
}

#[derive(FromPyObject)]
enum HyperelasticModel<'py> {
    ArrudaBoyce(Bound<'py, constitutive::ArrudaBoyce>),
    Fung(Bound<'py, constitutive::Fung>),
    Gent(Bound<'py, constitutive::Gent>),
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
            block: ElementBlock<$element<conspire::constitutive::solid::hyperelastic::$model<[Scalar; count_tts!($($parameter)?)]>>, $n>,
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
                    block: ElementBlock::new(
                        [$($parameter),+],
                        connectivity,
                        ReferenceNodalCoordinatesBlock::new(&reference_nodal_coordinates),
                    ),
                }
            }
            fn helmholtz_free_energy(
                &self,
                nodal_coordinates: Vec<[Scalar; 3]>,
            ) -> Result<Scalar, PyErrGlue> {
                Ok(self
                    .block
                    .helmholtz_free_energy(&NodalCoordinatesBlock::new(&nodal_coordinates))?)
            }
            fn nodal_forces<'py>(
                &self,
                py: Python<'py>,
                nodal_coordinates: Vec<[Scalar; 3]>,
            ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
                let forces: Vec<Vec<Scalar>> = self
                    .block
                    .nodal_forces(&NodalCoordinatesBlock::new(&nodal_coordinates))?
                    .into();
                Ok(PyArray2::from_vec2(py, &forces)?)
            }
            fn nodal_stiffnesses<'py>(
                &self,
                py: Python<'py>,
                nodal_coordinates: Vec<[Scalar; 3]>,
            ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
                let stiffnesses: Vec<Scalar> = self
                    .block
                    .nodal_stiffnesses(&NodalCoordinatesBlock::new(&nodal_coordinates))?
                    .into();
                let nodes = nodal_coordinates.len();
                Ok(PyArray4::from_array(
                    py,
                    &Array::from_shape_vec((nodes, nodes, 3, 3), stiffnesses)?,
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
