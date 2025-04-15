use super::PyErrGlue;
use crate::constitutive::solid::{
    elastic::AlmansiHamel,
    hyperelastic::{NeoHookean, SaintVenantKirchhoff},
};
use conspire::{
    constitutive::solid::{
        elastic::AlmansiHamel as AlmansiHamelConspire,
        hyperelastic::{
            NeoHookean as NeoHookeanConspire, SaintVenantKirchhoff as SaintVenantKirchhoffConspire,
        },
    },
    fem::{
        Connectivity, ElasticFiniteElementBlock, ElementBlock, FiniteElementBlock,
        HyperelasticFiniteElementBlock, LinearTetrahedron, NodalCoordinatesBlock,
        ReferenceNodalCoordinatesBlock,
    },
    math::TensorVec,
    mechanics::Scalar,
};
use numpy::PyArray2;
use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Block>()
}

const N: usize = 4;

/// Finite element block.
#[pyclass]
enum Block {
    ElasticBlock(Py<ElasticBlock>),
    HyperelasticBlock(Py<HyperelasticBlock>),
}

#[derive(FromPyObject)]
enum Model<'py> {
    AlmansiHamel(Bound<'py, AlmansiHamel>),
    NeoHookean(Bound<'py, NeoHookean>),
    SaintVenantKirchhoff(Bound<'py, SaintVenantKirchhoff>),
}

#[pymethods]
impl Block {
    #[new]
    fn new(
        py: Python,
        model: Model,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Self, PyErr> {
        match model {
            Model::AlmansiHamel(model) => {
                let bulk_modulus: Scalar = model.getattr("bulk_modulus")?.extract()?;
                let shear_modulus: Scalar = model.getattr("shear_modulus")?.extract()?;
                let block = AlmansiHamelBlock::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::ElasticBlock(Py::new(
                    py,
                    ElasticBlock::AlmansiHamel(Py::new(py, block)?),
                )?))
            }
            Model::NeoHookean(model) => {
                let bulk_modulus: Scalar = model.getattr("bulk_modulus")?.extract()?;
                let shear_modulus: Scalar = model.getattr("shear_modulus")?.extract()?;
                let block = NeoHookeanBlock::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::HyperelasticBlock(Py::new(
                    py,
                    HyperelasticBlock::NeoHookean(Py::new(py, block)?),
                )?))
            }
            Model::SaintVenantKirchhoff(model) => {
                let bulk_modulus: Scalar = model.getattr("bulk_modulus")?.extract()?;
                let shear_modulus: Scalar = model.getattr("shear_modulus")?.extract()?;
                let block = SaintVenantKirchhoffBlock::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::HyperelasticBlock(Py::new(
                    py,
                    HyperelasticBlock::SaintVenantKirchhoff(Py::new(py, block)?),
                )?))
            }
        }
    }
    /// $$
    /// A = \int_\Omega a\,dV
    /// $$
    fn helmholtz_free_energy(
        &self,
        py: Python,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Scalar, PyErrGlue> {
        match self {
            Self::ElasticBlock(_) => Err(PyErrGlue::new(
                "The Helmholtz free energy density is undefined for elastic constitutive models.",
            )),
            Self::HyperelasticBlock(block) => Ok(block
                .call_method1(py, "helmholtz_free_energy", (nodal_coordinates,))
                .unwrap()
                .extract(py)
                .unwrap()),
        }
    }
    /// $$
    /// \mathbf{f}_a = \frac{\partial A}{\partial\mathbf{x}_a}
    /// $$
    fn nodal_forces<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        match self {
            Self::ElasticBlock(block) => Ok(block
                .call_method1(py, "nodal_forces", (nodal_coordinates,))
                .unwrap()
                .extract(py)
                .unwrap()),
            Self::HyperelasticBlock(block) => Ok(block
                .call_method1(py, "nodal_forces", (nodal_coordinates,))
                .unwrap()
                .extract(py)
                .unwrap()),
        }
    }
}

#[pyclass]
enum ElasticBlock {
    AlmansiHamel(Py<AlmansiHamelBlock>),
}

#[derive(FromPyObject)]
enum ElasticModel<'py> {
    AlmansiHamel(Bound<'py, AlmansiHamel>),
}

#[pymethods]
impl ElasticBlock {
    #[new]
    fn new(
        py: Python,
        model: ElasticModel,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Self, PyErr> {
        match model {
            ElasticModel::AlmansiHamel(model) => {
                let bulk_modulus: Scalar = model.getattr("bulk_modulus")?.extract()?;
                let shear_modulus: Scalar = model.getattr("shear_modulus")?.extract()?;
                let block = AlmansiHamelBlock::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::AlmansiHamel(Py::new(py, block)?))
            }
        }
    }
    fn nodal_forces<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        match self {
            Self::AlmansiHamel(model) => Ok(model
                .call_method1(py, "nodal_forces", (nodal_coordinates,))
                .unwrap()
                .extract(py)
                .unwrap()),
        }
    }
}

#[pyclass]
struct AlmansiHamelBlock {
    block: ElementBlock<LinearTetrahedron<AlmansiHamelConspire<[Scalar; 2]>>, N>,
}

#[pymethods]
impl AlmansiHamelBlock {
    #[new]
    fn new(
        bulk_modulus: Scalar,
        shear_modulus: Scalar,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Self {
        Self {
            block: ElementBlock::new(
                [bulk_modulus, shear_modulus],
                connectivity,
                ReferenceNodalCoordinatesBlock::new(&reference_nodal_coordinates),
            ),
        }
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
}

#[pyclass]
enum HyperelasticBlock {
    NeoHookean(Py<NeoHookeanBlock>),
    SaintVenantKirchhoff(Py<SaintVenantKirchhoffBlock>),
}

#[derive(FromPyObject)]
enum HyperelasticModel<'py> {
    NeoHookean(Bound<'py, NeoHookean>),
    SaintVenantKirchhoff(Bound<'py, SaintVenantKirchhoff>),
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
        match model {
            HyperelasticModel::NeoHookean(model) => {
                let bulk_modulus: Scalar = model.getattr("bulk_modulus")?.extract()?;
                let shear_modulus: Scalar = model.getattr("shear_modulus")?.extract()?;
                let block = NeoHookeanBlock::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::NeoHookean(Py::new(py, block)?))
            }
            HyperelasticModel::SaintVenantKirchhoff(model) => {
                let bulk_modulus: Scalar = model.getattr("bulk_modulus")?.extract()?;
                let shear_modulus: Scalar = model.getattr("shear_modulus")?.extract()?;
                let block = SaintVenantKirchhoffBlock::new(
                    bulk_modulus,
                    shear_modulus,
                    connectivity,
                    reference_nodal_coordinates,
                );
                Ok(Self::SaintVenantKirchhoff(Py::new(py, block)?))
            }
        }
    }
    fn helmholtz_free_energy(
        &self,
        py: Python,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Scalar, PyErrGlue> {
        match self {
            Self::NeoHookean(model) => Ok(model
                .call_method1(py, "helmholtz_free_energy", (nodal_coordinates,))
                .unwrap()
                .extract(py)
                .unwrap()),
            Self::SaintVenantKirchhoff(model) => Ok(model
                .call_method1(py, "helmholtz_free_energy", (nodal_coordinates,))
                .unwrap()
                .extract(py)
                .unwrap()),
        }
    }
    fn nodal_forces<'py>(
        &self,
        py: Python<'py>,
        nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        match self {
            Self::NeoHookean(model) => Ok(model
                .call_method1(py, "nodal_forces", (nodal_coordinates,))
                .unwrap()
                .extract(py)
                .unwrap()),
            Self::SaintVenantKirchhoff(model) => Ok(model
                .call_method1(py, "nodal_forces", (nodal_coordinates,))
                .unwrap()
                .extract(py)
                .unwrap()),
        }
    }
}

#[pyclass]
struct NeoHookeanBlock {
    block: ElementBlock<LinearTetrahedron<NeoHookeanConspire<[Scalar; 2]>>, N>,
}

#[pyclass]
struct SaintVenantKirchhoffBlock {
    block: ElementBlock<LinearTetrahedron<SaintVenantKirchhoffConspire<[Scalar; 2]>>, N>,
}

#[pymethods]
impl NeoHookeanBlock {
    #[new]
    fn new(
        bulk_modulus: Scalar,
        shear_modulus: Scalar,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Self {
        Self {
            block: ElementBlock::new(
                [bulk_modulus, shear_modulus],
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
}

#[pymethods]
impl SaintVenantKirchhoffBlock {
    #[new]
    fn new(
        bulk_modulus: Scalar,
        shear_modulus: Scalar,
        connectivity: Connectivity<N>,
        reference_nodal_coordinates: Vec<[Scalar; 3]>,
    ) -> Self {
        Self {
            block: ElementBlock::new(
                [bulk_modulus, shear_modulus],
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
}
