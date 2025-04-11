use conspire::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{ElementBlock, FiniteElementBlock, LinearTetrahedron},
    math::{TensorRank1Vec, TensorVec},
};
use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Block>()
    // let submodule_special = PyModule::new(py, "special")?;
    // submodule_special.setattr("__doc__", "Special functions.\n\n")?;
    // m.add_submodule(&submodule_special)
    // special::register_module(&submodule_special)?;
    // py.import("sys")?
    //     .getattr("modules")?
    //     .set_item("conspire.math.special", submodule_special)
}

// Storing the full data is going seems impossible since parameters is a shared reference
//
// /// ???
// #[pyclass]
// pub struct Block {
//     data: ElementBlock<LinearTetrahedron<NeoHookean<'static>>, 4>
// }

// #[pymethods]
// impl Block {
//     #[new]
//     fn new(
//         constitutive_model_parameters: Vec<f64>,
//         connectivity: Vec<[usize; 4]>,
//         reference_nodal_coordinates: Vec<[f64; 3]>,
//     ) -> Self {
//         Self {
//             data: ElementBlock::<LinearTetrahedron<NeoHookean>, 4>::new(
//                 &constitutive_model_parameters,
//                 connectivity,
//                 TensorRank1Vec::new(&reference_nodal_coordinates),
//             )
//         }
//     }
// }

/// ???
#[pyclass]
pub struct Block {
    connectivity: Vec<[usize; 4]>,
    elements: Vec<LinearTetrahedron>,
    reference_nodal_coordinates: Vec<[f64; 3]>
}

#[pymethods]
impl Block {
    #[new]
    fn new(
        connectivity: Vec<[usize; 4]>,
        reference_nodal_coordinates: Vec<[f64; 3]>,
    ) -> Self {
        Self {
            connectivity,
            reference_nodal_coordinates,
        }
    }
}
