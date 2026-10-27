mod integrate;
mod special;

use crate::PyErrGlue;
use conspire::math::{Scalar, Tensor};
use ndarray::{ArrayD, IxDyn};
use pyo3::{Bound, prelude::*};

pub struct PyTensorRank2<T: Tensor>(T);

impl<T: Tensor> From<T> for PyTensorRank2<T> {
    fn from(tensor: T) -> Self {
        Self(tensor)
    }
}

impl<T: Tensor> PyTensorRank2<T>
where
    Vec<Vec<Scalar>>: From<T>,
{
    pub fn into_array(self) -> Result<ArrayD<Scalar>, PyErrGlue> {
        Ok(ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            Vec::<Vec<Scalar>>::from(self.0)
                .into_iter()
                .flatten()
                .collect(),
        )?)
    }
}

pub struct PyTensorRank4<T: Tensor>(T);

impl<T: Tensor> From<T> for PyTensorRank4<T> {
    fn from(tensor: T) -> Self {
        Self(tensor)
    }
}

impl<T: Tensor> PyTensorRank4<T>
where
    Vec<Scalar>: From<T>,
{
    pub fn into_array(self) -> Result<ArrayD<Scalar>, PyErrGlue> {
        Ok(ArrayD::from_shape_vec(
            IxDyn(&[3, 3, 3, 3]),
            Vec::<Scalar>::from(self.0),
        )?)
    }
}

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_integrate = PyModule::new(py, "integrate")?;
    let submodule_special = PyModule::new(py, "special")?;
    submodule_integrate.setattr("__doc__", "Integration and ODEs.\n\n")?;
    submodule_special.setattr("__doc__", "Special functions.\n\n")?;
    m.add_submodule(&submodule_integrate)?;
    m.add_submodule(&submodule_special)?;
    integrate::register_module(&submodule_integrate)?;
    special::register_module(&submodule_special)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.math.integrate", submodule_integrate)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.math.special", submodule_special)
}
