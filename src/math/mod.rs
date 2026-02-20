mod integrate;
mod special;

use crate::PyErrGlue;
use conspire::math::{Scalar, Tensor};
use ndarray::Array;
use numpy::{PyArray2, PyArray4};
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
    pub fn into_pyarray<'py>(
        self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyArray2<Scalar>>, PyErrGlue> {
        Ok(PyArray2::from_vec2(py, &Vec::<Vec<Scalar>>::from(self.0))?)
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
    pub fn into_pyarray<'py>(
        self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyArray4<Scalar>>, PyErrGlue> {
        Ok(PyArray4::from_owned_array(
            py,
            Array::from_shape_vec((3, 3, 3, 3), Vec::<Scalar>::from(self.0))?,
        ))
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
