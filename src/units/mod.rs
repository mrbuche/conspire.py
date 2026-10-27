use conspire::{
    math::{
        Current, Quantity, Reference, Scalar, Tensor, TensorRank1Vec, TensorRank2,
        TensorRank2SparseVec2D, TensorRank4,
    },
    units::{Energy, Force, ForcePerLength, Length, Stress},
};
use ndarray::{ArrayD, Ix2, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::{IntoPyObjectExt, exceptions::PyValueError, prelude::*};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnergy>()?;
    m.add_class::<PyForce>()?;
    m.add_class::<PyForcePerLength>()?;
    m.add_class::<PyLength>()?;
    m.add_class::<PyStress>()
}

fn array(value: &Bound<'_, PyAny>) -> PyResult<ArrayD<Scalar>> {
    match value.extract::<Scalar>() {
        Ok(scalar) => Ok(ArrayD::from_elem(IxDyn(&[]), scalar)),
        Err(_) => Ok(value
            .extract::<PyReadonlyArrayDyn<Scalar>>()?
            .as_array()
            .to_owned()),
    }
}

fn read(py: Python<'_>, array: &ArrayD<Scalar>, divisor: Scalar) -> PyResult<Py<PyAny>> {
    let scaled = array.mapv(|value| value / divisor);
    if scaled.ndim() == 0 {
        scaled[IxDyn(&[])].into_py_any(py)
    } else {
        Ok(scaled.into_pyarray(py).into_any().unbind())
    }
}

macro_rules! quantity {
    ($class: ident, $name: literal, $unit: ident, base $base: ident, $($scale: ident / $reader: ident),+ $(,)?) => {
        #[doc = concat!("A quantity of ", $name, ", held in the base scale.")]
        #[pyclass(name = $name, from_py_object)]
        #[derive(Clone, Debug)]
        pub struct $class(ArrayD<Scalar>);
        #[pymethods]
        impl $class {
            $(
                #[staticmethod]
                fn $scale(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                    Ok(Self(array(value)? * $unit::$scale(1.0).value()))
                }
            )+
            $(
                fn $reader(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                    read(py, &self.0, $unit::$scale(1.0).value())
                }
            )+
            fn __repr__(&self) -> String {
                format!(concat!($name, ".", stringify!($base), "({})"), self.0)
            }
        }
        impl $class {
            pub fn array(&self) -> &ArrayD<Scalar> {
                &self.0
            }
        }
        impl From<Quantity<$unit>> for $class {
            fn from(quantity: Quantity<$unit>) -> Self {
                Self(ArrayD::from_elem(IxDyn(&[]), quantity.value()))
            }
        }
        impl TryFrom<$class> for Quantity<$unit> {
            type Error = PyErr;
            fn try_from(quantity: $class) -> PyResult<Self> {
                if quantity.0.ndim() == 0 {
                    Ok($unit::$base(quantity.0[IxDyn(&[])]))
                } else {
                    Err(PyValueError::new_err(concat!("expected a single ", $name)))
                }
            }
        }
        impl<I, J> From<TensorRank2<3, I, J, $unit>> for $class {
            fn from(tensor: TensorRank2<3, I, J, $unit>) -> Self {
                Self(ArrayD::from_shape_fn(IxDyn(&[3, 3]), |ix| {
                    tensor[ix[0]][ix[1]].value()
                }))
            }
        }
        impl<I, J, K, L> From<TensorRank4<3, I, J, K, L, $unit>> for $class {
            fn from(tensor: TensorRank4<3, I, J, K, L, $unit>) -> Self {
                Self(ArrayD::from_shape_fn(IxDyn(&[3, 3, 3, 3]), |ix| {
                    tensor[ix[0]][ix[1]][ix[2]][ix[3]].value()
                }))
            }
        }
        impl<I> From<TensorRank1Vec<3, I, $unit>> for $class {
            fn from(vector: TensorRank1Vec<3, I, $unit>) -> Self {
                Self(ArrayD::from_shape_fn(IxDyn(&[vector.len(), 3]), |ix| {
                    vector[ix[0]][ix[1]].value()
                }))
            }
        }
    };
}

quantity!(
    PyEnergy, "Energy", Energy, base joules,
    joules / in_joules,
    millijoules / in_millijoules,
    kilojoules / in_kilojoules,
    electronvolts / in_electronvolts,
);
quantity!(
    PyForce, "Force", Force, base newtons,
    newtons / in_newtons,
    millinewtons / in_millinewtons,
    kilonewtons / in_kilonewtons,
    meganewtons / in_meganewtons,
    pounds_force / in_pounds_force,
);
quantity!(
    PyForcePerLength, "ForcePerLength", ForcePerLength, base newtons_per_meter,
    newtons_per_meter / in_newtons_per_meter,
    piconewtons_per_nanometer / in_piconewtons_per_nanometer,
);
quantity!(
    PyLength, "Length", Length, base meters,
    meters / in_meters,
    millimeters / in_millimeters,
    micrometers / in_micrometers,
    nanometers / in_nanometers,
    centimeters / in_centimeters,
    kilometers / in_kilometers,
    inches / in_inches,
    feet / in_feet,
);
quantity!(
    PyStress, "Stress", Stress, base pascals,
    pascals / in_pascals,
    kilopascals / in_kilopascals,
    megapascals / in_megapascals,
    gigapascals / in_gigapascals,
    bars / in_bars,
    psi / in_psi,
    ksi / in_ksi,
);

impl From<&TensorRank2SparseVec2D<3, Current, Current, ForcePerLength>> for PyForcePerLength {
    fn from(stiffnesses: &TensorRank2SparseVec2D<3, Current, Current, ForcePerLength>) -> Self {
        let nodes = stiffnesses.len();
        let mut array = ArrayD::<Scalar>::zeros(IxDyn(&[nodes, nodes, 3, 3]));
        stiffnesses.iter().enumerate().for_each(|(a, row)| {
            row.entries().for_each(|(b, entry)| {
                (0..3).for_each(|i| {
                    (0..3).for_each(|j| array[IxDyn(&[a, b, i, j])] = entry[i][j].value())
                })
            })
        });
        Self(array)
    }
}

fn nodes(length: &PyLength) -> PyResult<Vec<[Scalar; 3]>> {
    match length.0.view().into_dimensionality::<Ix2>() {
        Ok(array) if array.ncols() == 3 => Ok(array
            .rows()
            .into_iter()
            .map(|node| [node[0], node[1], node[2]])
            .collect()),
        _ => Err(PyValueError::new_err(
            "expected nodal coordinates with shape (nodes, 3)",
        )),
    }
}

impl TryFrom<&PyLength> for TensorRank1Vec<3, Current, Length> {
    type Error = PyErr;
    fn try_from(length: &PyLength) -> PyResult<Self> {
        Ok(nodes(length)?.into())
    }
}

impl TryFrom<&PyLength> for TensorRank1Vec<3, Reference, Length> {
    type Error = PyErr;
    fn try_from(length: &PyLength) -> PyResult<Self> {
        Ok(nodes(length)?.into())
    }
}
