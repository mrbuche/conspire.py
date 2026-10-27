use conspire::{
    math::{Quantity, Scalar},
    units::{Energy, Force, ForcePerLength, Length, Stress},
};
use ndarray::{ArrayD, IxDyn};
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
            pub fn scalar(&self) -> PyResult<Quantity<$unit>> {
                if self.0.ndim() == 0 {
                    Ok($unit::$base(self.0[IxDyn(&[])]))
                } else {
                    Err(PyValueError::new_err(concat!("expected a single ", $name)))
                }
            }
            pub fn from_scalar(quantity: Quantity<$unit>) -> Self {
                Self(ArrayD::from_elem(IxDyn(&[]), quantity.value()))
            }
            pub fn from_array(array: ArrayD<Scalar>) -> Self {
                Self(array)
            }
            pub fn array(&self) -> &ArrayD<Scalar> {
                &self.0
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

pub trait Parameter: Sized {
    type Inner;
    fn into_inner(self) -> PyResult<Self::Inner>;
    fn from_inner(inner: Self::Inner) -> Self;
}

impl Parameter for Scalar {
    type Inner = Scalar;
    fn into_inner(self) -> PyResult<Scalar> {
        Ok(self)
    }
    fn from_inner(inner: Scalar) -> Scalar {
        inner
    }
}

macro_rules! parameter {
    ($class: ident, $unit: ident) => {
        impl Parameter for $class {
            type Inner = Quantity<$unit>;
            fn into_inner(self) -> PyResult<Quantity<$unit>> {
                self.scalar()
            }
            fn from_inner(inner: Quantity<$unit>) -> Self {
                Self::from_scalar(inner)
            }
        }
    };
}
parameter!(PyEnergy, Energy);
parameter!(PyForce, Force);
parameter!(PyForcePerLength, ForcePerLength);
parameter!(PyLength, Length);
parameter!(PyStress, Stress);
