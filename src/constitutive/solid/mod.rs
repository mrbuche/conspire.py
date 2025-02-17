mod elastic;
mod hyperelastic;

use pyo3::prelude::*;

pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule_elastic = PyModule::new(py, "elastic")?;
    let submodule_hyperelastic = PyModule::new(py, "hyperelastic")?;
    submodule_elastic.setattr("__doc__", "Elastic constitutive models cannot be defined by a Helmholtz free energy density but still depend on only the deformation gradient. These constitutive models are therefore defined by a relation for some stress measure as a function of the deformation gradient. Consequently, the tangent stiffness associated with the first Piola-Kirchhoff stress is not symmetric for elastic constitutive models.\n$$\\mathcal{C}_{iJkL}\\neq\\mathcal{C}_{kLiJ}$$")?;
    submodule_hyperelastic.setattr("__doc__", "Hyperelastic constitutive models are completely defined by a Helmholtz free energy density function of the deformation gradient.\n$$\\mathbf{P}:\\dot{\\mathbf{F}} - \\dot{a}(\\mathbf{F}) \\geq 0$$\nSatisfying the second law of thermodynamics (here, equivalent to extremized or zero dissipation) yields a relation for the stress.\n$$\\mathbf{P} = \\frac{\\partial a}{\\partial\\mathbf{F}}$$\nConsequently, the tangent stiffness associated with the first Piola-Kirchhoff stress is symmetric for hyperelastic constitutive models.\n$$\\mathcal{C}_{iJkL} = \\mathcal{C}_{kLiJ}$$")?;
    m.add_submodule(&submodule_elastic)?;
    m.add_submodule(&submodule_hyperelastic)?;
    elastic::register_module(&submodule_elastic)?;
    hyperelastic::register_module(&submodule_hyperelastic)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("conspire.constitutive.solid.elastic", submodule_elastic)?;
    py.import("sys")?.getattr("modules")?.set_item(
        "conspire.constitutive.solid.hyperelastic",
        submodule_hyperelastic,
    )
}
