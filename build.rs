use std::{
    fs::{read_to_string, write},
    io::Error,
    path::Path,
};

const METHODS: [&str; 7] = [
    "cauchy_stress",
    "cauchy_tangent_stiffness",
    "first_piola_kirchhoff_stress",
    "first_piola_kirchhoff_tangent_stiffness",
    "second_piola_kirchhoff_stress",
    "second_piola_kirchhoff_tangent_stiffness",
    "helmholtz_free_energy_density",
];

const MODELS: [&str; 7] = [
    "elastic/almansi_hamel",
    "hyperelastic/arruda_boyce",
    "hyperelastic/fung",
    "hyperelastic/gent",
    "hyperelastic/mooney_rivlin",
    "hyperelastic/neo_hookean",
    "hyperelastic/saint_venant_kirchhoff",
];

fn main() -> Result<(), Error> {
    MODELS.iter().try_for_each(|model| {
        write(
            Path::new(format!("src/constitutive/solid/{model}/model.md").as_str()),
            read_to_string(Path::new(
                format!("conspire.rs/src/constitutive/solid/{model}/model.md").as_str(),
            ))
            .expect("Model description unavailable")
            .replace("$`", "$")
            .replace("`$", "$")
            .replace(
                "[Neo-Hookean model](super::NeoHookean)",
                "[Neo-Hookean model](#NeoHookean)",
            ),
        )?;
        METHODS.iter().try_for_each(|method| {
            write(
                Path::new(format!("src/constitutive/solid/{model}/{method}.md").as_str()),
                read_to_string(Path::new(
                    format!("conspire.rs/src/constitutive/solid/{model}/{method}.md").as_str(),
                ))
                .unwrap_or("@private".to_string())
                .replace("```math", "$$")
                .replace("```", "$$")
                .replace("\\begin{aligned}", "")
                .replace("\\end{aligned}", "")
                .replace("&", "")
                .replace("\\\\", "")
                .replace("\n", ""),
            )
        })
    })
}
