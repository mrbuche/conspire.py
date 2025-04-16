super::hyperelastic!(
    SaintVenantKirchhoff,
    "SaintVenantKirchhoff",
    r"The Saint Venant-Kirchhoff hyperelastic constitutive model.

    **Parameters**
    - The bulk modulus $\kappa$.
    - The shear modulus $\mu$.

    **External variables**
    - The deformation gradient $\mathbf{F}$.

    **Internal variables**
    - None.

    **Notes**
    - The Green-Saint Venant strain measure is given by $\mathbf{E}=\tfrac{1}{2}(\mathbf{C}-\mathbf{1})$.",
    r"a(\mathbf{F}) = \mu\,\mathrm{tr}(\mathbf{E}^2) + \frac{1}{2}\left(\kappa - \frac{2}{3}\,\mu\right)\mathrm{tr}(\mathbf{E})^2",
    "@private",
    "@private",
    "@private",
    "@private",
    r"\mathbf{S}(\mathbf{F}) = 2\mu\mathbf{E}' + \kappa\,\mathrm{tr}(\mathbf{E})\mathbf{1}",
    r"\mathcal{G}_{IJkL}(\mathbf{F}) = \mu\,\delta_{JL}F_{kI} + \mu\,\delta_{IL}F_{kJ} + \left(\kappa - \frac{2}{3}\,\mu\right)\delta_{IJ}F_{kL}",
    bulk_modulus,
    shear_modulus,
);
