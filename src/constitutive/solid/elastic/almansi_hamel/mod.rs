use conspire::constitutive::solid::elastic::AlmansiHamel as Base;

super::implement!(
    Base,
    AlmansiHamel,
    "AlmansiHamel",
    r"The Almansi-Hamel elastic constitutive model.

    **Parameters**
    - The bulk modulus $\kappa$.
    - The shear modulus $\mu$.

    **External variables**
    - The deformation gradient $\mathbf{F}$.

    **Internal variables**
    - None.

    **Notes**
    - The Almansi-Hamel strain measure is given by $\mathbf{e}=\tfrac{1}{2}(\mathbf{1}-\mathbf{B}^{-1})$.",
    r"\boldsymbol{\sigma}(\mathbf{F}) = \frac{2\mu}{J}\,\mathbf{e}' + \frac{\kappa}{J}\,\mathrm{tr}(\mathbf{e})\mathbf{1}",
    r"\mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu}{J}\left[B_{jk}^{-1}F_{iL}^{-T} + B_{ik}^{-1}F_{jL}^{-T} - \frac{2}{3}\,\delta_{ij}B_{km}^{-1}F_{mL}^{-T} - 2e_{ij}'F_{kL}^{-T}\right] + \frac{\kappa}{J}\left[\delta_{ij}B_{km}^{-1}F_{mL}^{-T} - \mathrm{tr}(\mathbf{e})\delta_{ij}F_{kL}^{-T}\right]",
    "@private",
    "@private",
    "@private",
    "@private"
);
