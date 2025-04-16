super::hyperelastic!(
    Fung,
    "Fung",
    r"The Fung hyperelastic constitutive model.[^fung]

    [^fung]: Y.C. Fung, [Am. J. Physiol. **213**, 1532 (1967)](https://doi.org/10.1152/ajplegacy.1967.213.6.1532).

    **Parameters**
    - The bulk modulus $\kappa$.
    - The shear modulus $\mu$.
    - The extra modulus $\mu_m$.
    - The exponent $c$.

    **External variables**
    - The deformation gradient $\mathbf{F}$.

    **Internal variables**
    - None.

    **Notes**
    - The Fung model reduces to the [Neo-Hookean model](#NeoHookean) when $\mu_m\to 0$ or $c\to 0$.",
    r"a(\mathbf{F}) = \frac{\mu - \mu_m}{2}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right] + \frac{\mu_m}{2c}\left(e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)",
    r"\boldsymbol{\sigma}(\mathbf{F}) = \frac{1}{J}\left[\mu + \mu_m\left(e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)\right]{\mathbf{B}^* }' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}",
    r"\mathcal{T}_{ijkL}(\mathbf{F}) = \frac{1}{J^{5/3}}\left[\mu + \mu_m\left(e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)\right]\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL} - \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) + \frac{2c\mu_m}{J^{7/3}}\,e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]}B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}",
    "@private",
    "@private",
    "@private",
    "@private",
    bulk_modulus,
    shear_modulus,
    extra_modulus,
    exponent,
);
