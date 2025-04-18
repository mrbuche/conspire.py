super::hyperelastic!(
    NeoHookean,
    "NeoHookean",
    r"The Neo-Hookean hyperelastic constitutive model.[^neohookean]

    [^neohookean]: R.S. Rivlin, [Philos. Trans. R. Soc. London, Ser. A **240**, 459 (1948)](https://doi.org/10.1098/rsta.1948.0002).

    **Parameters**
    - The bulk modulus $\kappa$.
    - The shear modulus $\mu$.

    **External variables**
    - The deformation gradient $\mathbf{F}$.

    **Internal variables**
    - None.",
    r"a(\mathbf{F}) = \frac{\mu}{2}\left[\mathrm{tr}(\mathbf{B}^*) - 3\right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]",
    r"\boldsymbol{\sigma}(\mathbf{F}) = \frac{\mu}{J}\,{\mathbf{B}^*}' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}",
    r"\mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu}{J^{5/3}}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL} - \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}",
    "@private",
    "@private",
    "@private",
    "@private",
    bulk_modulus,
    shear_modulus,
);
