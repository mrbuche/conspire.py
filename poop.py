from conspire.math.special import sinhc
from scipy.integrate import quad
import numpy as np


def foo(kappa, lam, eta):
    return (
        np.exp(lam * eta - kappa * (lam - 1) ** 2 / 2)
        * (1 - np.exp(-2 * lam * eta))
        / 2
        * lam
        / eta
    )


def bar(kappa, lam, eta):
    return foo(kappa, lam, eta) * (lam - 1) ** 2 / 2


def helper(kappa, eta):
    return eta / (kappa * np.tanh(eta) + eta)


def baz(kappa, eta):
    return (1 / 2 + helper(kappa, eta) + eta**2 / kappa / 2) / kappa


kappa = 1e-2  # 10
eta = np.linspace(1e-6, 10 * kappa, 5)
for i, eta_i in enumerate(eta):
    num = quad(lambda lam: bar(kappa, lam, eta_i), 0, np.inf)[0]
    den = quad(lambda lam: foo(kappa, lam, eta_i), 0, np.inf)[0]
    print(eta_i, num / den, (eta_i / kappa) ** 2 / 2, baz(kappa, eta_i))

print()


def bar_2(kappa, lam, eta):
    return foo(kappa, lam, eta) * ((lam - 1) ** 2 / 2) ** 2


def baz_2(kappa, eta):
    return (
        1 / 2 + helper(kappa, eta) * (2 - helper(kappa, eta)) + eta**2 / kappa
    ) / kappa**2


for i, eta_i in enumerate(eta):
    num = quad(lambda lam: bar(kappa, lam, eta_i), 0, np.inf)[0]
    num_2 = quad(lambda lam: bar_2(kappa, lam, eta_i), 0, np.inf)[0]
    den = quad(lambda lam: foo(kappa, lam, eta_i), 0, np.inf)[0]
    print(
        eta_i,
        np.sqrt(num_2 / den - (num / den) ** 2),
        np.sqrt(eta_i**2 / kappa**3),
        np.sqrt(baz_2(kappa, eta_i)),
    )

print()


def bar_3(kappa, lam, eta):
    return foo(kappa, lam, eta) * lam


# def baz_3(kappa, eta):
#     # return 1 + (1 + eta / np.tanh(eta)) / kappa  # feel like should be 1/2K, matches better too, but still not close enough
#     # return 1 + (0.5 + eta / np.tanh(eta)) / kappa
#     # return (1 + (eta**2 / 2 + 2 * eta / np.tanh(eta) + 1) / kappa) / (
#     #     1 + (eta**2 / 2 + eta / np.tanh(eta)) / kappa
#     # )
#     return (1 + (1 + 2 * eta / np.tanh(eta)) / kappa) / (
#         1 + eta / np.tanh(eta) / kappa
#     )

from scipy.special import erf, erfc


def foo_3(k, eta):
    prefactor = np.pi * np.sqrt(2 * np.pi / kappa) * np.exp(eta**2 / (2 * k)) / eta
    term2 = np.exp(+eta) * (eta / kappa + 1) * (1 + erf((eta + k) / np.sqrt(2 * k)))
    term3 = np.exp(-eta) * (eta / kappa - 1) * (1 - erf((eta - k) / np.sqrt(2 * k)))
    return prefactor * (term2 + term3)


def baz_3(k, eta):
    prefactor = np.pi * np.sqrt(2 * np.pi / kappa) * np.exp(eta**2 / (2 * k)) / eta
    term2 = (
        4
        * np.exp(-(eta**2) / 2 / k)
        * np.exp(-k / 2)
        / np.sqrt(2 * np.pi * k)
        * eta
        / kappa
    )
    term3 = (
        +np.exp(+eta)
        * (1 / k + (eta / k + 1) ** 2)
        * (1 + erf((eta + k) / np.sqrt(2 * k)))
    )
    term4 = (
        -np.exp(-eta)
        * (1 / k + (eta / k - 1) ** 2)
        * (1 - erf((eta - k) / np.sqrt(2 * k)))
    )
    return prefactor * (term2 + term3 + term4)


# def lambda_asmptotic(kappa, eta):
#     foo = eta / kappa
#     return (1 + 1/kappa + 2*foo / np.tanh(eta)) / (1 + foo / np.tanh(eta))


def lambda_asmptotic(kappa, eta):
    eta_over_kappa = eta / kappa
    return (
        1
        + eta_over_kappa
        + (1 / kappa + eta_over_kappa * (1 - eta_over_kappa) * (1 / np.tanh(eta) - 1))
        / (1 + eta_over_kappa / np.tanh(eta))
    )


for i, eta_i in enumerate(eta):
    num = quad(lambda lam: bar_3(kappa, lam, eta_i), 0, np.inf)[0]
    den = quad(lambda lam: foo(kappa, lam, eta_i), 0, np.inf)[0]
    print(
        eta_i,
        num / den,
        baz_3(kappa, eta_i) / foo_3(kappa, eta_i),
        lambda_asmptotic(kappa, eta_i),
    )

print()


def bar_4(kappa, lam, eta):
    return foo(kappa, lam, eta) * lam**2


def baz_4(k, a):
    term1 = 2 * np.sqrt(k) * (2 * k + (a + k) ** 2) + np.exp((a + k) ** 2 / (2 * k)) * (
        a + k
    ) * (3 * k + (a + k) ** 2) * np.sqrt(2 * np.pi) * (
        1 + erf((a + k) / (np.sqrt(2) * np.sqrt(k)))
    )
    term1 /= np.exp(k / 2) * (2 * k ** (7 / 2))
    term2 = 2 * np.sqrt(k) * ((a - k) ** 2 + 2 * k) - np.exp((a - k) ** 2 / (2 * k)) * (
        a - k
    ) * ((a - k) ** 2 + 3 * k) * np.sqrt(2 * np.pi) * erfc(
        (a - k) / (np.sqrt(2) * np.sqrt(k))
    )
    term2 /= np.exp(k / 2) * (2 * k ** (7 / 2))
    return (term1 - term2) / a * np.pi * 2


def lambda_asmptotic_2(kappa, eta):
    eta_over_kappa = eta / kappa
    return (
        1
        + (
            2 * eta_over_kappa**2
            + 3 / kappa
            + (3 / kappa + 2) * eta_over_kappa / np.tanh(eta)
        )
        / (1 + eta_over_kappa / np.tanh(eta))
        + eta_over_kappa**2
    )


for i, eta_i in enumerate(eta):
    num = quad(lambda lam: bar_4(kappa, lam, eta_i), 0, np.inf)[0]
    den = quad(lambda lam: foo(kappa, lam, eta_i), 0, np.inf)[0]
    print(
        eta_i,
        num / den,
        baz_4(kappa, eta_i) / foo_3(kappa, eta_i),
        lambda_asmptotic_2(kappa, eta_i),
    )

print()


def bar_5(kappa, lam, eta):
    return foo(kappa, lam, eta) * lam**3


def baz_5(k, a):
    return (
        np.exp(-k / 2)
        / (np.sqrt(2) * k ** (9 / 2))
        * (
            2 * np.sqrt(2) * a**3 * np.sqrt(k)
            + 10 * np.sqrt(2) * a * k ** (3 / 2)
            + 6 * np.sqrt(2) * a * k ** (5 / 2)
            - a**4 * np.exp((a - k) ** 2 / (2 * k)) * np.sqrt(np.pi)
            + a**4 * np.exp((a + k) ** 2 / (2 * k)) * np.sqrt(np.pi)
            - 6 * a**2 * np.exp((a - k) ** 2 / (2 * k)) * k * np.sqrt(np.pi)
            + 4 * a**3 * np.exp((a - k) ** 2 / (2 * k)) * k * np.sqrt(np.pi)
            + 6 * a**2 * np.exp((a + k) ** 2 / (2 * k)) * k * np.sqrt(np.pi)
            + 4 * a**3 * np.exp((a + k) ** 2 / (2 * k)) * k * np.sqrt(np.pi)
            - 3 * np.exp((a - k) ** 2 / (2 * k)) * k**2 * np.sqrt(np.pi)
            + 12 * a * np.exp((a - k) ** 2 / (2 * k)) * k**2 * np.sqrt(np.pi)
            - 6 * a**2 * np.exp((a - k) ** 2 / (2 * k)) * k**2 * np.sqrt(np.pi)
            + 3 * np.exp((a + k) ** 2 / (2 * k)) * k**2 * np.sqrt(np.pi)
            + 12 * a * np.exp((a + k) ** 2 / (2 * k)) * k**2 * np.sqrt(np.pi)
            + 6 * a**2 * np.exp((a + k) ** 2 / (2 * k)) * k**2 * np.sqrt(np.pi)
            - 6 * np.exp((a - k) ** 2 / (2 * k)) * k**3 * np.sqrt(np.pi)
            + 4 * a * np.exp((a - k) ** 2 / (2 * k)) * k**3 * np.sqrt(np.pi)
            + 6 * np.exp((a + k) ** 2 / (2 * k)) * k**3 * np.sqrt(np.pi)
            + 4 * a * np.exp((a + k) ** 2 / (2 * k)) * k**3 * np.sqrt(np.pi)
            - np.exp((a - k) ** 2 / (2 * k)) * k**4 * np.sqrt(np.pi)
            + np.exp((a + k) ** 2 / (2 * k)) * k**4 * np.sqrt(np.pi)
            + np.exp((a - k) ** 2 / (2 * k))
            * (
                a**4
                - 4 * a**3 * k
                + 6 * a**2 * k * (1 + k)
                - 4 * a * k**2 * (3 + k)
                + k**2 * (3 + 6 * k + k**2)
            )
            * np.sqrt(np.pi)
            * erf((a - k) / (np.sqrt(2) * np.sqrt(k)))
            + np.exp((a + k) ** 2 / (2 * k))
            * (
                a**4
                + 4 * a**3 * k
                + 6 * a**2 * k * (1 + k)
                + 4 * a * k**2 * (3 + k)
                + k**2 * (3 + 6 * k + k**2)
            )
            * np.sqrt(np.pi)
            * erf((a + k) / (np.sqrt(2) * np.sqrt(k)))
        ) / a * np.pi * 2
    )


for i, eta_i in enumerate(eta):
    num = quad(lambda lam: bar_5(kappa, lam, eta_i), 0, np.inf)[0]
    den = quad(lambda lam: foo(kappa, lam, eta_i), 0, np.inf)[0]
    print(
        eta_i,
        num / den,
        baz_5(kappa, eta_i) / foo_3(kappa, eta_i),
    )

print()


def bar_6(kappa, lam, eta):
    return foo(kappa, lam, eta) * lam**4


def baz_6(k, a):
    return (
        np.exp(-k / 2)
        / (2 * k ** (11 / 2))
        * (
            -2 * (a - k) ** 4 * np.sqrt(k)
            - 18 * (a - k) ** 2 * k ** (3 / 2)
            + 18 * k ** (7 / 2)
            + 2 * k ** (9 / 2)
            + 2 * np.sqrt(2) * a**3 * k * (
                2 * np.sqrt(2) * np.sqrt(k)
                + 5 * np.exp((a + k) ** 2 / (2 * k)) * np.sqrt(np.pi)
                + 5 * np.exp((a + k) ** 2 / (2 * k)) * k * np.sqrt(np.pi)
            )
            + a**5 * np.exp((a + k) ** 2 / (2 * k)) * np.sqrt(2 * np.pi)
            + 15 * np.exp((a + k) ** 2 / (2 * k)) * k**3 * np.sqrt(2 * np.pi)
            + 10 * np.exp((a + k) ** 2 / (2 * k)) * k**4 * np.sqrt(2 * np.pi)
            + np.exp((a + k) ** 2 / (2 * k)) * k**5 * np.sqrt(2 * np.pi)
            + a**4 * (
                2 * np.sqrt(k)
                + 5 * np.exp((a + k) ** 2 / (2 * k)) * k * np.sqrt(2 * np.pi)
            )
            + 2 * a**2 * k ** (3 / 2) * (
                9
                + 6 * k
                + 15 * np.exp((a + k) ** 2 / (2 * k)) * np.sqrt(k) * np.sqrt(2 * np.pi)
                + 5 * np.exp((a + k) ** 2 / (2 * k)) * k ** (3 / 2) * np.sqrt(2 * np.pi)
            )
            + a * k**2 * (
                36 * np.sqrt(k)
                + 8 * k ** (3 / 2)
                + 15 * np.exp((a + k) ** 2 / (2 * k)) * np.sqrt(2 * np.pi)
                + 30 * np.exp((a + k) ** 2 / (2 * k)) * k * np.sqrt(2 * np.pi)
                + 5 * np.exp((a + k) ** 2 / (2 * k)) * k**2 * np.sqrt(2 * np.pi)
            )
            - np.exp((a - k) ** 2 / (2 * k))
            * (a - k) ** 5
            * np.sqrt(2 * np.pi)
            * (-1 + erf((a - k) / (np.sqrt(2) * np.sqrt(k))))
            - 10
            * np.exp((a - k) ** 2 / (2 * k))
            * (a - k) ** 3
            * k
            * np.sqrt(2 * np.pi)
            * (-1 + erf((a - k) / (np.sqrt(2) * np.sqrt(k))))
            - 15
            * np.exp((a - k) ** 2 / (2 * k))
            * (a - k)
            * k**2
            * np.sqrt(2 * np.pi)
            * (-1 + erf((a - k) / (np.sqrt(2) * np.sqrt(k))))
            + np.exp((a + k) ** 2 / (2 * k))
            * (
                a**5
                + 5 * a**4 * k
                + 10 * a**3 * k * (1 + k)
                + 10 * a**2 * k**2 * (3 + k)
                + 5 * a * k**2 * (3 + 6 * k + k**2)
                + k**3 * (15 + 10 * k + k**2)
            )
            * np.sqrt(2 * np.pi)
            * erf((a + k) / (np.sqrt(2) * np.sqrt(k)))
        )
        / a
        * np.pi
        * 2
    )


for i, eta_i in enumerate(eta):
    num = quad(lambda lam: bar_6(kappa, lam, eta_i), 0, np.inf)[0]
    den = quad(lambda lam: foo(kappa, lam, eta_i), 0, np.inf)[0]
    print(
        eta_i,
        num / den,
        baz_6(kappa, eta_i) / foo_3(kappa, eta_i),
    )
