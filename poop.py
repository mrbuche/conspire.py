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


kappa = 0.1
eta = np.linspace(1e-6, kappa, 5)
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
    sqrt_term = np.pi * np.sqrt(2 * np.pi / kappa) * np.exp( eta**2 / (2 * k) ) / eta
    term2 = np.exp(eta) * (eta / kappa + 1) * (1 + erf((eta + k) / (np.sqrt(2 * k))))
    term3 = np.exp(-eta) * (eta / kappa - 1) * (1 - erf((eta - k) / (np.sqrt(2 * k))))
    return sqrt_term * (term2 + term3)


def baz_3(k, eta):
    term2 = 4 / np.sqrt( 2 * np.pi * k ) * eta / kappa
    term3 =  np.exp( eta**2/2/k + eta + k/2 ) * ( 1 / k + ( eta / k + 1 )** 2 ) * ( 1 + erf (( eta + k ) / np.sqrt( 2 * k )))
    term4 = -np.exp( eta**2/2/k - eta + k/2 ) * ( 1 / k + ( eta / k - 1 )** 2 ) * ( 1 - erf (( eta - k ) / np.sqrt( 2 * k )))
    return ( term2 + term3 + term4) * np.pi / ( np.exp( k / 2) * ( 2 * k ) ) / eta * np.sqrt( 2 * np.pi / kappa) * 2 * k


for i, eta_i in enumerate(eta):
    num = quad(lambda lam: bar_3(kappa, lam, eta_i), 0, np.inf)[0]
    den = quad(lambda lam: foo(kappa, lam, eta_i), 0, np.inf)[0]
    print(eta_i, num / den, baz_3(kappa, eta_i) / foo_3(kappa, eta_i))


# def p(kappa, lam, eta):
#     return (
#         1
#         / np.sqrt(2 * np.pi / kappa)
#         * lam
#         # * np.sinh(lam * eta)
#         # / np.sinh(eta)
#         * (
#             np.exp(eta * (lam - 1) - kappa * (lam - 1) ** 2 / 2 - eta**2 / 2 / kappa)
#             - np.exp(-eta * (lam + 1) - kappa * (lam - 1) ** 2 / 2 - eta**2 / 2 / kappa)
#         )
#         / (1.0 - np.exp(-2.0 * eta))
#         # * np.exp(-kappa * (lam - 1) ** 2 / 2 - eta**2 / 2 / kappa)
#         / (1 + eta / kappa / np.tanh(eta))
#     )


# import matplotlib.pyplot as plt

# lamb = np.linspace(1e-6, 4, 100)

# for i, eta_i in enumerate(eta):
#     print(quad(lambda lam: p(kappa, lam, eta_i), 1e-6, np.inf)[0])
#     plt.plot(lamb, p(kappa, lamb, eta_i))

# plt.show()
