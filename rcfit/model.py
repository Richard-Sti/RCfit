# Copyright (C) 2024 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""RC fitting models in NumPyro."""
from jax import numpy as jnp
import numpy as np
from numpyro import distributions as dist
from numpyro import deterministic, sample, factor
from numpyro.distributions.transforms import AffineTransform
from quadax import cumulative_trapezoid, fixed_quadgk


from .params import GNEWTON, LITTLE_H

from .utils import M200c2R200c

###############################################################################
#                             RAR (rocks)                                     #
###############################################################################


def RAR(gbar, a0):
    """
    RAR IF relating baryonic acceleration to observed acceleration. assuming
    some value of a0.
    """
    return gbar / (1 - jnp.exp(-jnp.sqrt(gbar / a0)))


###############################################################################
#                           NFW rotation velocity                             #
###############################################################################


def NFW_velocity_squared(r, M200c, c):
    """
    Squared circular velocity of an NFW profile at radius `r` for a halo with
    mass `M200c` and concentration `c`.
    """
    R200c = M200c2R200c(M200c)
    Rs = R200c / c
    return GNEWTON * M200c / r * (jnp.log(1 + r / Rs) - r / (r + Rs)) / (jnp.log(1 + c) - c / (1 + c))  # noqa


def gNFW_velocity_squared(r, M200c, c, alpha, beta, gamma):
    """
    Generalized NFW rotation velocity squared, calculated by numerically
    integrating the density profile.
    """
    R200c = M200c2R200c(M200c)
    Rs = R200c / c

    def integrand(x):
        return x**2 / (x**alpha * (1 + x**beta)**gamma)

    xmin, xmax = r[0] / Rs, r[-1] / Rs
    x = jnp.logspace(jnp.log10(xmin), jnp.log10(xmax), 2048)

    # Integrand from 0 to minial range.
    ymin = fixed_quadgk(integrand, 0., xmin)[0]
    # Normalizattoin constant
    norm = fixed_quadgk(integrand, 0., c)[0]

    # Integrate from xmin to xmax.
    ycum = cumulative_trapezoid(integrand(x), x) + ymin
    ycum = jnp.concatenate([jnp.array([ymin]), ycum])

    # Calculate the mass enclosed within r.
    Mr = M200c * jnp.interp(r / Rs, x, ycum) / norm
    return GNEWTON * Mr / r

###############################################################################
#                         SPARC model, independent                            #
###############################################################################


def transformed_normal(loc, scale):
    return dist.TransformedDistribution(
        dist.Normal(0, 1), AffineTransform(loc=loc, scale=scale))


def SPARC_independent(galaxy_data, kind="NFW"):
    # Sample halo/MOND parameters
    if kind in ["NFW", "gNFW", "alphaNFW", "betaNFW", "gammaNFW"]:
        logM200c = sample("logM200c", transformed_normal(12.0, 2.0))
        logc_mean = 0.905 - 0.101 * (logM200c - 12. + jnp.log10(LITTLE_H))
        logc = sample("logc", transformed_normal(logc_mean, 0.11))

        if kind == "NFW":
            pass
        elif kind == "gNFW":
            raise NotImplementedError("gNFW not implemented yet.")
        elif kind == "alphaNFW":
            alpha = sample("alpha", dist.TruncatedNormal(1.0, 0.5, low=0.))  # noqa
            beta = deterministic("beta", 2.0)
            gamma = deterministic("gamma", 1.0)
        elif kind == "betaNFW":
            alpha = deterministic("alpha", 1.0)
            beta = sample("beta", dist.TruncatedNormal(2.0, 0.5, low=0.))    # noqa
            gamma = deterministic("gamma", 1.0)
        elif kind == "gammaNFW":
            alpha = deterministic("alpha", 1.0)
            beta = deterministic("beta", 2.0)
            gamma = sample("gamma", dist.TruncatedNormal(1.0, 1.0, low=0.))  # noqa
        else:
            raise ValueError(f"Unknown kind: `{kind}`.")
    elif kind == "RAR":
        a0 = sample("a0", dist.TruncatedNormal(1.19, 0.1, low=0.))
    else:
        raise ValueError(f"Unknown kind: `{kind}`.")

    # Sample mass-to-light ratios
    log_Ups_disk = sample("log_Ups_disk", transformed_normal(jnp.log10(0.5), 0.1))        # noqa
    log_Ups_gas = sample("log_Ups_gas", transformed_normal(jnp.log10(1.), 0.04))          # noqa
    if galaxy_data["has_bulge"]:
        log_Ups_bulge = sample("log_Ups_bulge", transformed_normal(jnp.log10(0.7), 0.1))  # noqa
    else:
        log_Ups_bulge = deterministic("log_Ups_bulge", jnp.array(0.))
    Ups_disk, Ups_bulge, Ups_gas = 10**log_Ups_disk, 10**log_Ups_bulge, 10**log_Ups_gas   # noqa

    # Sample inclination and scale Vobs
    inc_min, inc_max = 15 * jnp.pi / 180, 150 * jnp.pi / 180
    inc = sample("inc",dist.TruncatedNormal(galaxy_data["inc"], galaxy_data["e_inc"], low=inc_min, high=inc_max))  # noqa
    inc_scaling = jnp.sin(galaxy_data["inc"]) / jnp.sin(inc)
    Vobs = deterministic("Vobs", galaxy_data["Vobs"] * inc_scaling)
    e_Vobs = deterministic("e_Vobs", galaxy_data["e_Vobs"] * inc_scaling)

    # Sample luminosity
    L36 = sample("L36", dist.TruncatedNormal(galaxy_data["L36"], galaxy_data["e_L36"], low=0))  # noqa
    Ups_disk *= L36 / galaxy_data["L36"]
    Ups_bulge *= L36 / galaxy_data["L36"]

    # Sample distance to the galaxy
    galdist = sample("dist", dist.TruncatedNormal(galaxy_data["dist"], galaxy_data["e_dist"], low=0))  # noqa
    dist_scaling = galdist / galaxy_data["dist"]

    Vbar_squared = (+ Ups_disk * galaxy_data["Vdisk2"]
                    + Ups_bulge * galaxy_data["Vbul2"]
                    + Ups_gas * galaxy_data["Vgas2"])
    Vbar_squared *= dist_scaling

    # Calculate the predicted velocity
    r = deterministic("r", galaxy_data["r"] * dist_scaling)
    if kind == "NFW":
        Vnfw_squared = NFW_velocity_squared(r, 10**logM200c, 10**logc)
        Vpred = deterministic("Vpred", jnp.sqrt(Vnfw_squared + Vbar_squared))  # noqa
    elif kind in ["gNFW", "alphaNFW", "betaNFW", "gammaNFW"]:
        Vgnfw_squared = gNFW_velocity_squared(
            r, 10**logM200c, 10**logc, alpha, beta, gamma)
        Vpred = deterministic("Vpred", jnp.sqrt(Vgnfw_squared + Vbar_squared))
    elif kind == "RAR":
        # Convert a0 from 1e-10 m/s^2 to km^2 / s^2 / kpc
        gobs = RAR(Vbar_squared / r, 3085.6776 * a0)
        Vpred = deterministic("Vpred", jnp.sqrt(gobs * r))
    else:
        raise ValueError(f"Unknown kind: `{kind}`.")

    ll = jnp.sum(dist.Normal(Vpred, e_Vobs).log_prob(Vobs))
    # We want to keep track of the log likelihood for BIC/AIC calculations.
    deterministic("log_likelihood", ll)
    factor("ll", ll)


###############################################################################
#                            BIC calculations                                 #
###############################################################################


def BIC_from_samples(samples, log_likelihood,
                     skip_keys=["Vobs", "Vpred", "e_Vobs", "log_likelihood", "r"]):  # noqa
    """
    Get the BIC from HMC samples of a Numpyro model.

    Parameters
    ----------
    samples: dict
        Dictionary of samples from the Numpyro MCMC object.
    log_likelihood: numpy array
        Log likelihood values of the samples.
    skip_keys: list
        List of keys to skip when counting the number of parameters

    Returns
    -------
    BIC, AIC: floats
    """
    ndata = samples["Vobs"].shape[1]
    kmax = np.argmax(log_likelihood)

    # How many parameters?
    nparam = 0
    for key, val in samples.items():
        if key in skip_keys or val.std() == 0:
            continue

        if val.ndim == 1:
            nparam += 1
        elif val.ndim == 2:
            nparam += val.shape[-1]
        else:
            raise ValueError("Invalid dimensionality of samples to count the number of parameters.")  # noqa

    BIC = nparam * np.log(ndata) - 2 * log_likelihood[kmax]

    return float(BIC)
