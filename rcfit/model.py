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
from jax.scipy.special import betainc
from numpyro import distributions as dist
from numpyro import deterministic
from numpyro.distributions.transforms import AffineTransform
import numpyro
from quadax import cumulative_trapezoid, fixed_quadgk, quadgk


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
    R200c = M200c2R200c(M200c)
    Rs = R200c / c

    def integrand(x):
        return x**2 / (x**alpha * (1 + x**beta)**gamma)

    # Better understand the range of x
    xmin, xmax = r[0] / Rs, r[-1] / Rs
    x = jnp.logspace(jnp.log10(xmin), jnp.log10(xmax), 2048)

    ymin = fixed_quadgk(integrand, 0., xmin)[0]
    # return ymin
    # ymin = 0.1
    norm = fixed_quadgk(integrand, 0., c)[0]
    # return ymin
    ycum = cumulative_trapezoid(integrand(x), x) + ymin

    ycum = jnp.concatenate([jnp.array([ymin]), ycum])


    Mr = M200c * jnp.interp(r / Rs, x, ycum) / norm

    return GNEWTON * Mr / r

###############################################################################
#                         SPARC model, independent                            #
###############################################################################


def SPARC_independent(galaxy_data, kind="NFW"):
    # Sample halo parameters
    if kind == "NFW":
        logM200c = numpyro.sample(
            "logM200c", dist.TransformedDistribution(dist.Normal(0, 1), AffineTransform(loc=12, scale=2)))  # noqa
        logc = numpyro.sample(
            "logc", dist.TransformedDistribution(dist.Normal(0, 1), AffineTransform(loc=0.905 - 0.101 * (logM200c - 12. + jnp.log10(LITTLE_H)), scale=0.11)))  # noqa
    elif kind in ["gNFW", "alphaNFW", "betaNFW", "gammaNFW"]:
        logM200c = numpyro.sample(
            "logM200c", dist.TransformedDistribution(dist.Normal(0, 1), AffineTransform(loc=12, scale=2)))  # noqa
        logc = numpyro.sample(
            "logc", dist.TransformedDistribution(dist.Normal(0, 1), AffineTransform(loc=0.905 - 0.101 * (logM200c - 12. + jnp.log10(LITTLE_H)), scale=0.11)))  # noqa

        if kind == "gNFW":
            raise NotImplementedError("gNFW not implemented yet.")
        elif kind == "alphaNFW":
            alpha = numpyro.sample("alpha", dist.TruncatedNormal(1.0, 0.5, low=0.))
            beta = deterministic("beta", 2.0)
            gamma = deterministic("gamma", 1.0)
        elif kind == "betaNFW":
            alpha = deterministic("alpha", 1.0)
            beta = numpyro.sample("beta", dist.TruncatedNormal(2.0, 0.5, low=0.))
            gamma = deterministic("gamma", 1.0)
        else:
            alpha = deterministic("alpha", 1.0)
            beta = deterministic("beta", 2.0)
            gamma = deterministic("gamma", 1.0)
            # gamma = numpyro.sample("gamma", dist.TruncatedNormal(1.0, 0.5, low=0.))
    elif kind == "RAR":
        a0 = numpyro.sample("a0", dist.TruncatedNormal(1.19, 0.1, low=0.))
    else:
        raise ValueError(f"Unknown kind: `{kind}`.")

    # Sample mass-to-light ratios
    log_Ups_disk = numpyro.sample(
        "lUps_disk", dist.TransformedDistribution(dist.Normal(0, 1), AffineTransform(loc=jnp.log10(0.5), scale=0.1)))  # noqa
    log_Ups_gas = numpyro.sample(
        "lUps_gas", dist.TransformedDistribution(dist.Normal(0, 1), AffineTransform(loc=jnp.log10(1.), scale=0.04)))  # noqa
    if galaxy_data["has_bulge"]:
        log_Ups_bulge = numpyro.sample(
            "lUps_bulge", dist.TransformedDistribution(dist.Normal(0, 1), AffineTransform(loc=jnp.log10(0.7), scale=0.1)))  # noqa
    else:
        log_Ups_bulge = numpyro.deterministic("lUps_bulge", jnp.array(0.))
    Ups_disk, Ups_bulge, Ups_gas = 10**log_Ups_disk, 10**log_Ups_bulge, 10**log_Ups_gas  # noqa

    # Sample inclination and scale Vobs
    inc = numpyro.sample("inc", dist.TruncatedNormal(galaxy_data["inc"], galaxy_data["e_inc"], low=15 * jnp.pi / 180, high=150 * jnp.pi / 180))  # noqa
    Vobs = numpyro.deterministic("Vobs", galaxy_data["Vobs"] * (jnp.sin(galaxy_data["inc"]) / jnp.sin(inc)))    # noqa
    # e_Vobs = galaxy_data["e_Vobs"] * (jnp.sin(galaxy_data["inc"]) / jnp.sin(inc))  # noqa

    # Sample luminosity
    L36 = numpyro.sample("L36", dist.TruncatedNormal(galaxy_data["L36"], galaxy_data["e_L36"], low=0))  # noqa
    Ups_disk *= L36 / galaxy_data["L36"]
    Ups_bulge *= L36 / galaxy_data["L36"]

    # Sample distance to the galaxy
    galdist = numpyro.sample("dist", dist.TruncatedNormal(galaxy_data["dist"], galaxy_data["e_dist"], low=0))  # noqa
    Vbar_squared = Ups_disk * galaxy_data["Vdisk2"] + Ups_bulge * galaxy_data["Vbul2"] + Ups_gas * galaxy_data["Vgas2"]  # noqa
    Vbar_squared *= galdist / galaxy_data["dist"]

    # Calculate the predicted velocity
    r = numpyro.deterministic("r", galaxy_data["r"] * (galdist / galaxy_data["dist"]))  # noqa
    if kind == "NFW":
        Vnfw_squared = NFW_velocity_squared(r, 10**logM200c, 10**logc)
        Vpred = numpyro.deterministic("Vpred", jnp.sqrt(Vnfw_squared + Vbar_squared))  # noqa
    elif kind in ["gNFW", "alphaNFW", "betaNFW", "gammaNFW"]:
        Vgnfw_squared = gNFW_velocity_squared(r, 10**logM200c, 10**logc, alpha, beta, gamma)
        Vpred = numpyro.deterministic("Vpred", jnp.sqrt(Vgnfw_squared + Vbar_squared))
    elif kind == "RAR":
        # Convert a0 from 1e-10 m/s^2 to km^2 / s^2 / kpc
        gobs = RAR(Vbar_squared / r, 3085.6776 * a0)
        Vpred = numpyro.deterministic("Vpred", jnp.sqrt(gobs * r))
    else:
        raise ValueError(f"Unknown kind: `{kind}`.")

    with numpyro.plate("data", len(galaxy_data["r"]), dim=-1):
        numpyro.sample("obs", dist.Normal(Vpred, galaxy_data["e_Vobs"]), obs=Vobs)
