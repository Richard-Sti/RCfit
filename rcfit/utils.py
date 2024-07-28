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
"""Various utilities."""
import numpy as np
from h5py import File
from jax import numpy as jnp

from .params import LITTLE_H, RHO200C

###############################################################################
#                               Data IO                                       #
###############################################################################


class SPARCReader:
    """
    SPARC data reader. The SPARC data is assumed to be stored in a HDF5 file.
    """

    def __init__(self, fname):
        self._fname = fname

    def print_nsamples_per_curve(self):
        with File(self._fname) as f:
            gal_names = np.array(list(f.keys()))
            npoints = np.array([f[f"{name}/Vobs"].size for name in gal_names])

            indxs = np.argsort(npoints)[::-1]
            gal_names = gal_names[indxs]
            npoints = npoints[indxs]

            print("Rotation curves are sampled with this many points:")
            for name, npoint in zip(gal_names, npoints):
                print(f"{str(name):<10} {npoint}")

    def __call__(self, name):
        """Load a single galaxy from SPARC."""
        data = {}
        with File(self._fname) as f:
            try:
                f[name]
            except ValueError as e:
                print(f"Galaxy {name} not found in SPARC.")
                raise e

            for key in ["r", "Vobs", "Vdisk", "Vbul", "Vgas",
                        "e_Vobs", "gobs"]:
                data[key] = jnp.array(f[f"{name}/{key}"][:])

            # Convert inclination to radians
            for key in ["inc", "e_inc"]:
                data[key] = np.deg2rad(f[f"{name}/{key}"][0])

            for key in ["dist", "e_dist", "L36", "e_L36"]:
                data[key] = f[f"{name}/{key}"][0]

        # Include squares of velocities too
        for key in ["Vdisk", "Vbul", "Vgas"]:
            data[f"{key}2"] = data[key] * np.abs(data[key])

        data["has_bulge"] = jnp.any(data["Vbul"] > 0)
        return data


###############################################################################
#           Spherical overdensity mass to radius conversion                   #
###############################################################################


def M200c2R200c(M200c):
    """
    Convert M200c [Msun] to R200c [kpc] using the definition of the critical
    density `RHO200C` (which defines the units of `M200c` and `R200c`).
    """
    return (3 * M200c / (4 * jnp.pi * RHO200C * LITTLE_H**2))**(1./3)


###############################################################################
#                           Plotting utitilies                                #
###############################################################################


def name2label(name):
    """Convert parameter names to LaTeX labels."""
    x = {"L36": r"$L_{36}$",
         "dist": r"$d$",
         "inc": r"$i$",
         "logM200c": r"$\log M_{200c}$",
         "Ups_disk": r"$\Upsilon_{\rm disk}$",
         "Ups_gas": r"$\Upsilon_{\rm gas}$",
         "Ups_bulge": r"$\Upsilon_{\rm bulge}$",
         "logc": r"$\log c$",
         "a0": r"$a_0$",
         "alpha": r"$\alpha$",
         "beta": r"$\beta$",
         "gamma": r"$\gamma$",
         }

    try:
        return x[name]
    except KeyError:
        return name
