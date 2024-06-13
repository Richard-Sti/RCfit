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
"""


"""
import numpy as np
from jax import numpy as jnp
from h5py import File


from .params import RHO200C, LITTLE_H


###############################################################################
#                               Data IO                                       #
###############################################################################


class SPARCReader:

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

            # TODO: order data

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


def M200c2R200c(M200c):
    """
    Convert M200c [Msun] to R200c [kpc] using the definition of the critical
    density `RHO200C` (which defines the units of `M200c` and `R200c`).
    """
    return (3 * M200c / (4 * jnp.pi * RHO200C * LITTLE_H**2))**(1./3)



def cumtrapezoid(y, x=None, dx=1.0, axis=-1):
    """
    Compute the cumulative trapezoidal integration of y with respect to x.

    Parameters:
    y : array_like
        Values to integrate.
    x : array_like, optional
        The sample points corresponding to the y values. If x is None, the sample
        points are assumed to be evenly spaced with spacing `dx`.
    dx : scalar, optional
        The spacing between sample points when `x` is None. Default is 1.0.
    axis : int, optional
        The axis along which to integrate. Default is the last axis.

    Returns:
    y_int : ndarray
        Cumulative trapezoidal integral of `y` along `axis`.
    """
    y = jnp.asarray(y)

    if x is None:
        d = dx
    else:
        x = jnp.asarray(x)
        d = jnp.diff(x, axis=axis)

    shape = list(y.shape)
    shape[axis] = 1

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim

    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    if x is None:
        trapezoids = 0.5 * (y[tuple(slice1)] + y[tuple(slice2)]) * d
    else:
        trapezoids = 0.5 * (y[tuple(slice1)] + y[tuple(slice2)]) * d

    # Prepend a zero to the cumulative sum to match the shape
    initial = jnp.zeros(shape, dtype=trapezoids.dtype)
    cumulative_integral = jnp.concatenate([initial, jnp.cumsum(trapezoids, axis=axis)], axis=axis)

    return cumulative_integral

# # Example usage:
# y = jnp.array([1, 2, 3, 4, 5])
# x = jnp.array([0, 1, 2, 3, 4])
#
# result = cumtrapezoid(y, x)
# print(result)