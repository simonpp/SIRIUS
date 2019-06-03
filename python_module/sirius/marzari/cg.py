from copy import deepcopy

import numpy as np
from mpi4py import MPI
from scipy.constants import physical_constants

from ..baarman.direct_minimization import df_fermi_entropy, fermi_entropy
from ..coefficient_array import diag, inner, spdiag
from ..edft.neugebaur import fermi_function, find_chemical_potential
from ..logger import Logger

from ..edft.neugebaur import kb

logger = Logger()


def _solve(A, X):
    """
    returns A⁻¹ X
    """
    out = type(X)(dtype=X.dtype, ctype=X.ctype)
    for k in X.keys():
        out[k] = np.linalg.solve(A[k], X[k])
    return out


class FreeEnergy:
    """
    copied from Baarman implementation
    """

    def __init__(self, E, T, H, delta=1e-4):
        """
        Keyword Arguments:
        energy      -- total energy object
        temperature -- temperature in Kelvin
        H           -- Hamiltonian
        delta       -- smoothing parameter for entropy gradient
        """
        self.energy = E
        self.T = T
        self.omega_k = self.energy.kpointset.w
        self.comm = self.energy.kpointset.ctx().comm_k()
        self.H = H
        self.kb = (
            physical_constants["Boltzmann constant in eV/K"][0]
            / physical_constants["Hartree energy in eV"][0]
        )
        self.delta = delta
        if self.H.hamiltonian.ctx().num_mag_dims() == 0:
            self.scale = 0.5
        else:
            self.scale = 1

    def __call__(self, cn, fn):
        """
        Keyword Arguments:
        cn   -- PW coefficients
        fn   -- occupations numbers
        """

        self.energy.kpointset.fn = fn
        E, HX = self.energy.compute(cn)
        S = fermi_entropy(self.scale * fn, dd=self.delta)
        entropy_loc = (
            self.kb
            * self.T
            * np.sum(np.array(list((self.omega_k * S)._data.values())))
        )

        loc = np.array(entropy_loc, dtype=np.float64)
        entropy = np.array(0.0, dtype=np.float64)
        MPI.COMM_WORLD.Allreduce([loc, MPI.DOUBLE], [entropy, MPI.DOUBLE], op=MPI.SUM)
        return E + np.asscalar(entropy), HX


def chemical_potential(ek, T, kset):
    """
    Keyword Arguments:
    ek   -- band energies
    kset -- kpointset
    """

    # TODO cleanup directory structure

    ne = kset.ctx().unit_cell().num_valence_electrons()
    m = kset.ctx().max_occupancy()

    comm = kset.ctx().comm_k()
    kw = kset.w
    vek = np.hstack(comm.allgather(ek.to_array()))
    vkw = deepcopy(ek)
    for k in vkw._data.keys():
        vkw[k] = np.ones_like(vkw[k]) * kw[k]
    vkw = np.hstack(comm.allgather(vkw.to_array()))

    # update occupation numbers
    mu = find_chemical_potential(
        lambda mu: ne - np.sum(vkw * fermi_function(vek, T, mu, m)), mu0=0
    )

    return mu


class CG:
    def __init__(self, free_energy):
        """
        Keyword Arguments:
        free_energy    --
        """
        self.free_energy = free_energy
        self.T = free_energy.T
        self.dd = 1e-5

    def run(self, X, fn, maxiter=10):
        from ..edft.neugebaur import make_kinetic_precond

        kset = self.free_energy.energy.kpointset

        kw = kset.w

        K = make_kinetic_precond(kset, eps=0.001)
        F, Hx = self.free_energy(X, fn)

        HX = Hx * kw
        Hij = X.H @ HX
        LL = Hij * fn
        g_X = HX * fn - X @ LL
        dX = -g_X  # delta_X
        G_X = dX

        for i in range(maxiter):
            X, fn, F, Hx = self.step_X(X, fn, F, G_X, g_X)
            logger("step %4d: %.10f" % (i, F))

            # inner loop (optimize fn)
            X, fn, F, Hx, U = self.step_fn(X, fn, num_iter=2)
            logger("step %4d: %.10f" % (i, F))

            # compute new search direction
            HX = Hx * kw
            Hij = X.H @ HX
            XhKHXF = X.H @ (K @ HX)
            XhKX = X.H @ (K @ X)
            LL = _solve(XhKX, XhKHXF)

            # previous search directions (including subspace rotation)
            gXp = g_X @ U
            dXp = dX @ U

            g_X = HX * fn - X @ LL
            dX = -K * (HX - X @ LL) / kw
            # conjugate directions

            beta_cg = max(0, np.real(inner(g_X, dX)) / np.real(inner(gXp, dXp)))
            G_X = dX + beta_cg * (gXp - X @ (X.H @ gXp))

        return X, fn

    def step_X(self, X, fn, F0, G_X, g_X):
        slope = 2*inner(G_X, g_X)
        tt = 0.2

        F1, Hx1 = self.free_energy(X + tt*G_X, fn)
        c = F0
        b = slope
        a = (F1-b*tt) / tt**2
        t_min = -b/(2*a)

        X = X + t_min * G_X

        F, Hx = self.free_energy(X, fn)
        Fpred = a * t_min**2 + b * t_min + c

        logger('predicition error:  %.10f' % F - Fpred)

        return X, fn, F, Hx

    def step_fn(self, X, fn, num_iter=2):
        from ..coefficient_array import diag, ones_like

        kset = self.free_energy.energy.kpointset
        kw = kset.w
        kT = self.free_energy.T * self.free_energy.kb

        # subspace rotation, update matrix
        Um = diag(ones_like(fn))
        for i in range(num_iter):
            F0, Hx = self.free_energy(X, fn)
            Hij = X.H @ Hx
            ek, U = Hij.eigh()
            # find mu
            mu = chemical_potential(ek, self.T, kset)

            m = kset.ctx().max_occupancy()
            fn_tilde = fermi_function(ek, self.T, mu, m)
            fij_tilde = U @ spdiag(fn_tilde) @ U.H

            dfn = fij_tilde - diag(fn)

            # quadratic line search
            # obtain free energy at trial point
            Ftrial, _ = self.free_energy(X @ U, fn_tilde)
            scale = self.free_energy.scale
            # get derivative
            dAdfij = kw * Hij + kw * kT * scale * diag(
                df_fermi_entropy(scale * fn, dd=self.dd)
            )
            slope = inner(dAdfij, dfn)

            # get new minimum, -> fn
            c = F0
            b = slope
            a = Ftrial - b - c
            beta_min = -b / (2 * a)

            fn += beta_min * dfn
            # diagonalize fn and rotate X accordingly
            fn, U = fn.eigh()
            X = X @ U
            Um = Um @ U
        # evaluate at destination point
        F, HX = self.free_energy(X, fn)
        return X, fn, F, Hx, Um
