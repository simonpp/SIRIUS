"""
Conjugate gradient method for Marzari, Vanderbilt, Payne:
- Marzari, N., Vanderbilt, D., & Payne, M. C. (). Ensemble Density-Functional
  Theory for Ab Initio Molecular Dynamics of Metals and Finite-Temperature
  Insulators. , 79(7), 1337–1340. http://dx.doi.org/10.1103/PhysRevLett.79.1337
"""

from collections import namedtuple
from copy import deepcopy

import numpy as np
from mpi4py import MPI
from scipy.constants import physical_constants

from ..baarman.direct_minimization import df_fermi_entropy, fermi_entropy
from ..coefficient_array import diag, inner, spdiag
from ..edft.neugebaur import fermi_function, find_chemical_potential, kb
from ..edft.ortho import loewdin
from ..helpers import save_state
from ..logger import Logger

logger = Logger()


def _solve(A, X):
    """
    returns A⁻¹ X
    """
    out = type(X)(dtype=X.dtype, ctype=X.ctype)
    for k in X.keys():
        out[k] = np.linalg.solve(A[k], X[k])
    return out


class StepError(Exception):
    pass


class SlopeError(Exception):
    pass


def btsearch(f, b, f0=None, maxiter=20, tau=0.5):
    """
    Backtracking search

    Arguments:
    f  -- function f(x)
    b  -- end point
    f0 -- f(0)
    """

    x = b
    for i in range(maxiter):
        fx = f(x)
        if fx.F > f0:
            x *= tau
        else:
            return x, fx
    raise ValueError('backtracking search could not find a new minimum')


class LineEvaluator():
    """
    Evaluate free energy along a fixed direction.
    """

    def __init__(self, X, fn, M, G_X):
        """
        Keyword Arguments:
        X     -- plane-wave coefficients
        fn    -- occupation numbers
        M     -- free energy evaluator
        G_X   -- direction X
        """
        self.fn = fn
        self.X = X
        self.M = M
        # search direction
        self.G_X = G_X

    def __call__(self, t):
        """
        Evaluate along line

        Returns:
        Named tuple with entries
        F  -- free energy
        Hx -- gradient (free of occupation, k-point-weight)
        X  -- plane-wave coefficients
        fn -- occupation numbers
        """
        X_new = self.X + t * self.G_X
        X = loewdin(X_new)

        Point = namedtuple('Point', ['F', 'Hx', 'X', 'fn'])

        FE, Hx = self.M(X, self.fn)

        return Point(F=FE, Hx=Hx, X=X, fn=self.fn)


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
    fd_slope_check = False

    def __init__(self, free_energy, fd_slope_check=False):
        """
        Keyword Arguments:
        free_energy    --
        """
        self.fd_slope_check = fd_slope_check
        self.free_energy = free_energy
        self.T = free_energy.T
        self.dd = 1e-5

    def run(self, X, fn,
            tol=1e-9,
            maxiter=100,
            ncgrestart=20,
            ninner=2,
            tau=0.3):
        """
        Keyword Arguments:
        self       --
        X          --
        fn         --
        tol        --
        maxiter    --
        ncgrestart --
        tau        --
        """

        from ..edft.neugebaur import make_kinetic_precond

        kset = self.free_energy.energy.kpointset
        kw = kset.w
        K = make_kinetic_precond(kset, eps=0.001)
        F, Hx = self.free_energy(X, fn)
        logger('initial free energy: %.10f' % F)

        HX = Hx * kw
        XhKHXF = X.H @ (K @ HX)
        XhKX = X.H @ (K @ X)
        LL = _solve(XhKX, XhKHXF)

        g_X = HX * fn - X @ LL
        dX = -K * (HX - X @ LL) / kw
        G_X = dX

        for i in range(maxiter):
            try:
                X, fn, F, Hx, slope_X = self.stepX(X, fn, F, G_X, g_X, tol=tol)
            except (StepError, SlopeError):
                fline = LineEvaluator(X=X, fn=fn, M=self.free_energy, G_X=G_X)
                res = fline(0)
                F = res.F

                logger('--- CG RESTART ---')
                G_X = dX  # un-conjugated search direction
                X, fn, F, Hx, slope_X = self.stepX(X, fn, F, G_X, g_X,
                                                   tol=tol, tau=tau)

                # _, fx = btsearch(fline, b=1, f0=F, maxiter=20, tau=0.2)
                # X = fx.X
                # fn = fx.fn
                # F = fx.F
            logger("  stepX %4d: %.10f" % (i, F))

            # inner loop (optimize fn)
            X, fn, F, Hx, U, slope_fn = self.step_fn(X, fn, tol=tol, num_iter=ninner)
            logger("step %5d F: %.11f res: X,fn %+10.5e %+10.5e" % (i, F, slope_X, slope_fn))

            if np.abs(slope_fn) + 2*np.abs(slope_X) < tol:
                return F, X, fn, True

            dX, G_X, g_X = self.conjugate_directions(K=K, X=X, fn=fn,
                                                     Hx=Hx,
                                                     G_X=G_X,
                                                     g_X=g_X,
                                                     dX=dX,
                                                     U=U,
                                                     restart=(i % ncgrestart) == 0)

        return F, X, fn, False

    def conjugate_directions(self, K, X, fn, Hx, G_X, g_X, dX, U, restart):
        """
        Keyword Arguments:
        K       -- wfct preconditioner
        X       -- pw coefficients
        fn      -- occupation numbers
        Hx      -- H@X
        G_X     -- search direction
        g_X     -- gradient
        dX      -- preconditioned gradient
        U       -- subspace rotation from inner loop
        restart -- true|false
        """

        kw = self.free_energy.energy.kpointset.w
        # compute new search direction
        HX = Hx * kw
        XhKHXF = X.H @ (K @ HX)
        XhKX = X.H @ (K @ X)
        LL = _solve(XhKX, XhKHXF)

        # previous search directions (including subspace rotation)
        gXp = g_X @ U
        dXp = dX @ U

        g_X = HX * fn - X @ LL
        dX = -K * (HX - X @ LL) / kw
        # conjugate directions
        if restart:
            beta_cg = 0
            G_X = dX
        else:
            beta_cg = max(0, np.real(inner(g_X, dX)) / np.real(inner(gXp, dXp)))
            G_X = dX + beta_cg * (gXp - X @ (X.H @ gXp))

        logger('beta_cg: %.6f' % beta_cg)
        # beta_cg = 0
        return dX, G_X, g_X

    def stepX(self, X, fn, F0, G_X, g_X, tol, tau=0.3):
        slope = 2*np.real(inner(G_X, g_X))

        if slope > 0:
            raise SlopeError

        line_eval = LineEvaluator(X, fn, self.free_energy, G_X)

        if np.abs(slope) < tol:
            return X, fn, F0, None, slope
        try:
            X, fn, F, Hx = self.stepX_quadratic(F0, line_eval, slope)
        except StepError:
            # fallback to backtracking search
            logger('VERBOSE:: BACKTRACKING SEARCH')
            X, fn, F, Hx = self.stepX_btsearch(F0, line_eval, tau=tau)

        return X, fn, F, Hx, slope

    def stepX_quadratic(self, F0, T, slope):
        """
        Arguments:
        T -- transport
        """
        tt = 0.2
        if self.fd_slope_check:
            _dt = 1e-7
            fx = T(_dt)
            slope_fd = (fx.F - F0) / _dt
            logger('VERRBOSE slope: %.6f, slope_fd: %.6f'  % (slope, slope_fd))
        b = slope
        c = F0
        while True:
            F1, Hx1, _, _ = T(tt)
            if not b < 0:
                raise SlopeError
            a = (F1-b*tt-c) / tt**2
            t_min = -b/(2*a)
            if a < 0:
                # increase tt
                tt *= 5
                logger(' ... increase trial point  %.2f' % tt)
            else:
                break

        logger('VERBOSE:: stepX: t_min = %.4f' % t_min)
        # evaluate at destination point
        F, Hx, X, fn = T(t_min)

        # compute free energy predicted by the quad. approx.
        Fpred = a * t_min**2 + b * t_min + c

        assert t_min > 0

        logger('VERBOSE:: stepX qline prediction error:  %.10f' % (F - Fpred))
        # logger('step_X:: Fpred=%.10f' % Fpred)
        # logger('step_X::     F=%.10f' % F)
        # logger('step_X::    F0=%.10f' % F0)
        if not F < F0:
            raise StepError

        return X, fn, F, Hx

    def stepX_btsearch(self, F0, T, tau):
        """
        Arguments:
        F0   -- free energy
        T    -- line evaluator
        tau  -- reduction parameter
        """
        t_min, fx = btsearch(T, 1, f0=F0, tau=tau)
        return fx.X, fx.fn, fx.F, fx.Hx


    def step_fn(self, X, fn, tol, num_iter=2):
        from ..coefficient_array import diag, ones_like

        kset = self.free_energy.energy.kpointset
        kw = kset.w
        kT = self.free_energy.T * self.free_energy.kb
        Um = diag(ones_like(fn))  # subspace rotation, update matrix
        Fpred = -1
        b = 0
        for i in range(num_iter):
            F0, Hx = self.free_energy(X, fn)
            logger('  inner %d\t%.10f\t%.10f' % (i, F0, Fpred))
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
            slope = np.real(inner(dAdfij, dfn))
            # get new minimum, -> fn
            c = F0
            b = slope
            logger('VERBOSE:: step_fn::slope=%.5e' % b)
            if np.abs(b) < tol:
                F, Hx = self.free_energy(X, fn)
                return X, fn, F, Hx, Um, b
            assert b < 0
            a = Ftrial - b - c
            beta_min = min(-b / (2 * a), 1)
            logger('VERBOSE:: beta_min:', beta_min)
            Fpred = a * beta_min**2 + b * beta_min + c

            fnij = diag(fn) + beta_min * dfn
            # diagonalize fn and rotate X accordingly
            fn, U = fnij.eigh()
            for k in fn.keys():
                fn[k] = np.where(np.isclose(fn[k], 0, atol=1e-10, rtol=1e-10), 0, fn[k])
                fn[k] = np.where(np.isclose(fn[k], 1, atol=1e-10, rtol=1e-10), 1, fn[k])
                assert np.all(np.logical_and(fn[k] >= 0, fn[k] <= 1))
            X = X @ U
            Um = Um @ U
        # evaluate at destination point
        F, Hx = self.free_energy(X, fn)
        logger('  inner %d\t%.10f\t%.10f (qline)' % (num_iter, F, Fpred))
        return X, fn, F, Hx, Um, b
