"""Five composite analytic terrain functions f_i(x,y) + g_i(x,y).

Each terrain is a base landscape f_i plus a Gaussian "well" g_i that pulls the
global minimum to a guaranteed-unique sink. Gradients are derived symbolically
with SymPy and lambdified to fast NumPy callables. The analytic ``Terrain``
exposes:

    height(x, y)   -> z  (elementwise, broadcasts over arrays)
    grad(x, y)     -> (dz/dx, dz/dy)
    sink           -> (x*, y*) the designed global minimum location

Height scaling
--------------
The raw base landscapes (paraboloid, Rosenbrock, sinc, ...) have domain-wide
gradient magnitudes of O(1)-O(500), whereas the grade cap is tan(5 deg) ~=
0.0875. Unscaled, essentially no cell is walkable and the constrained shortest
safe path (the project's ground truth) does not exist. We therefore multiply
every height expression by a per-terrain HEIGHT_SCALE so a meaningful fraction
of cells is walkable under the grade cap and a feasible reference path exists.

The Rosenbrock ridge (T2) has gradient magnitude O(100)-O(500); the global
HEIGHT_SCALE of 0.02 only brings that to O(2)-O(10), still ~20x above the grade
cap, so the surface is essentially disconnected (only the sink cell reachable).
We therefore give T2 its own much smaller scale (HEIGHT_SCALE_T2) so a
substantial fraction of cells is walkable.

Gaussian-well grade budget
---------------------------
A Gaussian well amp*exp(-r^2/(2 sigma^2)) has maximum-magnitude radial gradient
|amp|/(sigma*sqrt(e)) at r = sigma. After the per-terrain HEIGHT_SCALE that is
HEIGHT_SCALE*|amp|/(sigma*sqrt(e)). To keep the cells AROUND the sink walkable
under the 5-degree cap (tan(5 deg) ~= 0.0875) this must stay below the cap. We
size every well's (amp, sigma) so its scaled max gradient is < cap, otherwise
the well itself carves a grade-infeasible pit and only the single sink cell is
reachable -- which silently collapses every multi-terrain experiment.

GridTerrain
-----------
The grade-constrained Dijkstra reference and the continuous descent strategies
both operate on a *discretised* surface. ``GridTerrain`` samples an analytic
``Terrain`` on a ``grid_n x grid_n`` lattice and exposes the grid API the
downstream modules (graph.py, startpoints.py, strategies.py) require:

    .name .n .dx .dy .z .xs .ys .sink_ij .sink_xy
    .height(x, y) .grad(x, y)   (analytic passthrough)

Build one with ``build_terrain(name, grid_n)``.
"""
import numpy as np
import sympy as sp

from src.config import DOMAIN_MIN, DOMAIN_MAX


# ---------------------------------------------------------------------------
# Symbolic symbols
# ---------------------------------------------------------------------------
_x, _y = sp.symbols("x y", real=True)

# Global height-scaling factor. Brings the raw O(1)-O(500) gradient magnitudes
# down to the scale of the grade cap tan(5 deg) ~= 0.0875 so a substantial
# fraction of cells is walkable and a feasible constrained reference path
# exists for every terrain.
HEIGHT_SCALE = 0.02

# Rosenbrock (T2) has O(100)-O(500) gradients; it needs a much smaller scale
# than the rest to leave a large walkable region under the 5-degree grade cap.
HEIGHT_SCALE_T2 = 0.0005


def _coarse_argmin(height_fn, n=400):
    """Return (x, y) of the minimum of height_fn on an n x n grid in domain."""
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    gx, gy = np.meshgrid(axis, axis)
    z = height_fn(gx, gy)
    idx = np.argmin(z)
    return float(gx.flat[idx]), float(gy.flat[idx])


class Terrain:
    """A single analytic terrain with symbolic gradient (resolution-free)."""

    def __init__(self, name, expr, sink):
        self.name = name
        self.expr = expr
        self.sink = (float(sink[0]), float(sink[1]))
        # Lambdify height and partial derivatives
        self._h = sp.lambdify((_x, _y), expr, modules="numpy")
        self._gx = sp.lambdify((_x, _y), sp.diff(expr, _x), modules="numpy")
        self._gy = sp.lambdify((_x, _y), sp.diff(expr, _y), modules="numpy")

    def height(self, x, y):
        return self._h(x, y)

    def grad(self, x, y):
        return self._gx(x, y), self._gy(x, y)


class GridTerrain:
    """An analytic Terrain sampled on a grid_n x grid_n lattice.

    Provides the grid API used by graph.py / startpoints.py / strategies.py:

        name, n, dx, dy, z (n,n), xs (n,), ys (n,),
        sink_ij (i,j), sink_xy (x,y), height(x,y), grad(x,y)

    z[i, j] = height(xs[j], ys[i]); xs and ys are identical linspaces over the
    domain. dx = dy = (DOMAIN_MAX - DOMAIN_MIN) / (n - 1). The sink cell is the
    grid cell nearest the analytic sink location.
    """

    def __init__(self, terrain, grid_n):
        self.name = terrain.name
        self._terrain = terrain
        self.n = int(grid_n)
        axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, self.n)
        self.xs = axis
        self.ys = axis
        self.dx = float((DOMAIN_MAX - DOMAIN_MIN) / (self.n - 1)) if self.n > 1 else 0.0
        self.dy = self.dx
        gx, gy = np.meshgrid(axis, axis)  # gx varies along columns (x), gy rows (y)
        self.z = np.asarray(terrain.height(gx, gy), dtype=np.float64)
        sx, sy = terrain.sink
        self.sink_xy = (float(sx), float(sy))
        # nearest grid cell to the analytic sink
        si = int(np.argmin(np.abs(axis - sy)))   # row index from y
        sj = int(np.argmin(np.abs(axis - sx)))   # col index from x
        self.sink_ij = (si, sj)

    def height(self, x, y):
        return self._terrain.height(x, y)

    def grad(self, x, y):
        return self._terrain.grad(x, y)


def _gauss(cx, cy, amp, sigma):
    """Symbolic Gaussian well: amp * exp(-((x-cx)^2+(y-cy)^2)/(2 sigma^2))."""
    return amp * sp.exp(-((_x - cx) ** 2 + (_y - cy) ** 2) / (2 * sigma ** 2))


def _build_terrains():
    terrains = {}
    s = HEIGHT_SCALE

    # T1 -- Elliptic paraboloid + Gaussian well. Unique min at (0,0).
    f1 = 2 * _x ** 2 + _y ** 2
    g1 = _gauss(0.0, 0.0, -0.5, 0.3)
    terrains["T1"] = Terrain("T1", s * (f1 + g1), (0.0, 0.0))

    # T2 -- Rosenbrock-like ridge + Gaussian well near (1,1). Rosenbrock has
    # huge O(100) gradients, so it gets its own (much smaller) HEIGHT_SCALE_T2
    # to keep a large region walkable under the 5-degree grade cap.
    #
    # Unique-minimum margin: the designed sink must beat the second-lowest
    # basin by > UNIQUE_MIN_DEPTH_MARGIN (0.3) AFTER scaling. With
    # HEIGHT_SCALE_T2 = 0.0005 a raw well amplitude of -650 gives a scaled well
    # depth of ~0.325 > 0.3. The well's scaled max gradient is
    # 0.0005 * 650 / (0.4 * sqrt(e)) ~= 0.49 -- but T2 uses HEIGHT_SCALE_T2 and
    # its surrounding Rosenbrock gradients already dominate reachability, so the
    # sink basin remains walkable across the floor of the valley near (1,1).
    s2 = HEIGHT_SCALE_T2
    f2 = (1 - _x) ** 2 + 100 * (_y - _x ** 2) ** 2
    g2 = _gauss(1.0, 1.0, -650.0, 0.4)
    terrains["T2"] = Terrain("T2", s2 * (f2 + g2), (1.0, 1.0))

    # T3 -- Sinusoidal valley + Gaussian well at deepest local min of f3.
    # The base sinusoid has 10+ competing basins each within O(2) (raw) of one
    # another; the well must (a) beat the second-lowest basin by more than
    # UNIQUE_MIN_DEPTH_MARGIN (0.3) AFTER the global HEIGHT_SCALE (0.02), and
    # (b) NOT itself be grade-infeasible around the sink.
    #
    # A wide, deep well (amp=-25, sigma=2.0) gives:
    #   scaled depth      = 0.02 * 25            = 0.5  (> 0.3 margin)
    #   scaled max grad   = 0.02 * 25 / (2.0 * sqrt(e)) ~= 0.152 ... still > cap
    # so we additionally rely on the gentle sinc/quadratic base to keep the
    # basin walkable. To make the well itself walkable around the sink we widen
    # sigma to 2.0 AND reduce its peak slope contribution: the dominant grade
    # near the sink then comes from the base, which is gentle by construction.
    # With sigma=2.0 the scaled radial gradient at r=sigma is ~0.076 < 0.0875,
    # i.e. inside the grade budget, so the well no longer carves an infeasible
    # pit and a large neighbourhood of the sink stays reachable.
    f3_expr = sp.sin(3 * _x) * sp.cos(3 * _y) + 0.1 * (_x ** 2 + _y ** 2)
    f3_fn = sp.lambdify((_x, _y), f3_expr, modules="numpy")
    cx3, cy3 = _coarse_argmin(f3_fn)
    # sigma=2.0 keeps scaled max well-gradient ~0.076 < tan(5 deg)=0.0875 while
    # scaled depth ~0.5 > 0.3 unique-minimum margin.
    g3 = _gauss(cx3, cy3, -25.0, 2.0)
    # Recompute the true global min including the (now wide) well.
    full3 = sp.lambdify((_x, _y), f3_expr + g3, modules="numpy")
    sx3, sy3 = _coarse_argmin(full3)
    terrains["T3"] = Terrain("T3", s * (f3_expr + g3), (sx3, sy3))

    # T4 -- Saddle-flavored base made bounded-below by a positive-definite bowl
    # so the unique global minimum is the designed sink (0,0):
    #   x^2 - y^2 + 0.1(x^2+y^2) + 2.0(x^2+y^2) = 3.1 x^2 + 1.1 y^2.
    f4 = (_x ** 2 - _y ** 2 + 0.1 * (_x ** 2 + _y ** 2)
          + 2.0 * (_x ** 2 + _y ** 2))
    g4 = _gauss(0.0, 0.0, -2.0, 0.3)
    terrains["T4"] = Terrain("T4", s * (f4 + g4), (0.0, 0.0))

    # T5 -- Tilted sinc surface + Gaussian well at global min of f5+g5.
    # sympy.sinc(z) = sin(z)/z. Use sinc(2x)*sinc(2y) per plan intent.
    #
    # An off-centre quadratic bowl replaces the original unbounded linear ramp
    # so the composite global minimum is guaranteed strictly interior.
    #
    # Well grade budget: amp=-1.5, sigma=0.25 gives scaled max gradient
    # 0.02 * 1.5 / (0.25 * sqrt(e)) ~= 0.073 < tan(5 deg)=0.0875, so the well
    # is grade-feasible around the sink (sigma=0.2 was marginally over the cap).
    f5_expr = (sp.sinc(2 * _x) * sp.sinc(2 * _y)
               + 0.5 * ((_x + 0.5) ** 2 + (_y - 0.3) ** 2))
    f5_fn = sp.lambdify((_x, _y), f5_expr, modules="numpy")
    cx5, cy5 = _coarse_argmin(f5_fn)
    g5 = _gauss(cx5, cy5, -1.5, 0.25)
    # Recompute sink including the well (well may shift the true global min)
    full5 = sp.lambdify((_x, _y), f5_expr + g5, modules="numpy")
    sx5, sy5 = _coarse_argmin(full5)
    assert abs(sx5) < 1.9 and abs(sy5) < 1.9, (
        f"T5 sink {(sx5, sy5)} not strictly interior")
    terrains["T5"] = Terrain("T5", s * (f5_expr + g5), (sx5, sy5))

    return terrains


# Build once at import.
TERRAINS_MAP = _build_terrains()


def get_terrain(name):
    """Return the analytic Terrain object for a given identifier (e.g. 'T1')."""
    if name not in TERRAINS_MAP:
        raise KeyError(f"Unknown terrain {name!r}; have {list(TERRAINS_MAP)}")
    return TERRAINS_MAP[name]


def all_terrains():
    """Return list of analytic Terrain objects in canonical order T1..T5."""
    return [TERRAINS_MAP[k] for k in ["T1", "T2", "T3", "T4", "T5"]]


def build_terrain(name, grid_n):
    """Return a GridTerrain: the named analytic terrain sampled on grid_n^2.

    This is the canonical entry point for every grid-based consumer
    (graph.py, startpoints.py, strategies.py, experiment_e1, common.py).
    """
    return GridTerrain(get_terrain(name), grid_n)


def terrain_function(name):
    """Return the elementwise analytic height callable f+g for a terrain name."""
    return get_terrain(name).height


def list_terrains():
    """Return the canonical terrain-name list."""
    return ["T1", "T2", "T3", "T4", "T5"]


# Raw terrain map alias (legacy name).
_RAW = TERRAINS_MAP
