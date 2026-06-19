"""Shared configuration constants for the limited-descent E0 baseline.

All values anchor on the project experiment defaults and the methodology plan.
No external datasets: every input here is analytic / generated in-code.
"""
import math

# Spatial domain for all terrains: x, y in [-2, 2]
DOMAIN_MIN = -2.0
DOMAIN_MAX = 2.0

# Grid resolutions (number of nodes per axis)
GRID_PRIMARY = 1000      # primary Theta* / Dijkstra grid
GRID_VALIDATION = 500    # resolution-convergence validation grid
GRID_UNIQUEMIN = 2000    # grid for unique-minimum verification
GRID_ESCALATE = 2000     # escalation grid if convergence check fails

# Grade constraint: maximum walkable grade in degrees.
MAX_GRADE_DEGREES = 5.0
# Slope threshold = tan(5 deg). Edge removed if |dz|/dl_horizontal > this.
MAX_GRADE_SLOPE = math.tan(math.radians(MAX_GRADE_DEGREES))

# Heuristic / descent parameters (used by later experiments; defined here so
# the whole project shares one source of truth).
STEP_LENGTH = 0.1
ROTATION_INCREMENT_DEGREES = 1.0
N_INITIAL_POINTS = 8           # plan uses 8 start points per terrain

# Start-point sampling box (kept inside domain to avoid boundary effects)
START_MIN = -1.8
START_MAX = 1.8

# Terrain identifiers (analytic suite T1..T5)
TERRAINS = ["T1", "T2", "T3", "T4", "T5"]
SYMMETRIC_TERRAINS = ["T1", "T4"]
ASYMMETRIC_TERRAINS = ["T2", "T3", "T5"]

# Reference / random seed
DEFAULT_SEED = 42

# E0 pass/fail thresholds (from stop_conditions in the plan)
RESOLUTION_TOL_FRAC = 0.02         # 500 vs 1000 must agree within 2%
RESOLUTION_ESCALATE_FRAC = 0.03    # > 3% escalates that terrain
GRAD_FD_TOL = 1e-4                 # analytic vs finite-difference tolerance
UNIQUE_MIN_DEPTH_MARGIN = 0.3      # well must beat 2nd-lowest local min by >= 0.3
N_GRAD_CHECK_POINTS = 100          # random points for gradient check per terrain

# T1 baseline sanity (internal-consistency baseline)
T1_BASELINE_START = (1.0, 0.0)
T1_BASELINE_SINK = (0.0, 0.0)
T1_BASELINE_EXPECTED_LEN = 2.05
T1_BASELINE_TOL = 0.10

# Output directory for structured logs
RUNS_DIR = "runs"

# ---------------------------------------------------------------------------
# Backward-compatibility aliases for legacy modules/tests that predate the
# canonical names above. Each maps to a real, already-defined constant.
# ---------------------------------------------------------------------------
MAX_GRADE_TAN = MAX_GRADE_SLOPE          # legacy name for tan(5 deg)
GRID_N = GRID_PRIMARY                    # legacy single-grid resolution
START_SEED = DEFAULT_SEED                # legacy start-sampling seed
N_START_POINTS = N_INITIAL_POINTS        # legacy start-point count
N_QUINTILES = N_INITIAL_POINTS           # legacy LHS stratum count
T1_GEODESIC_TOL = T1_BASELINE_TOL        # legacy T1 tolerance name
