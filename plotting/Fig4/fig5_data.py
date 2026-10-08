"""Shared I/O + melt-fraction physics for Fig. 5 (partial melting).

Data layout of the i2vis output in ``Fig5_MeltingCalc_80km``:

* ``goucNNN.txt`` – run-length-encoded composition on a uniform 6001 x 4001 grid
  covering 0-800 km (x) and 0-600 km (z, model coordinate incl. 10 km sticky air)
* ``goutNNN.txt`` – temperature (K) on the 501 x 376 node grid (1.6 km spacing)
* ``goupNNN.txt`` – pressure (Pa) on the same node grid

Plot depth = model z - Z_OFFSET.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RUN = BASE / "Fig5_MeltingCalc_80km"

X_SIZE, Z_SIZE = 800.0, 600.0
Z_OFFSET = 10.0  # sticky-air thickness removed when plotting

# Rock ids used by this model
ID_AIR = 0
ID_UPPER_CRUST = 5
ID_LOWER_CRUST = 6
ID_WEAK_ML = 8
ID_ASTH = 10
ID_ECLOGITE = (12, 15)
ID_PYROXENITE = (13, 14)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_time_myr(path: Path) -> float:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return float(f.read(80).split()[0]) / 1.0e6


def decode_composition(path: Path) -> tuple[float, np.ndarray]:
    """Decode an RLE composition file into (time_yr, grid[nz, nx])."""
    data = np.fromfile(path, sep=" ", dtype=np.float64)
    if data.size < 3:
        raise ValueError(f"Empty composition file: {path}")
    time = float(data[0])
    nx, nz = int(data[1]), int(data[2])
    vec = data[3:]

    flat = np.ones(nz * nx, dtype=np.float64)
    num = ind = 0
    n = vec.size
    while num < n and ind < flat.size:
        value = vec[num]
        if value == -2:
            k = int(vec[num + 1])
            flat[ind : ind + k] = vec[num + 2]
            ind += k
            num += 3
        else:
            flat[ind] = np.nan if value == -1 else value
            ind += 1
            num += 1
    return time, flat.reshape((nz, nx), order="F")


def read_node_field(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Read a node-based scalar field → (time_yr, x_km, z_km, field[nz, nx])."""
    d = np.fromfile(path, sep=" ", dtype=np.float64)
    time = float(d[0])
    nx, nz = int(d[1]), int(d[2])
    x = d[3 : 3 + nx] / 1000.0
    z = d[3 + nx : 3 + nx + nz] / 1000.0
    v = d[3 + nx + nz : 3 + nx + nz + nx * nz].reshape((nz, nx), order="F")
    return time, x, z, v


def repair_pressure_boundary(p: np.ndarray) -> np.ndarray:
    """Pressure lives on the staggered (cell-centre) grid.

    The first row/column of the exported array carries half the neighbouring
    value and the last row/column is identically zero; both are meaningless.
    Replace them with their inner neighbours so interpolation stays valid right
    up to the model edges.
    """
    q = np.array(p, copy=True)
    q[0, :] = q[1, :]
    q[-1, :] = q[-2, :]
    q[:, 0] = q[:, 1]
    q[:, -1] = q[:, -2]
    return q


def read_pressure_gpa(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, xn, zn, pres = read_node_field(RUN / f"goup{step}.txt")
    return xn, zn, repair_pressure_boundary(pres) / 1.0e9


def load_step(step: int) -> dict:
    """Composition + T (deg C) + P (GPa) for one output step."""
    t_yr, comp = decode_composition(RUN / f"gouc{step}.txt")
    _, xn, zn, temp = read_node_field(RUN / f"gout{step}.txt")
    _, _, _, pres = read_node_field(RUN / f"goup{step}.txt")
    pres = repair_pressure_boundary(pres)

    nz, nx = comp.shape
    return {
        "step": step,
        "time_myr": t_yr / 1.0e6,
        "comp": comp,
        "xc": np.linspace(0.0, X_SIZE, nx),
        "zc": np.linspace(0.0, Z_SIZE, nz),
        "xn": xn,
        "zn": zn,
        "T": temp - 273.0,
        "P": pres / 1.0e9,
    }


# ---------------------------------------------------------------------------
# Solidus / liquidus (identical to vis_cto3_Melt_Wet.m and PlotP-T-New.py)
# ---------------------------------------------------------------------------
def liquidus(p_gpa):
    """Peridotite liquidus, deg C."""
    p = np.asarray(p_gpa, dtype=float)
    return np.where(
        p < 10.0,
        1566.9 + 38.478 * p + 1.6237 * p**2 - 273.0,
        1440.0 + 67.3 * p - 273.0,
    )


def solidus_wet(p_gpa):
    """Wet (hydrated) peridotite solidus, deg C."""
    p = np.asarray(p_gpa, dtype=float)
    return np.where(
        p < 5.0,
        -4.48 * p**2 + 102.8 * p + 1174.0 - 273.0,
        -0.63 * p**2 + 59.45 * p + 1295.3 - 273.0,
    )


def solidus_dry(p_gpa):
    """Dry peridotite solidus, deg C."""
    p = np.asarray(p_gpa, dtype=float)
    return np.where(
        p <= 10.0,
        1190.0 + 102.34 * p - 2.217 * p**2 - 273.0,
        1323.0 + 66.9 * p - 273.0,
    )


def solidus_saturated(p_gpa):
    """Water-saturated basalt/eclogite solidus, deg C."""
    p = np.asarray(p_gpa, dtype=float)
    return np.select(
        [p < 1.0, p < 2.0, p < 3.8],
        [
            -178.9 * p + 1141.3,
            np.full_like(p, 962.4),
            18.222 * p**2 - 70.911 * p + 1031.333,
        ],
        default=49.28 * p + 837.73,
    )


def solidus_pyroxenite(p_gpa):
    """Solidus used for eclogite / pyroxenite: saturated solidus shifted -150 C."""
    return solidus_saturated(p_gpa) - 150.0


def solidus_crust(p_gpa):
    """Wet crustal solidus, deg C (P given in GPa, formula uses MPa)."""
    p_mpa = np.asarray(p_gpa, dtype=float) * 1000.0
    return np.where(
        p_mpa < 1600.0,
        973.0 - 70400.0 / (p_mpa + 354.0) + 77.8e6 / (p_mpa + 354.0) ** 2,
        935.0 + 0.0035 * p_mpa + 0.0000062 * p_mpa**2,
    )


def liquidus_crust(p_gpa):
    return 1423.0 + 0.105 * np.asarray(p_gpa, dtype=float) * 1000.0


# ---------------------------------------------------------------------------
# Melt fraction field
# ---------------------------------------------------------------------------
def melt_fraction(comp: np.ndarray, T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Linear-between-solidus-and-liquidus melt fraction, per rock type.

    ``comp``, ``T`` and ``P`` must already share the same grid; T in deg C,
    P in GPa.  Mirrors the branch structure of vis_cto3_Melt_Wet.m.
    """
    m = np.zeros_like(T, dtype=float)

    # Mantle (asthenosphere + hydrated mantle lithosphere) — wet solidus
    sel = np.isin(comp, [ID_ASTH, ID_WEAK_ML])
    if sel.any():
        ts = solidus_wet(P[sel])
        tl = liquidus(P[sel])
        m[sel] = np.clip((T[sel] - ts) / (tl - ts), 0.0, 1.0)

    # Eclogite + pyroxenite — solidus and liquidus both shifted by -150 C
    sel = np.isin(comp, list(ID_ECLOGITE) + list(ID_PYROXENITE))
    if sel.any():
        ts = solidus_pyroxenite(P[sel])
        tl = liquidus(P[sel]) - 150.0
        m[sel] = np.clip((T[sel] - ts) / (tl - ts), 0.0, 1.0)

    # Crust — wet crustal solidus
    sel = np.isin(comp, [ID_UPPER_CRUST, ID_LOWER_CRUST])
    if sel.any():
        ts = solidus_crust(P[sel])
        tl = liquidus_crust(P[sel])
        m[sel] = np.clip((T[sel] - ts) / (tl - ts), 0.0, 1.0)

    return m


def melt_on_composition_grid(data: dict, stride: int = 4) -> tuple:
    """Melt fraction on a (strided) composition grid.

    Returns (x_km, depth_km, melt, comp_sub).
    """
    comp = data["comp"][::stride, ::stride]
    xc = data["xc"][::stride]
    zc = data["zc"][::stride]

    # Node grid is uniform → nearest/linear interpolation via index arithmetic
    xn, zn = data["xn"], data["zn"]
    T = _bilinear(xn, zn, data["T"], xc, zc)
    P = _bilinear(xn, zn, data["P"], xc, zc)

    m = melt_fraction(comp, T, P)
    return xc, zc - Z_OFFSET, m, comp


def _bilinear(xn, zn, field, xq, zq):
    """Bilinear sampling of a node field onto the outer product of (xq, zq)."""
    ix = np.clip(np.searchsorted(xn, xq) - 1, 0, len(xn) - 2)
    iz = np.clip(np.searchsorted(zn, zq) - 1, 0, len(zn) - 2)
    tx = np.clip((xq - xn[ix]) / (xn[ix + 1] - xn[ix]), 0.0, 1.0)
    tz = np.clip((zq - zn[iz]) / (zn[iz + 1] - zn[iz]), 0.0, 1.0)

    f00 = field[np.ix_(iz, ix)]
    f01 = field[np.ix_(iz, ix + 1)]
    f10 = field[np.ix_(iz + 1, ix)]
    f11 = field[np.ix_(iz + 1, ix + 1)]
    TX = tx[None, :]
    TZ = tz[:, None]
    return (
        f00 * (1 - TX) * (1 - TZ)
        + f01 * TX * (1 - TZ)
        + f10 * (1 - TX) * TZ
        + f11 * TX * TZ
    )


def sample_point(xn, zn, field, x, z) -> float:
    return float(_bilinear(xn, zn, field, np.array([x]), np.array([z]))[0, 0])


def available_steps() -> list[int]:
    out = []
    for p in RUN.glob("gouc*.txt"):
        s = p.stem[4:]
        if s.isdigit() and p.stat().st_size > 100:
            out.append(int(s))
    return sorted(out)
