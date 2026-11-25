"""
Fig. 3 — Sequential drips.

2 models (Fig4a–b) × 3 times. Each panel = Elevation strip (Topo.py) + composition.
Style / layout adapted from plot_fig3_existence.py; elevation from Topo.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

# %% ========================================================================
# USER CONTROLS  —  edit here for Spyder debugging (Ctrl+Enter cell)
# ============================================================================
BASE = Path(__file__).resolve().parent
OUT_PNG = BASE / "Fig3_sequential_drips_NoIsostasy.png"
OUT_PDF = BASE / "Fig3_sequential_drips_NoIsostasy.pdf"
OUT_SVG = BASE / "Fig3_sequential_drips_NoIsostasy.svg"
OUT_EPS = BASE / "Fig3_sequential_drips_NoIsostasy.eps"
OUT_TIFF = BASE / "Fig3_sequential_drips_NoIsostasy.tiff"

# Display time = model_time_Myr - TIME0_MODEL_MYR  (model 50 Myr → "0 Myr")
TIME0_MODEL_MYR = 50.0

# Composition window (km). Topo CSVs were written from vis_cto3 zoom z_end=310;
# composition panels use a deeper window so both drips remain visible.
X_SIZE, Z_SIZE = 800.0, 600.0
X_BEG, X_END = 0.0, 800.0
Z_BEG, Z_END = 0.0, 550.0
Z_OFFSET = 10.0  # MATLAB sticky-air offset: z_vec = z_vec - 10

# Elevation (Fig4b Topo.py): topo CSVs are full-domain ColorGrid (nz×nx)
TOPO_Y_MAX = 600.0  # model vertical size in km (matches topo row mapping)
TOPO_SUR0 = 10.0
TOPO_X0, TOPO_XE = 0.0, 800.0
TOPO_SMOOTH = 100
# ΔElevation relative to panels a/d (col-0 baseline)
ELEV_YLIM = (-2.0, 3.0)
ELEV_YTICKS = [-1, 0, 1, 2, 3]  # scale to -2, but do not label -2
ELEV_LABELSIZE = 7
ELEV_TICKSIZE = 7
ELEV_YLABEL = r"$\Delta_{\mathrm{Elevation}}$ (km)"


@dataclass(frozen=True)
class PanelRow:
    """One figure row = one model folder, three gouc steps (cols 0–2)."""

    folder: str
    steps: tuple[int, int, int]
    # For col=2: earlier topo step used only to orient the uplift arrow
    first_drip_topo_step: int | None = None
    annotation: str | None = None  # unused placeholder
    # Left-panel water / ML labels: list of (x_km, depth_km, text)
    left_labels: tuple[tuple[float, float, str], ...] = ()
    # Composition annotations: (col, x_km, depth_km, text)
    drip_labels: tuple[tuple[int, float, float, str], ...] = ()
    # Elevation red arrows on cols 1 and 2: (col, x_km|None); None → max uplift
    elev_arrows: tuple[tuple[int, float | None], ...] = ()
    force_display: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    # Surface-basin x (km) on composition panels; None → auto from sediments
    basin_xs: tuple[float, ...] | None = None
    # Per-column elevation offset (Myr) after the composition frame.
    # None keeps the elevation on the same gouc step as the panel.
    elev_dt_myr: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )


# a/d = model 50 Myr → display 0; b=90, c=240; e=80, f=170
ROWS = [
    PanelRow(
        folder="Fig4a_50Myr_volume",
        steps=(23, 90, 240),  # 50.0 / 78.32 / 83.18 Myr
        first_drip_topo_step=90,
        drip_labels=(
            (1, 300.0, 300.0, "First Drip"),
            (2, 480.0, 300.0, "Second Drip"),
        ),
        elev_arrows=(),
        force_display=(0.0, None, None),
    ),
    PanelRow(
        folder="Fig4b_50Myrwater400ppm10ppm",
        steps=(19, 80, 170),  # 50.0 / 76.33 / 76.40 Myr
        first_drip_topo_step=80,
        left_labels=(
            (200.0, 110.0, r"400 ppm H/Si"),  # up 20 km vs 130
            (600.0, 110.0, r"10 ppm H/Si"),
        ),
        drip_labels=(
            (1, 300.0, 255.0, "First Drip"),  # up 20 km vs 275
            (2, 500.0, 320.0, "Second Drip"),  # down 20 km vs 300
        ),
        elev_arrows=(),
        force_display=(0.0, None, None),
        # Panel f composition stays at step 170; elevation is gouc221
        # (~+0.59 Myr). Later frames exceed the +3 km axis limit.
        elev_dt_myr=(None, None, 0.59),
    ),
]

MATERIAL_LEGEND = [
    (0, "Air"),
    (1, "Sediments"),
    (5, "Upper Crust"),
    (6, "Lower Crust"),
    (12, "Eclogite"),
    (13, "Pyroxenite"),
    (10, "Asthenosphere"),
    (8, "Weak Mantle\nLithosphere"),
]

# %% ========================================================================
# Style / figure geometry
# ============================================================================
FIG_W_MM = 183.0
FIG_W_IN = FIG_W_MM / 25.4
DPI = 600
N_COLS = 3
HSPACE = 0.10  # gap between model rows (fraction of panel stack)
WSPACE = 0.08
ELEV_FRAC = 0.48  # elevation height / composition height (taller strip)
ELEV_GAP_FRAC = 0.02  # gap elev↔comp within a panel stack

# Legend swatch: keep width; thickness from width for 9:6
LEGEND_SWATCH_W_PT = 18
LEGEND_SWATCH_H_PT = 12  # W * 6/9
LEGEND_LABEL_GAP = 0.08
TICK_LABELSIZE = 9
LEGEND_FONTSIZE = 8


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------
def material_cmap():
    colors = np.zeros((39, 3), dtype=float)
    colors[:] = 1.0
    colors[0] = [1.0, 1.0, 1.0]  # Air
    colors[1] = [238 / 255, 232 / 255, 170 / 255]  # Sediments
    colors[2] = colors[1]
    colors[3] = [0.9290, 0.6940, 0.1250]
    colors[4] = [0.8500, 0.3250, 0.0980]
    colors[5] = [236 / 255, 235 / 255, 194 / 255]  # Upper Crust
    colors[6] = [202 / 255, 178 / 255, 114 / 255]  # Lower Crust
    colors[7] = [168 / 255, 191 / 255, 143 / 255]  # Weak ML
    colors[8] = [168 / 255, 191 / 255, 143 / 255]  # Weak ML
    colors[10] = [245 / 255, 210 / 255, 188 / 255]  # Asthenosphere
    colors[12] = [152 / 255, 79 / 255, 49 / 255]  # Eclogite
    colors[13] = [69 / 255, 70 / 255, 94 / 255]  # Pyroxenite
    colors[14] = colors[13]
    colors[15] = colors[12]

    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, colors


def remap_for_display(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    out = np.where(out == 14, 13, out)
    out = np.where(out == 15, 12, out)
    out = np.where(out == 7, 8, out)  # unify wet/weak ML green
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_numbers(path: Path) -> np.ndarray:
    return np.fromfile(path, sep=" ", dtype=np.float64)


def read_time_myr(path: Path) -> float:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        chunk = f.read(80)
    return float(chunk.split()[0]) / 1.0e6


def decode_composition(path: Path) -> tuple[float, np.ndarray]:
    data = load_numbers(path)
    if data.size < 3:
        raise ValueError(f"Empty or unreadable composition file: {path}")
    time = float(data[0])
    nx = int(data[1])
    nz = int(data[2])
    vec = data[3:]

    flat = np.ones(nz * nx, dtype=np.float64)
    num = 0
    ind = 0
    n = len(vec)
    while num < n and ind < flat.size:
        value = vec[num]
        if value == -2:
            n_colors = int(vec[num + 1])
            material = vec[num + 2]
            flat[ind : ind + n_colors] = material
            ind += n_colors
            num += 3
        else:
            flat[ind] = np.nan if value == -1 else value
            ind += 1
            num += 1

    grid = flat.reshape((nz, nx), order="F")
    return time, grid


def load_topo_array(path: Path) -> np.ndarray:
    """Load topo CSV written by vis_cto3 (full composition ColorGrid)."""
    topo = np.loadtxt(path, delimiter=",")
    if topo.ndim != 2:
        raise ValueError(f"Unexpected topo shape in {path}: {topo.shape}")
    return topo


def _smooth_elev(
    ele: np.ndarray,
    *,
    x0: float = TOPO_X0,
    xe: float = TOPO_XE,
    window: int = TOPO_SMOOTH,
) -> tuple[np.ndarray, np.ndarray]:
    nx = ele.size
    x = x0 + (xe - x0) * np.arange(nx) / nx
    if window < 2 or window > nx:
        return x, ele
    y_smooth = np.convolve(ele, np.ones(window) / window, mode="valid")
    x_smooth = x[window - 1 :]
    return x_smooth, y_smooth


def elevation_from_topo_array(
    topo: np.ndarray,
    *,
    y_max: float = TOPO_Y_MAX,
    sur0: float = TOPO_SUR0,
) -> tuple[np.ndarray, np.ndarray]:
    """Topo.py elevation: first non-zero/non-air row → height, then smooth."""
    nz, nx = topo.shape
    ele = np.empty(nx, dtype=float)
    for c in range(nx):
        r = 0
        while r < nz and (topo[r, c] == 0 or np.isnan(topo[r, c])):
            r += 1
        ele[c] = sur0 - y_max / nz * r
    return _smooth_elev(ele)


def elevation_from_topo(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return elevation_from_topo_array(load_topo_array(path))


def elevation_from_composition(grid_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build topo from composition with the same Topo.py rule (full model height)."""
    return elevation_from_topo_array(grid_full, y_max=Z_SIZE)


def get_elevation(folder: Path, step: int, grid_full: np.ndarray | None = None):
    """Elevation via Topo.py rule on the composition grid (air→rock interface).

    Uses gouc composition rather than topo_*.csv so all steps (including the
    50 Myr frames without topo files) share one consistent absolute scale.
    """
    if grid_full is None:
        _, grid_full = decode_composition(folder / f"gouc{step}.txt")
    return elevation_from_composition(grid_full)


def zoom_composition(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz, nx = grid.shape
    x_beg_i = 0 if X_BEG == 0 else int(round((X_BEG / X_SIZE) * nx))
    x_end_i = int(round((X_END / X_SIZE) * nx))
    z_beg_i = 0 if Z_BEG == 0 else int(round((Z_BEG / Z_SIZE) * nz))
    z_end_i = int(round((Z_END / Z_SIZE) * nz))
    cropped = grid[z_beg_i:z_end_i, x_beg_i:x_end_i]
    nz_c, nx_c = cropped.shape
    x = np.linspace(X_BEG, X_END, nx_c)
    z = np.linspace(Z_BEG, Z_END, nz_c) - Z_OFFSET
    return cropped, x, z


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def apply_nature_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TICK_LABELSIZE,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": TICK_LABELSIZE,
            "ytick.labelsize": TICK_LABELSIZE,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": DPI,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
        }
    )


def format_display_time(t_disp: float) -> str:
    t = max(0.0, t_disp)
    if t < 0.05:
        return "0"
    if abs(t - round(t)) < 0.05:
        return f"{int(round(t))}"
    return f"{t:.1f}"


def panel_label(ax, label: str):
    ax.text(
        0.03,
        0.94,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="k",
        bbox=dict(
            boxstyle="square,pad=0.12",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
        zorder=6,
    )


def time_label(ax, t_disp: float):
    ax.text(
        0.97,
        0.94,
        f"{format_display_time(t_disp)} Myr",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="k",
        bbox=dict(
            boxstyle="square,pad=0.12",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
        zorder=6,
    )


def style_comp_axis(ax, *, show_xlabel: bool, show_ylabel: bool, show_xticklabels: bool):
    ax.set_xlim(X_BEG, X_END)
    ax.set_ylim(Z_END - Z_OFFSET, Z_BEG - Z_OFFSET)
    ax.set_aspect("auto")
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_yticks([0, 200, 400])
    ax.tick_params(which="both", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    if show_xlabel:
        ax.set_xlabel("Width (km)", labelpad=3)
    else:
        ax.set_xlabel("")
    if not show_xticklabels:
        ax.set_xticklabels([])
    else:
        labs = ax.set_xticklabels(["0", "200", "400", "600", "800"])
        labs[0].set_ha("left")
        labs[-1].set_ha("right")

    if show_ylabel:
        ax.set_ylabel("Depth (km)", labelpad=3)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])


def style_elev_axis(ax, *, show_ylabel: bool):
    ax.set_xlim(X_BEG, X_END)
    ax.set_ylim(*ELEV_YLIM)
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xticklabels([])
    ax.set_yticks(ELEV_YTICKS)
    ax.tick_params(
        which="both",
        top=False,
        right=False,
        bottom=False,
        length=2.2,
        labelsize=ELEV_TICKSIZE,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(0.6)
    if show_ylabel:
        ax.set_ylabel(ELEV_YLABEL, labelpad=2, fontsize=ELEV_LABELSIZE)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])


def delta_elev(
    x: np.ndarray,
    y: np.ndarray,
    x_base: np.ndarray,
    y_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Elevation anomaly relative to the panel a/d baseline profile."""
    return x, y - np.interp(x, x_base, y_base)


def add_elev_arrow(
    ax,
    x_km: float,
    y_km: float,
    *,
    uplift: bool = True,
    shaft_frac: float = 0.42,
):
    """Red arrow marking uplift (↑) or subsidence (↓) for this step.

    shaft_frac: arrow length as a fraction of the elevation-axes height.
    Tip is kept clear of the elevation line (gap below/above the curve).
    """
    # Mixed coords keep the shaft vertical in display space (data-x / axes-y).
    y0, y1 = ELEV_YLIM
    y_frac = (y_km - y0) / (y1 - y0)
    # Shorten shaft by 1/4; leave a gap so the tip does not touch the curve
    shaft = shaft_frac * 0.75
    tip_gap = 0.08
    if uplift:
        y_tip = min(0.97, y_frac - tip_gap)
        y_base = max(0.04, y_tip - shaft)
        xy = (x_km, y_tip)
        xytext = (x_km, y_base)
    else:
        y_tip = max(0.03, y_frac + tip_gap)
        y_base = min(0.96, y_tip + shaft)
        xy = (x_km, y_tip)
        xytext = (x_km, y_base)
    ax.annotate(
        "",
        xy=xy,
        xytext=xytext,
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(
            arrowstyle="-|>",
            color="red",
            lw=0.85,
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=5,
        clip_on=False,
    )


def uplift_arrow_site(
    x: np.ndarray,
    y: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    x_hint: float | None = None,
) -> tuple[float, float, bool]:
    """Return (x, y_current, is_uplift) at the strongest uplift vs reference.

    Prefers max(Δelev). If the profile only subsided, uses min(Δelev) and
    points the arrow downward.
    """
    y_ref_i = np.interp(x, x_ref, y_ref)
    dy = y - y_ref_i
    if x_hint is not None:
        m = (x >= x_hint - 100.0) & (x <= x_hint + 100.0)
        if not np.any(m):
            m = np.ones_like(dy, dtype=bool)
    else:
        m = np.ones_like(dy, dtype=bool)
    dy_m = dy[m]
    idxs = np.where(m)[0]
    if float(np.max(dy_m)) > 0.05:
        full_idx = int(idxs[int(np.argmax(dy_m))])
        uplift = True
    else:
        full_idx = int(idxs[int(np.argmin(dy_m))])
        uplift = False
    return float(x[full_idx]), float(y[full_idx]), uplift


def _draw_sediment_swatch(leg_ax, x0, y0, sw_w, sw_h, colors: np.ndarray):
    """Sediments swatch: thin horizontal stripes alternating rock types 3 and 2."""
    c2 = colors[2]
    c3 = colors[3]
    n_stripes = 10  # thinner stripes
    dh = sw_h / n_stripes
    for i in range(n_stripes):
        # Swap vs previous: start with type 3, then type 2
        face = c3 if (i % 2 == 0) else c2
        leg_ax.add_patch(
            Rectangle(
                (x0, y0 + i * dh),
                sw_w,
                dh,
                facecolor=face,
                edgecolor="none",
                linewidth=0,
                transform=leg_ax.transAxes,
                clip_on=False,
            )
        )
    # Outer border
    leg_ax.add_patch(
        Rectangle(
            (x0, y0),
            sw_w,
            sw_h,
            facecolor="none",
            edgecolor="0.2",
            linewidth=0.55,
            transform=leg_ax.transAxes,
            clip_on=False,
        )
    )


def add_material_legend(leg_ax, colors: np.ndarray):
    """One-row legend; rightmost label's right edge flush to panel right (x=1)."""
    n = len(MATERIAL_LEGEND)
    label_gap = LEGEND_LABEL_GAP
    y_top = 0.90

    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    fig = leg_ax.figure
    fig.canvas.draw()
    bbox = leg_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_w_pt = bbox.width * 72.0
    ax_h_pt = bbox.height * 72.0
    sw_w = LEGEND_SWATCH_W_PT / max(ax_w_pt, 1.0)  # keep original width
    # Thickness only: physical height = width * 6/9
    sw_h = (LEGEND_SWATCH_W_PT * 6.0 / 9.0) / max(ax_h_pt, 1.0)
    # Move labels down by swatch thickening (fixed width, height -> W*6/9)
    # delta_h = h_9x6 - h_prev; h_prev taken as flatter W/2 (2:1) with same width
    label_down = sw_h - (LEGEND_SWATCH_W_PT * 0.5) / max(ax_h_pt, 1.0)

    # Measure rightmost label width in axes fraction
    last_name = MATERIAL_LEGEND[-1][1]
    probe = leg_ax.text(
        0.0,
        0.0,
        last_name,
        transform=leg_ax.transAxes,
        fontsize=LEGEND_FONTSIZE,
        ha="left",
        va="top",
        linespacing=1.05,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pb = probe.get_window_extent(renderer=renderer)
    inv = leg_ax.transAxes.inverted()
    x0p, _ = inv.transform((pb.x0, pb.y0))
    x1p, _ = inv.transform((pb.x1, pb.y1))
    last_w = float(abs(x1p - x0p))
    probe.remove()

    cx0 = 0.5 * sw_w
    # Last swatch sits above the right-aligned label (label right edge at x=1)
    cx_last = 1.0 - 0.5 * last_w
    centres = np.array([cx0]) if n == 1 else np.linspace(cx0, cx_last, n)
    y0 = y_top - sw_h

    for k, (idx, name) in enumerate(MATERIAL_LEGEND):
        cx = float(centres[k])
        x0 = cx - 0.5 * sw_w
        is_last = k == n - 1

        if name == "Sediments" or idx == 1:
            _draw_sediment_swatch(leg_ax, x0, y0, sw_w, sw_h, colors)
        else:
            face = colors[idx]
            edge = "0.35" if float(np.mean(face)) > 0.92 else "0.2"
            leg_ax.add_patch(
                Rectangle(
                    (x0, y0),
                    sw_w,
                    sw_h,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.55,
                    transform=leg_ax.transAxes,
                    clip_on=False,
                )
            )

        if is_last:
            leg_ax.text(
                1.0,
                y0 - label_gap - label_down,
                name,
                transform=leg_ax.transAxes,
                ha="right",
                va="top",
                fontsize=LEGEND_FONTSIZE,
                color="k",
                linespacing=1.05,
                clip_on=False,
            )
        else:
            leg_ax.text(
                cx,
                y0 - label_gap - label_down,
                name,
                transform=leg_ax.transAxes,
                ha="center",
                va="top",
                fontsize=LEGEND_FONTSIZE,
                color="k",
                linespacing=1.05,
                clip_on=False,
            )


def _step_path(folder: Path, step: int) -> tuple[int, float, Path]:
    path = folder / f"gouc{step}.txt"
    if not path.exists() or path.stat().st_size < 100:
        raise FileNotFoundError(f"Missing composition file: {path}")
    return step, read_time_myr(path), path


def nearest_gouc_step(folder: Path, target_myr: float) -> tuple[int, float]:
    """Return (step, model_time_Myr) of the gouc file closest to target_myr."""
    best: tuple[float, int, float] | None = None
    for path in folder.glob("gouc*.txt"):
        tail = path.stem[4:]
        if not tail.isdigit() or path.stat().st_size < 100:
            continue
        step = int(tail)
        t = read_time_myr(path)
        err = abs(t - target_myr)
        if best is None or err < best[0]:
            best = (err, step, t)
    if best is None:
        raise FileNotFoundError(f"No gouc files in {folder}")
    return best[1], best[2]


def resolve_folder(name: str) -> Path:
    """Use the named model folder; if it has no gouc files, try the isostasy twin."""
    primary = BASE / name
    if any(primary.glob("gouc*.txt")):
        return primary
    if name.endswith("_NoIsostasy"):
        alt = BASE / name[: -len("_NoIsostasy")]
        if any(alt.glob("gouc*.txt")):
            print(f"Note: {name} has no gouc files; using {alt.name}")
            return alt
    return primary


def peak_near(x: np.ndarray, y: np.ndarray, x0: float, half_win: float = 80.0) -> float:
    m = (x >= x0 - half_win) & (x <= x0 + half_win)
    if not np.any(m):
        return float(np.max(y))
    return float(np.max(y[m]))


SEDIMENT_IDS = (1, 2, 3)


def sediment_basin_sites(
    grid: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    *,
    xs: tuple[float, ...] | None = None,
    z_max: float = 25.0,
    min_thick: float = 4.0,
    min_sep_km: float = 80.0,
    max_n: int = 2,
) -> list[tuple[float, float]]:
    """Near-surface sediment thickenings: (x_km, z_base_km) for basin pointers.

    Sediment fill is only a few km thick, so these sites are the tiny surface
    basins. z_base is the deepest sediment in that column (arrow/triangle tip
    sits just below it).
    """
    sed = np.isin(grid, SEDIMENT_IDS) & np.isfinite(grid)
    near = sed & (z[:, None] < z_max)
    if not np.any(near):
        return []

    def z_base_at(x_km: float) -> float:
        i = int(np.argmin(np.abs(x - x_km)))
        col = near[:, i]
        if np.any(col):
            return float(z[col].max())
        return 0.0

    if xs is not None:
        return [(float(x_km), z_base_at(float(x_km))) for x_km in xs]

    dz = float(abs(z[1] - z[0])) if z.size > 1 else 1.0
    dx = float(abs(x[1] - x[0])) if x.size > 1 else 1.0
    thick = near.sum(axis=0).astype(float) * dz
    win = max(3, int(round(8.0 / max(dx, 1e-6))))
    if win % 2 == 0:
        win += 1
    ts = np.convolve(thick, np.ones(win) / win, mode="same")
    cand: list[tuple[float, int]] = []
    for i in range(2, ts.size - 2):
        if ts[i] >= ts[i - 1] and ts[i] >= ts[i + 1] and ts[i] >= min_thick:
            cand.append((float(ts[i]), i))
    cand.sort(reverse=True)
    picked: list[int] = []
    for _, i in cand:
        if all(abs(x[i] - x[j]) >= min_sep_km for j in picked):
            picked.append(i)
        if len(picked) >= max_n:
            break
    picked.sort()
    sites: list[tuple[float, float]] = []
    for i in picked:
        col = near[:, i]
        z_base = float(z[col].max()) if np.any(col) else 0.0
        sites.append((float(x[i]), z_base))
    return sites


def add_basin_callouts(ax, sites: list[tuple[float, float]]):
    """One centred 'Basins' label with two small arrows to the basin floors.

    Label is 20 km shallower than the previous z=72 km placement (now 52 km)
    and at panel mid-width. Each arrow tips 5 km below that basin's sediment
    base. Shafts start on the top edge of the label box, not inside it.
    """
    if not sites:
        return
    x_lab = 0.5 * (X_BEG + X_END)
    z_lab = 52.0  # was 72 km; shifted 20 km toward the surface
    fontsize = 7.5
    bbox_pad = 0.08
    txt = ax.text(
        x_lab,
        z_lab,
        "Basins",
        ha="center",
        va="top",
        fontsize=fontsize,
        fontstyle="italic",
        color="k",
        zorder=8,
        clip_on=True,
        bbox=dict(
            boxstyle=f"square,pad={bbox_pad}",
            facecolor="white",
            edgecolor="none",
            alpha=0.55,
        ),
    )
    pos = ax.get_position()
    fig_w_in, fig_h_in = ax.figure.get_size_inches()
    ax_h_pt = pos.height * fig_h_in * 72.0
    z0, z1 = ax.get_ylim()
    dz_pt = abs(z1 - z0) / max(ax_h_pt, 1.0)
    # Top of the padded box (shallower than the va='top' text anchor).
    z_top = z_lab - bbox_pad * fontsize * dz_pt
    patch = txt.get_bbox_patch()
    n = len(sites)
    for k, (x_km, z_base) in enumerate(sites):
        z_tip = float(z_base) + 5.0
        if n == 1:
            x_start = x_lab
        else:
            x_start = x_lab - 12.0 if k < n / 2 else x_lab + 12.0
        ax.annotate(
            "",
            xy=(x_km, z_tip),
            xytext=(x_start, z_top),
            arrowprops=dict(
                arrowstyle="-|>",
                color="k",
                lw=0.65,
                mutation_scale=7.0,
                shrinkA=0,
                shrinkB=0,
                patchA=patch,
            ),
            zorder=7,
            annotation_clip=True,
        )


# %% ========================================================================
# Figure
# ============================================================================
def main():
    apply_nature_style()
    cmap_c, norm_c, colors = material_cmap()

    n_rows = len(ROWS)
    n_cols = N_COLS
    left, right = 0.10, 0.915

    # Geometry in inches: each "panel stack" = elev + gap + composition
    usable_w = FIG_W_IN * (right - left)
    panel_w_in = usable_w / (n_cols + (n_cols - 1) * WSPACE)
    data_asp = (Z_END - Z_BEG) / (X_END - X_BEG)
    comp_h_in = panel_w_in * data_asp
    elev_h_in = ELEV_FRAC * comp_h_in
    elev_gap_in = ELEV_GAP_FRAC * comp_h_in
    stack_h_in = elev_h_in + elev_gap_in + comp_h_in
    row_gap_in = HSPACE * stack_h_in
    panels_h_in = n_rows * stack_h_in + (n_rows - 1) * row_gap_in

    top_pad_in = 0.06
    width_label_gap_in = 0.50
    leg_h_in = 0.48
    bottom_pad_in = 0.04
    fig_h = top_pad_in + panels_h_in + width_label_gap_in + leg_h_in + bottom_pad_in
    fig_h = min(fig_h, 247.0 / 25.4)

    fig = plt.figure(figsize=(FIG_W_IN, fig_h))

    col_w = panel_w_in / FIG_W_IN
    col_gap = (WSPACE * panel_w_in) / FIG_W_IN
    x0s = [left + j * (col_w + col_gap) for j in range(n_cols)]

    y_top = 1.0 - top_pad_in / fig_h
    elev_h_frac = elev_h_in / fig_h
    elev_gap_frac = elev_gap_in / fig_h
    comp_h_frac = comp_h_in / fig_h
    stack_h_frac = stack_h_in / fig_h
    row_gap_frac = row_gap_in / fig_h

    elev_axes = np.empty((n_rows, n_cols), dtype=object)
    comp_axes = np.empty((n_rows, n_cols), dtype=object)

    for i in range(n_rows):
        stack_top = y_top - i * (stack_h_frac + row_gap_frac)
        elev_y0 = stack_top - elev_h_frac
        comp_y0 = elev_y0 - elev_gap_frac - comp_h_frac
        for j in range(n_cols):
            elev_axes[i, j] = fig.add_axes([x0s[j], elev_y0, col_w, elev_h_frac])
            comp_axes[i, j] = fig.add_axes([x0s[j], comp_y0, col_w, comp_h_frac])

    leg_y0 = bottom_pad_in / fig_h
    leg_h_frac = leg_h_in / fig_h
    leg_ax = fig.add_axes([x0s[0], leg_y0, x0s[-1] + col_w - x0s[0], leg_h_frac])

    letters = "abcdef"

    for i, row in enumerate(ROWS):
        folder = resolve_folder(row.folder)
        for j in range(n_cols):
            ax_e = elev_axes[i, j]
            ax_c = comp_axes[i, j]
            last = i == n_rows - 1
            step, t_model, path = _step_path(folder, row.steps[j])
            t_yr, grid_full = decode_composition(path)
            t_model = t_yr / 1.0e6
            t_disp = t_model - TIME0_MODEL_MYR
            if row.force_display[j] is not None:
                t_disp = row.force_display[j]

            grid, x, z = zoom_composition(grid_full)
            grid = remap_for_display(grid)

            ax_c.pcolormesh(
                x,
                z,
                grid,
                cmap=cmap_c,
                norm=norm_c,
                shading="nearest",
                rasterized=True,
            )
            style_comp_axis(
                ax_c,
                show_xlabel=(last and j == 1),
                show_ylabel=(j == 0),
                show_xticklabels=last,
            )
            panel_label(ax_c, letters[i * n_cols + j])
            time_label(ax_c, t_disp)

            # --- ΔElevation relative to panels a/d (col 0) ---
            # Optional per-column offset: elevation from a later gouc snapshot.
            x_base, y_base = get_elevation(folder, row.steps[0])
            elev_step = step
            elev_t = t_model
            dt_elev = row.elev_dt_myr[j]
            if dt_elev is not None:
                elev_step, elev_t = nearest_gouc_step(folder, t_model + dt_elev)
                xe_abs, ye_abs = get_elevation(folder, elev_step)
            else:
                xe_abs, ye_abs = get_elevation(folder, step, grid_full)
            xe, ye = delta_elev(xe_abs, ye_abs, x_base, y_base)
            style_elev_axis(ax_e, show_ylabel=(j == 0))
            # No gray zero-reference line on b/c/e/f (only a/d are the baseline)

            # Reference profile for the uplift arrow (also as ΔElev)
            x_ref = y_ref = None
            if j == 1:
                # vs a/d → ΔElev reference is the zero line
                x_ref, y_ref = xe, np.zeros_like(ye)
            elif j == 2 and row.first_drip_topo_step is not None:
                x1_abs, y1_abs = get_elevation(folder, row.first_drip_topo_step)
                x_ref, y_ref = delta_elev(x1_abs, y1_abs, x_base, y_base)

            ax_e.plot(xe, ye, color="k", lw=1.1, zorder=4)

            for col_a, x_hint in row.elev_arrows:
                if col_a != j:
                    continue
                if x_hint is not None:
                    # Fixed x (e.g. panel f at 500 km): tip at that column's ΔElev
                    xa = float(x_hint)
                    ya = float(np.interp(xa, xe, ye))
                    up = True
                    if x_ref is not None:
                        y_ref_i = float(np.interp(xa, x_ref, y_ref))
                        up = ya >= y_ref_i
                elif x_ref is not None:
                    xa, ya, up = uplift_arrow_site(xe, ye, x_ref, y_ref, None)
                else:
                    ia = int(np.argmax(ye))
                    xa, ya = float(xe[ia]), float(ye[ia])
                    up = True
                # Panel f: shorten shaft by 1/3 vs previous 0.55 → ≈0.367
                shaft = (0.55 * (2.0 / 3.0)) if (i == 1 and j == 2) else 0.36
                add_elev_arrow(ax_e, xa, ya, uplift=up, shaft_frac=shaft)

            if j == 0:
                for x_lab, z_lab, text in row.left_labels:
                    ax_c.text(
                        x_lab,
                        z_lab,
                        text,
                        ha="center",
                        va="top",
                        fontsize=8,
                        color="k",
                        fontstyle="italic",
                        clip_on=True,
                    )

            for col_d, x_d, z_d, text in row.drip_labels:
                if col_d == j:
                    ax_c.text(
                        x_d,
                        z_d,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="k",
                        fontstyle="italic",
                        clip_on=True,
                        bbox=dict(
                            boxstyle="square,pad=0.12",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.55,
                        ),
                    )

            if j == 2:
                basin_sites = sediment_basin_sites(grid, x, z, xs=row.basin_xs)
                add_basin_callouts(ax_c, basin_sites)
                basin_txt = " ".join(f"{bx:.0f}" for bx, _ in basin_sites) or "none"
            else:
                basin_txt = "-"

            elev_src = (
                f"elev_step={elev_step} model={elev_t:7.2f} "
                f"(dt={elev_t - t_model:+.2f} Myr)"
                if elev_step != step
                else f"elev_step={elev_step} (same)"
            )
            print(
                f"{row.folder:32s} col={j} step={step:4d} "
                f"model={t_model:7.2f} Myr  display={t_disp:7.2f} Myr  "
                f"elev=[{ye.min():.2f},{ye.max():.2f}]  {elev_src}  "
                f"basins=[{basin_txt}]  ({path.name})"
            )

    add_material_legend(leg_ax, colors)

    fig.canvas.draw()
    for j in range(n_cols):
        ax = comp_axes[-1, j]
        for lab in ax.get_xticklabels():
            t = lab.get_text().strip()
            if t == "0":
                lab.set_ha("left")
            elif t == "800":
                lab.set_ha("right")

    for out in (OUT_PNG, OUT_PDF, OUT_SVG, OUT_TIFF):
        fig.savefig(out, dpi=DPI, bbox_inches=None)
        print(f"Saved: {out}")
    # EPS with editable TrueType text (ps.fonttype=42); composition stays rasterized
    fig.savefig(OUT_EPS, dpi=DPI, bbox_inches=None, format="eps")
    print(f"Saved: {OUT_EPS}")
    w_mm = fig.get_size_inches()[0] * 25.4
    h_mm = fig.get_size_inches()[1] * 25.4
    print(
        f"Figure size: {w_mm:.1f} × {h_mm:.1f} mm "
        f"(Nature double-column width = {FIG_W_MM:.0f} mm)"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
