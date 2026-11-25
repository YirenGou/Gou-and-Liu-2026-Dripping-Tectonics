"""
Fig. S5 — Grain size reduction in mantle lithosphere (s220, no isostasy).

3 rows (100 / 10 / 1 μm) × 3 columns:
  col 0  Time = 0 (model 50 Myr)
  col 1  mantle-lithosphere drip approaching the plot-window bottom (~350 km)
  col 2  1–2 Myr after the drip, when LAB thinning has slowed
         (+ Initial LAB / New LAB / Δh annotations)

    Display time = model_time − TIME0_MODEL_MYR.
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
OUT_PNG = BASE / "FigS5_grain_size_mantle_lithosphere.png"

# Display time = model_time_Myr - TIME0_MODEL_MYR  (model 50 Myr → "0 Myr")
TIME0_MODEL_MYR = 50.0

# Zoom window (km) — deeper than Fig2 so drip exit is visible (~0–350 km)
X_SIZE, Z_SIZE = 800.0, 600.0
X_BEG, X_END = 200.0, 600.0
Z_BEG, Z_END = 0.0, 360.0
Z_OFFSET = 10.0  # MATLAB: z_vec = z_vec - 10

N_COLS = 3
ECLOGITE_ID = 12
ASTHENOSPHERE_ID = 10


@dataclass(frozen=True)
class PanelRow:
    """One figure row = one grain-size model; 3 gouc steps (L→R)."""

    folder: str
    steps: tuple[int, int, int]
    grain_label: str
    # Depth (km) of the TOP of the grain-size text (leftmost panel)
    grain_depth_km: float = 110.0
    force_display_left: float | None = 0.0


# Steps: t≈50 Myr | ML tip approaching window bottom | drip + 1–2 Myr (LAB rate slowed)
ROWS = [
    PanelRow(
        folder="FigS3a_SameAsFig1c_100um_NoIsostasy_s220",
        steps=(84, 199, 276),
        grain_label=r"Grain size = 100 $\mathrm{\mu}$m",
    ),
    PanelRow(
        folder="FigS3b_SameAsFig1c_10um_NoIsostasy_s220",
        steps=(84, 197, 270),
        grain_label=r"Grain size = 10 $\mathrm{\mu}$m",
    ),
    PanelRow(
        folder="FigS3c_SameAsFig1c_1um_NoIsostasy_s220",
        steps=(84, 192, 275),
        grain_label=r"Grain size = 1 $\mathrm{\mu}$m",
    ),
]

# Legend: materials present in these weak-ML grain-size models
MATERIAL_LEGEND = [
    (0, "Air"),
    (5, "Upper Crust"),
    (6, "Lower Crust"),
    (12, "Eclogite"),
    (10, "Asthenosphere"),
    (8, "Weak Mantle\nLithosphere"),
]

# %% ========================================================================
# Style / figure geometry (usually leave alone)
# ============================================================================
FIG_W_MM = 183.0
FIG_W_IN = FIG_W_MM / 25.4
DPI = 600
HSPACE = 0.02
WSPACE = 0.08

# Legend swatch: keep width; thickness from width for 9:6
LEGEND_SWATCH_W_PT = 24
LEGEND_SWATCH_H_PT = 16  # W * 6/9
LEGEND_LABEL_GAP = 0.04
LEGEND_ROW_GAP = 0.015
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = TICK_LABELSIZE


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------
def material_cmap():
    colors = np.zeros((39, 3), dtype=float)
    colors[:] = 1.0
    colors[0] = [1.0, 1.0, 1.0]  # Air
    colors[5] = [236 / 255, 235 / 255, 194 / 255]  # Upper Crust
    colors[6] = [202 / 255, 178 / 255, 114 / 255]  # Lower Crust
    colors[7] = [104 / 255, 148 / 255, 92 / 255]  # Normal ML
    colors[8] = [168 / 255, 191 / 255, 143 / 255]  # Weak ML
    colors[9] = [99 / 255, 99 / 255, 102 / 255]  # Carbonated ML
    colors[10] = [245 / 255, 210 / 255, 188 / 255]  # Asthenosphere
    colors[12] = [152 / 255, 79 / 255, 49 / 255]  # Eclogite
    colors[13] = [69 / 255, 70 / 255, 94 / 255]
    colors[14] = [69 / 255, 70 / 255, 94 / 255]
    colors[15] = colors[8]

    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, colors


def remap_for_display(grid: np.ndarray) -> np.ndarray:
    """Keep rock 8 as Weak ML (all Fig. S5 grain-size models)."""
    return np.array(grid, copy=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_numbers(path: Path) -> np.ndarray:
    return np.fromfile(path, sep=" ", dtype=np.float64)


def decode_composition(path: Path) -> tuple[float, np.ndarray]:
    """Decode RLE composition file → (time_yr, grid[nz, nx]), MATLAB order."""
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


def zoom_composition(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz, nx = grid.shape
    x_beg_i = 0 if X_BEG == 0 else int(round((X_BEG / X_SIZE) * nx))
    x_end_i = int(round((X_END / X_SIZE) * nx))
    z_beg_i = 0 if Z_BEG == 0 else int(round((Z_BEG / Z_SIZE) * nz))
    z_end_i = int(round((Z_END / Z_SIZE) * nz))

    if Z_BEG == 0 and X_BEG != 0:
        z_beg_i = 0
    cropped = grid[z_beg_i:z_end_i, x_beg_i:x_end_i]

    nz_c, nx_c = cropped.shape
    x = np.linspace(X_BEG, X_END, nx_c)
    z = np.linspace(Z_BEG, Z_END, nz_c) - Z_OFFSET
    return cropped, x, z


def lab_depth_km(grid: np.ndarray, z: np.ndarray) -> float | None:
    """Shallowest asthenosphere (rock 10) depth = LAB."""
    ast = grid == ASTHENOSPHERE_ID
    if not ast.any():
        return None
    zi, _ = np.where(ast)
    return float(z[zi].min())


def new_lab_peak(grid: np.ndarray, x: np.ndarray, z: np.ndarray) -> tuple[float, float] | None:
    """(x_km, depth_km) of the shallowest asthenosphere cell (New LAB peak)."""
    ast = grid == ASTHENOSPHERE_ID
    if not ast.any():
        return None
    zi, xi = np.where(ast)
    k = int(np.argmin(z[zi]))
    return float(x[xi[k]]), float(z[zi[k]])


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def apply_nature_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TICK_LABELSIZE,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
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
            "savefig.dpi": DPI,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
        }
    )


def wspace_factor() -> float:
    return WSPACE


def figure_height_in(
    n_rows: int, n_cols: int, *, hspace: float, bottom: float, top: float
) -> float:
    left, right = 0.08, 0.98
    usable_w = FIG_W_IN * (right - left)
    panel_w = usable_w / (n_cols + (n_cols - 1) * wspace_factor())
    data_aspect = (Z_END - Z_BEG) / (X_END - X_BEG)
    panel_h = panel_w * data_aspect
    axes_block = n_rows * panel_h * (1.0 + hspace) - hspace * panel_h
    return axes_block / (top - bottom)


def style_axis(
    ax,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    show_xticklabels: bool,
):
    ax.set_xlim(X_BEG, X_END)
    ax.set_ylim(Z_END - Z_OFFSET, Z_BEG - Z_OFFSET)
    ax.set_aspect("auto")
    ax.set_xticks([200, 400, 600])
    ax.set_yticks([0, 100, 200, 300])
    ax.tick_params(which="both", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    if show_xlabel:
        ax.set_xlabel("Width (km)", labelpad=6)
    else:
        ax.set_xlabel("")
    if not show_xticklabels:
        ax.set_xticklabels([])
    else:
        tick_labels = ax.set_xticklabels(["200", "400", "600"])
        tick_labels[0].set_ha("left")
        tick_labels[-1].set_ha("right")

    if show_ylabel:
        ax.set_ylabel("Depth (km)", labelpad=4)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])


def pack_panel_rows(axes: np.ndarray, hspace: float, *, min_bottom: float = 0.14):
    """Stack panels so physical H/W equals the plotted window in km.

    ``panel_h_frac = panel_w_frac * data_aspect`` is only correct when the
    figure is square. Convert through inches so n_rows / fig_h cannot
    squash or stretch the model window. Grow the figure if needed; never
    scale panel height down.
    """
    ax0 = next(ax for ax in axes.flat if ax is not None)
    fig = ax0.figure
    fig.canvas.draw()

    n_rows = axes.shape[0]
    z_span = (Z_END - Z_OFFSET) - (Z_BEG - Z_OFFSET)
    x_span = X_END - X_BEG
    data_aspect = z_span / x_span

    fig_w, fig_h = fig.get_size_inches()
    pos00 = ax0.get_position()
    panel_w = pos00.width
    panel_h = panel_w * data_aspect * (fig_w / fig_h)
    gap = hspace * panel_h
    y_top = pos00.y1

    total_h = n_rows * panel_h + (n_rows - 1) * gap
    avail = y_top - min_bottom
    if total_h > avail > 0:
        need = total_h - avail
        new_h = fig_h * (1.0 + need)
        scale_up = fig_h / new_h
        for ax in axes.flat:
            if ax is None:
                continue
            p = ax.get_position()
            y1_new = 1.0 - (1.0 - p.y1) * scale_up
            y0_new = 1.0 - (1.0 - p.y0) * scale_up
            ax.set_position([p.x0, y0_new, p.width, y1_new - y0_new])
        fig.set_size_inches(fig_w, new_h, forward=True)
        fig.canvas.draw()
        fig_h = new_h
        pos00 = ax0.get_position()
        panel_w = pos00.width
        panel_h = panel_w * data_aspect * (fig_w / fig_h)
        gap = hspace * panel_h
        y_top = pos00.y1

    for i in range(n_rows):
        y1 = y_top - i * (panel_h + gap)
        y0 = y1 - panel_h
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            if ax is None:
                continue
            p = ax.get_position()
            ax.set_position([p.x0, y0, p.width, panel_h])


def place_legend_and_trim(
    fig,
    axes: np.ndarray,
    leg_ax,
    colors: np.ndarray,
    *,
    legend_gap_in: float = 0.14 * 1.7,  # gap below "Width (km)" → legend
    legend_h_in: float = 0.48,
    bottom_pad_in: float = 0.04,
):
    """Put legend below 'Width (km)' with a clear gap; crop bottom dead space."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    xlabel_y0 = 1.0
    for j in range(axes.shape[1]):
        lab = axes[-1, j].xaxis.get_label()
        if not lab.get_text():
            continue
        bb = lab.get_window_extent(renderer=renderer)
        _, y0 = inv.transform((bb.x0, bb.y0))
        xlabel_y0 = min(xlabel_y0, float(y0))

    pos_l = axes[-1, 0].get_position()
    pos_r = axes[-1, -1].get_position()
    fig_h_in = float(fig.get_size_inches()[1])
    leg_gap = legend_gap_in / fig_h_in
    leg_h = legend_h_in / fig_h_in
    leg_y0 = xlabel_y0 - leg_gap - leg_h
    if leg_y0 < 0.01:
        need = (0.01 + leg_gap + leg_h) - xlabel_y0
        new_h = fig_h_in * (1.0 + need)
        scale_up = fig_h_in / new_h
        for ax in list(axes.flat) + [leg_ax]:
            if ax is None:
                continue
            p = ax.get_position()
            y1_new = 1.0 - (1.0 - p.y1) * scale_up
            y0_new = 1.0 - (1.0 - p.y0) * scale_up
            ax.set_position([p.x0, y0_new, p.width, y1_new - y0_new])
        fig.set_size_inches(FIG_W_IN, new_h, forward=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        xlabel_y0 = 1.0
        for j in range(axes.shape[1]):
            lab = axes[-1, j].xaxis.get_label()
            if not lab.get_text():
                continue
            bb = lab.get_window_extent(renderer=renderer)
            _, y0 = inv.transform((bb.x0, bb.y0))
            xlabel_y0 = min(xlabel_y0, float(y0))
        pos_l = axes[-1, 0].get_position()
        pos_r = axes[-1, -1].get_position()
        fig_h_in = float(fig.get_size_inches()[1])
        leg_gap = legend_gap_in / fig_h_in
        leg_h = legend_h_in / fig_h_in
        leg_y0 = xlabel_y0 - leg_gap - leg_h

    # Legend top sits legend_gap_in below the x-label; height is generous for now
    leg_ax.set_position([pos_l.x0, leg_y0, pos_r.x1 - pos_l.x0, leg_h])
    add_material_legend(leg_ax, colors)

    # Measure true legend content bottom (swatches + labels), then crop tightly
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    content_y0 = leg_ax.get_position().y1
    for artist in list(leg_ax.patches) + list(leg_ax.texts):
        bb = artist.get_window_extent(renderer=renderer)
        _, y0 = inv.transform((bb.x0, bb.y0))
        content_y0 = min(content_y0, float(y0))

    fig_h_in = float(fig.get_size_inches()[1])
    bottom_pad = bottom_pad_in / fig_h_in
    crop = content_y0 - bottom_pad
    if crop > 0.008:
        new_h = fig_h_in * (1.0 - crop)
        for ax in list(axes.flat) + [leg_ax]:
            if ax is None:
                continue
            p = ax.get_position()
            y0n = (p.y0 - crop) / (1.0 - crop)
            y1n = (p.y1 - crop) / (1.0 - crop)
            ax.set_position([p.x0, y0n, p.width, y1n - y0n])
        fig.set_size_inches(FIG_W_IN, new_h, forward=True)

def format_display_time(t_disp: float) -> str:
    t = max(0.0, t_disp)
    if t < 0.05:
        return "0"
    if abs(t - round(t)) < 0.05:
        return f"{int(round(t))}"
    return f"{t:.1f}"


def panel_label(ax, label: str):
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="k",
        bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
    )


def time_label(ax, t_disp: float):
    ax.text(
        0.98,
        0.96,
        f"{format_display_time(t_disp)} Myr",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="k",
        bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
    )


def add_grain_annotation(ax, text: str, depth_km: float):
    x_mid = 0.5 * (X_BEG + X_END)
    ax.text(
        x_mid,
        depth_km,
        text,
        ha="center",
        va="top",
        fontsize=11,
        color="k",
        fontstyle="italic",
        clip_on=True,
    )


def add_lab_annotations(ax, initial_lab: float, new_lab: float, peak_x: float):
    """Initial LAB (grey dashed), New LAB (black dashed), Δh double arrow.

    Arrow is shortened 10% toward its centre; Δh sits at that centre height.
    New LAB / Initial LAB share the same line→label gap.
    """
    dh = initial_lab - new_lab
    dh_label = int(round(dh))
    dh_text = rf"$\Delta h$ = {dh_label} km"
    # Same line→label gap for Initial LAB (below) and New LAB (above)
    lab_line_gap = 6.0  # km

    # Full-width Initial LAB; label parked at far left
    ax.axhline(
        initial_lab,
        color="0.45",
        ls="--",
        lw=1.0,
        zorder=4,
        clip_on=True,
    )
    # Label below the grey line (outside the Δh span) so it cannot meet Δh
    ax.text(
        X_BEG + 8,
        initial_lab + lab_line_gap,
        "Initial LAB",
        ha="left",
        va="top",
        fontsize=8,
        color="0.25",
        clip_on=True,
        zorder=5,
    )

    # Short New LAB mark centred on the upwelling peak
    half = 75.0
    x0 = max(X_BEG + 5, peak_x - half)
    x1 = min(X_END - 5, peak_x + half)
    ax.plot(
        [x0, x1],
        [new_lab, new_lab],
        color="k",
        ls="--",
        lw=1.1,
        zorder=4,
        clip_on=True,
    )

    # Arrow / Δh side first so New LAB can sit on the opposite end of the dash
    init_lab_clear = X_BEG + 115.0  # right edge of "Initial LAB" text band
    panel_mid = 0.5 * (X_BEG + X_END)
    label_gap = 18.0
    if peak_x >= panel_mid:
        # Peak on the right → arrow left of peak; Δh further left (toward centre)
        arrow_x = peak_x - 35.0
        label_side = "left"
        # New LAB above the right end of the black dash
        new_lab_x, new_lab_ha = x1 - 8.0, "right"
    else:
        # Peak on the left → arrow right of peak (and of "Initial LAB"); Δh further right
        arrow_x = max(peak_x + 40.0, init_lab_clear + 20.0)
        label_side = "right"
        # New LAB above the left end of the black dash
        new_lab_x, new_lab_ha = x0 + 8.0, "left"
    arrow_x = float(np.clip(arrow_x, x0 + 10.0, x1 - 10.0))

    # New LAB above the black dashed segment (same line→label gap as Initial LAB)
    ax.text(
        new_lab_x,
        new_lab - lab_line_gap,
        "New LAB",
        ha=new_lab_ha,
        va="bottom",
        fontsize=8,
        color="k",
        clip_on=True,
        zorder=5,
    )

    # Double arrow shortened 10% toward its centre (each tip moves in 5% of Δh).
    head_h = 4.5  # km along depth
    head_w = 5.5  # km half-width
    shrink = 0.05 * dh
    y_top = new_lab + shrink
    y_bot = initial_lab - shrink
    arrow_mid = 0.5 * (y_top + y_bot)
    ax.plot(
        [arrow_x, arrow_x],
        [y_top + head_h, y_bot - head_h],
        color="k",
        lw=1.0,
        solid_capstyle="butt",
        zorder=6,
        clip_on=True,
    )
    ax.fill(
        [arrow_x, arrow_x - head_w, arrow_x + head_w],
        [y_top, y_top + head_h, y_top + head_h],
        color="k",
        zorder=7,
        clip_on=True,
    )
    ax.fill(
        [arrow_x, arrow_x - head_w, arrow_x + head_w],
        [y_bot, y_bot - head_h, y_bot - head_h],
        color="k",
        zorder=7,
        clip_on=True,
    )

    # Δh at the same height as the arrow centre, toward panel centre
    if label_side == "right":
        lab_x, lab_ha = arrow_x + label_gap, "left"
    else:
        lab_x, lab_ha = arrow_x - label_gap, "right"
    ax.text(
        lab_x,
        arrow_mid,
        dh_text,
        ha=lab_ha,
        va="center",
        fontsize=9,
        color="k",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=6,
        clip_on=True,
    )


def _text_width_axes(leg_ax, text: str, fontsize: float) -> float:
    fig = leg_ax.figure
    probe = leg_ax.text(
        0.0,
        0.0,
        text,
        transform=leg_ax.transAxes,
        fontsize=fontsize,
        ha="left",
        va="top",
        linespacing=1.05,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = probe.get_window_extent(renderer=renderer)
    inv = leg_ax.transAxes.inverted()
    x0, _ = inv.transform((bb.x0, bb.y0))
    x1, _ = inv.transform((bb.x1, bb.y1))
    probe.remove()
    return float(abs(x1 - x0))


def add_material_legend(leg_ax, colors: np.ndarray):
    """Single-row legend spanning the panel width."""
    items = MATERIAL_LEGEND
    ncols = len(items)

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

    w_last = _text_width_axes(leg_ax, items[-1][1], LEGEND_FONTSIZE)
    cx0 = 0.5 * sw_w
    cx_right = 1.0 - 0.5 * w_last
    centres = np.linspace(cx0, cx_right, ncols)

    # Top-align so thickening extends downward; labels follow
    y0 = 0.90 - sw_h
    for col, (idx, name) in enumerate(items):
        cx = float(centres[col])
        x0 = 0.0 if col == 0 else cx - 0.5 * sw_w
        if col == 0:
            cx = cx0
        face = colors[idx]
        edge = "0.35" if float(np.mean(face)) > 0.92 else "0.2"
        leg_ax.add_patch(
            Rectangle(
                (x0, y0),
                sw_w,
                sw_h,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.7,
                transform=leg_ax.transAxes,
                clip_on=False,
            )
        )
        leg_ax.text(
            cx,
            y0 - LEGEND_LABEL_GAP - label_down,
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
    t = float(load_numbers(path)[0]) / 1.0e6
    return step, t, path


def resolve_panel(row: PanelRow, j: int) -> tuple[int, float, Path]:
    return _step_path(BASE / row.folder, row.steps[j])


# %% ========================================================================
# Figure
# ============================================================================
def main():
    apply_nature_style()
    cmap_c, norm_c, colors = material_cmap()

    n_rows = len(ROWS)
    n_cols = N_COLS
    left, right, top = 0.10, 0.915, 0.985
    # Room under panels for tick labels + "Width (km)" + gap + legend
    panel_bottom = 0.22
    fig_h = figure_height_in(
        n_rows, n_cols, hspace=HSPACE, bottom=panel_bottom, top=top
    )
    # Modest pad; place_legend_and_trim crops leftover bottom whitespace
    fig_h += 0.55
    fig_h = min(fig_h, 247.0 / 25.4)

    fig = plt.figure(figsize=(FIG_W_IN, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        wspace=WSPACE,
        hspace=HSPACE,
        left=left,
        right=right,
        top=top,
        bottom=panel_bottom,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])

    leg_ax = fig.add_axes([left, 0.02, right - left, 0.08])
    letters = "abcdefghijklmno"
    letter_i = 0

    # Initial LAB from each row's leftmost (Time=0) panel
    initial_labs: list[float] = []

    for i, row in enumerate(ROWS):
        last = i == n_rows - 1
        for j in range(n_cols):
            ax = axes[i, j]
            step, t_model, path = resolve_panel(row, j)
            t_yr, grid = decode_composition(path)
            t_model = t_yr / 1.0e6
            t_disp = t_model - TIME0_MODEL_MYR
            if j == 0 and row.force_display_left is not None:
                t_disp = row.force_display_left

            grid, x, z = zoom_composition(grid)
            grid = remap_for_display(grid)

            ax.pcolormesh(
                x,
                z,
                grid,
                cmap=cmap_c,
                norm=norm_c,
                shading="nearest",
                rasterized=True,
            )
            style_axis(
                ax,
                show_xlabel=last,
                show_ylabel=(j == 0),
                show_xticklabels=last,
            )
            panel_label(ax, letters[letter_i])
            letter_i += 1
            time_label(ax, t_disp)

            if j == 0:
                add_grain_annotation(ax, row.grain_label, row.grain_depth_km)
                lab0 = lab_depth_km(grid, z)
                initial_labs.append(lab0 if lab0 is not None else 170.0)

            if j == 2:
                init_lab = initial_labs[i]
                peak = new_lab_peak(grid, x, z)
                if peak is not None:
                    peak_x, new_lab = peak
                    add_lab_annotations(ax, init_lab, new_lab, peak_x)
                    print(
                        f"{row.folder:36s} Δh: Initial LAB={init_lab:.1f}  "
                        f"New LAB={new_lab:.1f}  Δh={init_lab - new_lab:.1f} km"
                    )

            print(
                f"{row.folder:36s} col{j} step={step:4d} "
                f"model={t_model:7.2f} Myr  display={t_disp:7.2f} Myr  ({path.name})"
            )

    pack_panel_rows(axes, HSPACE, min_bottom=0.20)
    place_legend_and_trim(fig, axes, leg_ax, colors)

    fig.canvas.draw()
    for j in range(n_cols):
        ax = axes[-1, j]
        for lab in ax.get_xticklabels():
            t = lab.get_text().strip()
            if t == "200":
                lab.set_ha("left")
            elif t == "600":
                lab.set_ha("right")

    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches=None)
    w_mm = fig.get_size_inches()[0] * 25.4
    h_mm = fig.get_size_inches()[1] * 25.4
    print(f"Saved: {OUT_PNG}")
    print(
        f"Figure size: {w_mm:.1f} × {h_mm:.1f} mm "
        f"(Nature double-column width = {FIG_W_MM:.0f} mm)"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
