"""
Fig. S2 — Effect of carbonatitic metasomatism on effective viscosity.

2 rows × 2 columns:
  left   composition field
  right  effective viscosity (log10 Pa·s)

Row a  without carbonatitic metasomatism  (Fig2c)
Row b  with carbonatitic metasomatism     (Fig2d)

Steps chosen where eclogite drip tip ≈ 100 km depth
(display time = model_time − TIME0_MODEL_MYR).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Rectangle

# %% ========================================================================
# USER CONTROLS  —  edit here for Spyder debugging (Ctrl+Enter cell)
# ============================================================================
BASE = Path(__file__).resolve().parent
FIG2 = BASE.parent / "Fig2"
OUT_PNG = BASE / "FigS2_carbonatite_viscosity.png"

TIME0_MODEL_MYR = 50.0

# Zoom window (km) — same as Fig2
X_SIZE, Z_SIZE = 800.0, 600.0
X_BEG, X_END = 200.0, 600.0
Z_BEG, Z_END = 0.0, 300.0
Z_OFFSET = 10.0

ECLOGITE_ID = 12
VISC_VMIN, VISC_VMAX = 16.0, 26.0  # log10(Pa·s), matches vis_node3.m


@dataclass(frozen=True)
class ModelRow:
    """One figure row = one model; composition + viscosity steps."""

    folder: Path
    # Composition step where Ec tip ≈ 90 km
    comp_step: int
    # Viscosity step (same as comp when gouv exists; else nearest)
    visc_step: int
    label: str
    annotation: str
    annotation_depth_km: float = 110.0


# Ec tip ≈ 100 km: Fig2c step 109 (99.7 km); nearest gouv = 120.
# Fig2d step 100 (101.1 km); gouv100 present.
ROWS = [
    ModelRow(
        folder=FIG2 / "Fig2c_50MyrInitiation_80km_NoIsostasy",
        comp_step=120,
        visc_step=120,
        # Two lines at panel a/c lower-left: line1 With/Without; line2 shared
        label="Without\ncarbonatitic metasomatism",
        annotation="Weakened",
        annotation_depth_km=110.0,
    ),
    ModelRow(
        folder=FIG2 / "Fig2d_50MyrIni_Carbonated_Isostasy_80km_NoIsostasy_Carb38km",
        comp_step=90,
        visc_step=90,
        label="With\ncarbonatitic metasomatism",
        annotation="Weakened",
        annotation_depth_km=100.0,
    ),
]

# Warm sequential (Nature-family); reversed so high η is dark
VISC_CMAP = "inferno_r"

MATERIAL_LEGEND_COLS = [
    [(0, "Air"), (10, "Asthenosphere")],
    [(5, "Upper Crust"), (6, "Lower Crust")],
    [(8, "Weak Mantle\nLithosphere"), (9, "Carbonatite-\nmetasomatized Mantle")],
    [(12, "Eclogite"),],
]

# %% ========================================================================
# Style / figure geometry
# ============================================================================
FIG_W_MM = 183.0
FIG_W_IN = FIG_W_MM / 25.4
DPI = 600
HSPACE = 0.06
WSPACE = 0.10

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
    colors[0] = [1.0, 1.0, 1.0]
    colors[5] = [236 / 255, 235 / 255, 194 / 255]
    colors[6] = [202 / 255, 178 / 255, 114 / 255]
    colors[7] = [104 / 255, 148 / 255, 92 / 255]
    colors[8] = [168 / 255, 191 / 255, 143 / 255]
    colors[9] = [176 / 255, 176 / 255, 179 / 255]  # Carbonated ML (same light gray as Fig. 2)
    colors[10] = [245 / 255, 210 / 255, 188 / 255]
    colors[12] = [152 / 255, 79 / 255, 49 / 255]
    colors[13] = [69 / 255, 70 / 255, 94 / 255]
    colors[14] = [69 / 255, 70 / 255, 94 / 255]
    colors[15] = colors[8]

    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, colors


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_numbers(path: Path) -> np.ndarray:
    return np.fromfile(path, sep=" ", dtype=np.float64)


def decode_composition(path: Path) -> tuple[float, np.ndarray]:
    """Decode RLE composition file → (time_yr, grid[nz, nx])."""
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


def read_node_field(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Node scalar field → (time_yr, x_km, z_km, field[nz, nx])."""
    d = load_numbers(path)
    time = float(d[0])
    nx, nz = int(d[1]), int(d[2])
    x = d[3 : 3 + nx] / 1000.0
    z = d[3 + nx : 3 + nx + nz] / 1000.0
    v = d[3 + nx + nz : 3 + nx + nz + nx * nz].reshape((nz, nx), order="F")
    return time, x, z, v


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


def zoom_node_field(
    x: np.ndarray, z: np.ndarray, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop node field to the same zoom window; depth = z − Z_OFFSET."""
    xmask = (x >= X_BEG) & (x <= X_END)
    zmask = (z >= Z_BEG) & (z <= Z_END)
    xc = x[xmask]
    zc = z[zmask] - Z_OFFSET
    fc = field[np.ix_(zmask, xmask)]
    return fc, xc, zc


def eclogite_tip_km(grid: np.ndarray, z: np.ndarray) -> float | None:
    """Deepest eclogite cell in the zoomed composition grid."""
    mask = grid == ECLOGITE_ID
    if not mask.any():
        return None
    zi, _ = np.where(mask)
    return float(z[zi].max())


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
    ax.set_yticks([0, 100, 200])
    ax.tick_params(which="both", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    if show_xlabel:
        ax.set_xlabel("Width (km)", labelpad=5)
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


def add_ml_annotation(ax, text: str, depth_km: float):
    x_mid = 0.5 * (X_BEG + X_END)
    ax.text(
        x_mid,
        depth_km,
        text,
        ha="center",
        va="top",
        fontsize=12,
        color="k",
        fontstyle="italic",
        clip_on=True,
    )


def condition_label(ax, text: str):
    """Lower-left condition note inside composition panels (a, c)."""
    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="k",
        linespacing=1.15,
        bbox=dict(
            boxstyle="square,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
    )


def place_legend_and_trim(
    fig,
    axes: np.ndarray,
    leg_ax,
    colors: np.ndarray,
    extra_axes=None,
    *,
    legend_gap_in: float = 0.22,
    legend_h_in: float = 0.55,
    bottom_pad_in: float = 0.03,
):
    """Place legend below 'Width (km)' with a clear gap; crop bottom dead space."""
    extra_axes = list(extra_axes or [])
    all_axes = [ax for ax in list(axes.flat) + [leg_ax] + extra_axes if ax is not None]

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
        for ax in all_axes:
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

    leg_ax.set_position([pos_l.x0, leg_y0, pos_r.x1 - pos_l.x0, leg_h])
    add_material_legend(leg_ax, colors)

    # Crop empty space below the true legend content
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
        for ax in all_axes:
            p = ax.get_position()
            y0n = (p.y0 - crop) / (1.0 - crop)
            y1n = (p.y1 - crop) / (1.0 - crop)
            ax.set_position([p.x0, y0n, p.width, y1n - y0n])
        fig.set_size_inches(FIG_W_IN, new_h, forward=True)


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
    cols = MATERIAL_LEGEND_COLS
    ncols = len(cols)

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

    last_name = cols[-1][0][1]
    w_last = _text_width_axes(leg_ax, last_name, LEGEND_FONTSIZE)
    cx0 = 0.5 * sw_w
    cx_right = 1.0 - 0.5 * w_last
    centres = np.linspace(cx0, cx_right, ncols)

    max_lines_top = max((1 + items[0][1].count("\n")) if items else 1 for items in cols)
    text_h_top = (max_lines_top * 1.25) * LEGEND_FONTSIZE / max(ax_h_pt, 1.0)

    y_top_swatch = 0.97 - sw_h
    y_bot_swatch = y_top_swatch - LEGEND_LABEL_GAP - label_down - text_h_top - LEGEND_ROW_GAP - sw_h
    row_y0 = [y_top_swatch, y_bot_swatch]

    for col, items in enumerate(cols):
        cx = float(centres[col])
        if col == 0:
            x0 = 0.0
            cx = cx0
        else:
            x0 = cx - 0.5 * sw_w

        for row, (idx, name) in enumerate(items):
            y0 = row_y0[row]
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


# %% ========================================================================
# Figure
# ============================================================================
def main():
    apply_nature_style()
    cmap_c, norm_c, colors = material_cmap()
    visc_norm = Normalize(vmin=VISC_VMIN, vmax=VISC_VMAX)

    n_rows = len(ROWS)
    n_cols = 2
    # No side row titles anymore — labels sit inside panels a/c
    left, right, top = 0.10, 0.88, 0.97
    panel_bottom = 0.22
    fig_h = figure_height_in(
        n_rows, n_cols, hspace=HSPACE, bottom=panel_bottom, top=top
    )
    # Pad for Width label + gap + two-row legend; trimmed after placement
    fig_h += 0.70
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

    leg_ax = fig.add_axes([left, 0.02, right - left, 0.12])
    letters = "abcdefgh"
    letter_i = 0
    last_visc_mappable = None

    for i, row in enumerate(ROWS):
        last = i == n_rows - 1

        # ---- composition (left) ----
        cpath = row.folder / f"gouc{row.comp_step}.txt"
        t_yr, grid = decode_composition(cpath)
        t_model = t_yr / 1.0e6
        t_disp = t_model - TIME0_MODEL_MYR
        grid_z, x, z = zoom_composition(grid)
        tip = eclogite_tip_km(grid_z, z)

        ax = axes[i, 0]
        ax.pcolormesh(
            x,
            z,
            grid_z,
            cmap=cmap_c,
            norm=norm_c,
            shading="nearest",
            rasterized=True,
        )
        style_axis(
            ax,
            show_xlabel=last,
            show_ylabel=True,
            show_xticklabels=last,
        )
        panel_label(ax, letters[letter_i])
        letter_i += 1
        time_label(ax, t_disp)
        #add_ml_annotation(ax, row.annotation, row.annotation_depth_km)
        condition_label(ax, row.label)

        print(
            f"{row.folder.name:40s} composition step={row.comp_step:4d}  "
            f"model={t_model:7.2f} Myr  display={t_disp:6.2f} Myr  "
            f"Ec tip={tip:.1f} km  ({cpath.name})"
        )

        # ---- viscosity (right) ----
        vpath = row.folder / f"gouv{row.visc_step}.txt"
        t_yr_v, xn, zn, visc = read_node_field(vpath)
        t_model_v = t_yr_v / 1.0e6
        t_disp_v = t_model_v - TIME0_MODEL_MYR
        visc_z, xv, zv = zoom_node_field(xn, zn, visc)
        logv = np.log10(np.clip(visc_z, 10**VISC_VMIN, 10**VISC_VMAX))

        ax = axes[i, 1]
        last_visc_mappable = ax.pcolormesh(
            xv,
            zv,
            logv,
            cmap=VISC_CMAP,
            norm=visc_norm,
            shading="nearest",
            rasterized=True,
        )
        style_axis(
            ax,
            show_xlabel=last,
            show_ylabel=False,
            show_xticklabels=last,
        )
        panel_label(ax, letters[letter_i])
        letter_i += 1
        time_label(ax, t_disp_v)

        print(
            f"{row.folder.name:40s} viscosity   step={row.visc_step:4d}  "
            f"model={t_model_v:7.2f} Myr  display={t_disp_v:6.2f} Myr  "
            f"({vpath.name})"
        )

    pack_panel_rows(axes, HSPACE, min_bottom=0.20)

    # Shared viscosity colorbar to the right of the panels
    pos_tr = axes[0, 1].get_position()
    pos_br = axes[-1, 1].get_position()
    cax = fig.add_axes([pos_tr.x1 + 0.015, pos_br.y0, 0.018, pos_tr.y1 - pos_br.y0])
    cbar = fig.colorbar(last_visc_mappable, cax=cax)
    cbar.set_label(r"log$_{10}$($\eta_{\mathrm{eff}}$ / Pa$\cdot$s)", fontsize=10)
    cbar.set_ticks([16, 18, 20, 22, 24, 26])
    cbar.ax.tick_params(labelsize=9, length=2.5, width=0.6)

    # Legend below Width label (gap so they do not overlap), then crop bottom whitespace
    place_legend_and_trim(fig, axes, leg_ax, colors, extra_axes=[cax])

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
