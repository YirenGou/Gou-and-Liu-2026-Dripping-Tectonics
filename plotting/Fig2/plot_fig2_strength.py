"""
Fig. 2 — The strength of mantle lithosphere.

5 rows × 3 columns (rows a–d use 2 panels; row e uses 3).
Display time = model_time − TIME0_MODEL_MYR (model 50 Myr → "0 Myr").
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
# Paths
BASE = Path(__file__).resolve().parent
OUT_PNG = BASE / "Fig2_strength_mantle_lithosphere.png"

# Display time = model_time_Myr - TIME0_MODEL_MYR  (model 50 Myr → "0 Myr")
TIME0_MODEL_MYR = 50.0

# Zoom window (km) — same as Fig1a/b/d vis_cto3.m
X_SIZE, Z_SIZE = 800.0, 600.0
X_BEG, X_END = 200.0, 600.0
Z_BEG, Z_END = 0.0, 300.0
Z_OFFSET = 10.0  # MATLAB: z_vec = z_vec - 10

# Grid: 5 rows × 3 columns (unused cells in a row stay blank)
N_COLS = 3


@dataclass(frozen=True)
class PanelRow:
    """One figure row = one model folder; up to N_COLS gouc steps (L→R)."""

    folder: str
    # gouc step numbers, left to right (length 1–3)
    steps: tuple[int, ...]
    # 'normal' → rock 8 as Normal ML; 'weak' → Weak ML (+ carbonated if present)
    ml_style: str = "normal"
    annotation: str | None = None
    # Depth (km) of the TOP of the annotation text (leftmost panel only)
    annotation_depth_km: float | None = None
    # Override displayed time on the leftmost panel (e.g. 0 for a/c)
    force_display_left: float | None = None
    # Override displayed time by column (None = use model time)
    force_display: tuple[float | None, ...] | None = None


# --- Panel file selection (folder under BASE, gouc{step}.txt) ---------------
ROWS = [
    PanelRow(
        folder="Fig2a_Normal80km_NoIsostasy",
        steps=(53, 363, 499),
        ml_style="normal",
        annotation="Normal (dry) M.L.",
        annotation_depth_km=110.0,
        force_display_left=0.0,
        # panel c: model ~100.5 Myr → show integer "100"
        force_display=(None, None, 100.0),
    ),
    PanelRow(
        folder="Fig2b_Normal300km_NoIsostasy",
        steps=(56, 548, 1076),
        ml_style="normal",
        annotation="Normal (dry) M.L.",
        annotation_depth_km=110.0,
        force_display_left=0.0,
    ),
    PanelRow(
        folder="Fig2c_50MyrInitiation_80km_NoIsostasy",
        steps=(52, 170, 240),
        ml_style="weak",
        annotation="Weakened (wet) M.L.",
        annotation_depth_km=100.0,
    ),
    PanelRow(
        folder="Fig2d_50MyrIni_Carbonated_Isostasy_80km_NoIsostasy_Carb38km",
        steps=(52, 155, 330),
        ml_style="weak",
        annotation="Weakened (wet) M.L.",
        annotation_depth_km=90.0,
    ),
    PanelRow(
        folder="Fig2e_NoIsostasy",
        steps=(54, 100, 200),
        ml_style="weak",
        annotation="Weakened (wet) M.L.",
        annotation_depth_km=125.0,
        force_display_left=0.0,
    ),
]

# Legend columns (top → bottom within each column)
# Col0: Air / Asthenosphere
# Col1: Upper Crust / Lower Crust
# Col2: Normal / Weak Mantle Lithosphere
# Col3: Eclogite / Pyroxenite
# Col4: Carbonatite only
MATERIAL_LEGEND_COLS = [
    [(0, "Air"), (10, "Asthenosphere")],
    [(5, "Upper Crust"), (6, "Lower Crust")],
    [(7, "Normal Mantle\nLithosphere"), (8, "Weak Mantle\nLithosphere")],
    [(12, "Eclogite"), (14, "Pyroxenite")],
    [(9, "Carbonatite-\nmetasomatized Mantle")],
]

# %% ========================================================================
# Style / figure geometry (usually leave alone)
# ============================================================================
# Nature standard: single 89 mm, double 183 mm; page depth ≤ 247 mm
FIG_W_MM = 183.0
FIG_W_IN = FIG_W_MM / 25.4
DPI = 1200
HSPACE = 0.02  # tight vertical gap between panel rows
WSPACE = 0.08  # 3-column spacing

# Legend swatch: keep width; thickness from width for 9:6
LEGEND_SWATCH_W_PT = 24
LEGEND_SWATCH_H_PT = 16  # W * 6/9
LEGEND_LABEL_GAP = 0.04
LEGEND_ROW_GAP = 0.015
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = TICK_LABELSIZE


# ---------------------------------------------------------------------------
# Colormap (unified display indices)
# ---------------------------------------------------------------------------
def material_cmap():
    """Discrete material colours; indices match rock IDs after remapping."""
    colors = np.zeros((39, 3), dtype=float)
    colors[:] = 1.0
    colors[0] = [1.0, 1.0, 1.0]  # Air
    colors[5] = [236 / 255, 235 / 255, 194 / 255]  # Upper Crust
    colors[6] = [202 / 255, 178 / 255, 114 / 255]  # Lower Crust
    colors[7] = [104 / 255, 148 / 255, 92 / 255]  # Normal ML (Fig1a vis Dry)
    colors[8] = [168 / 255, 191 / 255, 143 / 255]  # Weak ML (Fig1c/d vis Wet)
    colors[9] = [176 / 255, 176 / 255, 179 / 255]  # Carbonated ML (light gray; distinct from pyroxenite)
    colors[10] = [245 / 255, 210 / 255, 188 / 255]  # Asthenosphere
    colors[12] = [152 / 255, 79 / 255, 49 / 255]  # Eclogite
    colors[13] = [69 / 255, 70 / 255, 94 / 255]  # Pyroxenite (vis Wet Pyr)
    colors[14] = [69 / 255, 70 / 255, 94 / 255]  # Pyroxenite
    colors[15] = colors[8]  # Weakened_mantle_lithosphere → weak green

    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, colors


def remap_for_display(grid: np.ndarray, ml_style: str) -> np.ndarray:
    """Map model rock IDs onto the unified legend colours."""
    out = np.array(grid, copy=True)
    if ml_style == "normal":
        out = np.where(out == 8, 7, out)
    # weak: keep 8 as Weak ML; 15 also weak (already same colour)
    return out


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


def rle_count_material(path: Path, material: int) -> tuple[float, int]:
    """Return (time_Myr, count) of a material without building the full grid."""
    data = load_numbers(path)
    if data.size < 3:
        return float("nan"), 0
    time = float(data[0]) / 1.0e6
    vec = data[3:]
    num = 0
    count = 0
    n = len(vec)
    while num < n:
        value = vec[num]
        if value == -2:
            n_colors = int(vec[num + 1])
            mat = vec[num + 2]
            if mat == material:
                count += n_colors
            num += 3
        else:
            if value == material:
                count += 1
            num += 1
    return time, count


def list_steps(folder: Path) -> list[tuple[int, Path]]:
    steps = []
    for p in folder.glob("gouc*.txt"):
        if p.stat().st_size < 100:
            continue
        try:
            step = int(p.stem[4:])
        except ValueError:
            continue
        steps.append((step, p))
    steps.sort(key=lambda x: x[0])
    return steps


def nearest_step(
    folder: Path,
    target_myr: float,
    *,
    min_myr: float | None = None,
) -> tuple[int, float, Path]:
    best = None
    for step, path in list_steps(folder):
        t = float(load_numbers(path)[0]) / 1.0e6
        if min_myr is not None and t < min_myr:
            continue
        if best is None or abs(t - target_myr) < abs(best[1] - target_myr):
            best = (step, t, path)
    if best is None and min_myr is not None:
        # Fallback if every snapshot is earlier than min_myr
        return nearest_step(folder, target_myr, min_myr=None)
    if best is None:
        raise FileNotFoundError(f"No gouc*.txt in {folder}")
    return best


def first_carbonated_step(folder: Path, after_myr: float = 49.5) -> tuple[int, float, Path]:
    """First output at/after after_myr that contains rock 9 (Carbonated ML)."""
    best = None
    for step, path in list_steps(folder):
        t, n9 = rle_count_material(path, 9)
        if np.isnan(t) or t < after_myr or n9 <= 0:
            continue
        if best is None or t < best[1]:
            best = (step, t, path)
    if best is None:
        return nearest_step(folder, TIME0_MODEL_MYR)
    return best


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


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def apply_nature_style():
    # Draft slightly above Nature's 5–8 pt print guidance so labels stay
    # readable at 183 mm double-column width after journal scaling.
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
            # Keep exact Nature width; do not crop with bbox_inches='tight'
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
        }
    )


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


def wspace_factor() -> float:
    return WSPACE


def style_axis(
    ax,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    show_xticklabels: bool,
    right_column: bool = False,
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
        # Edge ticks: pull inward so they stay in-figure and don't collide across columns
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


def case_label(ax, text: str):
    """Case name at the lower-left corner of the leftmost panel in a row."""
    ax.text(
        0.02,
        0.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="k",
        bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
    )


def add_ml_annotation(ax, text: str, depth_km: float):
    """Place italic label; text top aligned to depth_km (depth positive down)."""
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


def _text_width_axes(leg_ax, text: str, fontsize: float) -> float:
    """Width of multiline text as a fraction of legend-axes width."""
    fig = leg_ax.figure
    # Probe text at origin; measure longest line in axes coordinates
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
    """Column-wise legend; keep left-swatch / right-Carbonatite flush alignment.

    Col0 swatch left = panel left; Carbonatite text right = panel right;
    labels centred on their swatches (same centre within each column).
    """
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

    # Right flush from Carbonatite label width (last column)
    carb_name = cols[-1][0][1]
    w_carb = _text_width_axes(leg_ax, carb_name, LEGEND_FONTSIZE)
    cx0 = 0.5 * sw_w
    cx_right = 1.0 - 0.5 * w_carb
    centres = np.linspace(cx0, cx_right, ncols)

    # Vertical layout: two slots (top / bottom); single-item columns use top slot
    max_lines_top = max(
        (1 + items[0][1].count("\n")) if items else 1 for items in cols
    )
    max_lines_bot = max(
        (1 + items[1][1].count("\n")) if len(items) > 1 else 1 for items in cols
    )
    text_h_top = (max_lines_top * 1.25) * LEGEND_FONTSIZE / max(ax_h_pt, 1.0)
    text_h_bot = (max_lines_bot * 1.25) * LEGEND_FONTSIZE / max(ax_h_pt, 1.0)

    y_top_swatch = 0.97 - sw_h
    y_bot_swatch = (
        y_top_swatch - LEGEND_LABEL_GAP - label_down - text_h_top - LEGEND_ROW_GAP - sw_h
    )
    row_y0 = [y_top_swatch, y_bot_swatch]

    for col, items in enumerate(cols):
        cx = float(centres[col])
        if col == 0:
            # Air / Asthenosphere: swatch left = panel left
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


def _step_path(folder: Path, step: int) -> tuple[int, float, Path]:
    path = folder / f"gouc{step}.txt"
    if not path.exists() or path.stat().st_size < 100:
        raise FileNotFoundError(f"Missing composition file: {path}")
    t = float(load_numbers(path)[0]) / 1.0e6
    return step, t, path


def resolve_panel(row: PanelRow, j: int) -> tuple[int, float, Path]:
    """Resolve (step, model_time_Myr, path) for column j from USER CONTROLS."""
    return _step_path(BASE / row.folder, row.steps[j])


# %% ========================================================================
# Figure
# ============================================================================
def main():
    apply_nature_style()
    cmap_c, norm_c, colors = material_cmap()

    n_rows = len(ROWS)
    n_cols = N_COLS
    # Extra right margin so rightmost "600" is fully inside the figure
    left, right, top = 0.10, 0.915, 0.985
    # Leave bottom margin for x-labels + two-row legend below row 5
    panel_bottom = 0.20
    fig_h = figure_height_in(
        n_rows, n_cols, hspace=HSPACE, bottom=panel_bottom, top=top
    )
    fig_h += 1.20
    # Nature page depth is 247 mm; pack_panel_rows scales if needed
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
    for i, row in enumerate(ROWS):
        for j in range(n_cols):
            if j < len(row.steps):
                axes[i, j] = fig.add_subplot(gs[i, j])
            else:
                axes[i, j] = None

    leg_ax = fig.add_axes([left, 0.015, right - left, 0.11])
    letters = "abcdefghijklmno"
    letter_i = 0

    for i, row in enumerate(ROWS):
        n_here = len(row.steps)
        last = i == n_rows - 1
        for j in range(n_here):
            ax = axes[i, j]
            step, t_model, path = resolve_panel(row, j)
            t_yr, grid = decode_composition(path)
            t_model = t_yr / 1.0e6
            t_disp = t_model - TIME0_MODEL_MYR
            if (
                row.force_display is not None
                and j < len(row.force_display)
                and row.force_display[j] is not None
            ):
                t_disp = row.force_display[j]
            elif j == 0 and row.force_display_left is not None:
                t_disp = row.force_display_left

            grid, x, z = zoom_composition(grid)
            grid = remap_for_display(grid, row.ml_style)

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
                right_column=(j == n_here - 1),
            )
            panel_label(ax, letters[letter_i])
            letter_i += 1
            time_label(ax, t_disp)
            if j == 0:
                case_label(ax, f"Case {i + 1}")

            if j == 0 and row.annotation and row.annotation_depth_km is not None:
                add_ml_annotation(ax, row.annotation, row.annotation_depth_km)

            print(
                f"{row.folder:28s} col{j} step={step:4d} "
                f"model={t_model:7.2f} Myr  display={t_disp:7.2f} Myr  ({path.name})"
            )

    pack_panel_rows(axes, HSPACE, min_bottom=0.185)

    # Legend below the 5th row, spanning leftmost → rightmost panel of that row
    pos_l = axes[-1, 0].get_position()
    last_j = len(ROWS[-1].steps) - 1
    pos_r = axes[-1, last_j].get_position()
    leg_h = 0.115
    leg_gap = 0.050  # room for "Width (km)"
    leg_y0 = max(0.018, pos_l.y0 - leg_gap - leg_h)
    leg_ax.set_position([pos_l.x0, leg_y0, pos_r.x1 - pos_l.x0, leg_h])
    add_material_legend(leg_ax, colors)

    # Re-assert edge tick alignment on bottom row
    fig.canvas.draw()
    for j in range(len(ROWS[-1].steps)):
        ax = axes[-1, j]
        if ax is None:
            continue
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
    print(f"Figure size: {w_mm:.1f} × {h_mm:.1f} mm (Nature double-column width = {FIG_W_MM:.0f} mm)")
    plt.close(fig)


if __name__ == "__main__":
    main()
