"""
Fig. S4 — Pyroxenite alone does not drive convective dripping.

2 models (Fig3a–b) × 3 times (display 0 / 100 / 200 Myr).
Composition only; Nature-style layout from plot_fig2_strength.py,
one-row legend style from Chemical Plume plot_fig3.py.
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
OUT_PNG = BASE / "FigS4_pyroxenite_alone.png"

# Display time = model_time_Myr - TIME0_MODEL_MYR  (model 50 Myr → "0 Myr")
TIME0_MODEL_MYR = 50.0

# Zoom window (km) — same as Fig2 plot_fig2_strength.py
X_SIZE, Z_SIZE = 800.0, 600.0
X_BEG, X_END = 200.0, 600.0
Z_BEG, Z_END = 0.0, 300.0
Z_OFFSET = 10.0  # MATLAB: z_vec = z_vec - 10


@dataclass(frozen=True)
class PanelRow:
    """One figure row = one model folder, three gouc steps (cols 0–2)."""

    folder: str
    # Primary: set gouc step numbers for Spyder (None → auto by time below)
    steps: tuple[int | None, int | None, int | None] = (None, None, None)
    # Fallback targets (Myr, display time) if steps[j] is None
    t_display: tuple[float, float, float] = (0.0, 100.0, 200.0)
    # 'normal' → Normal ML; 'weak' → Weak ML
    ml_style: str = "normal"
    annotation: str | None = None
    # Depth (km) of the TOP of the annotation text (first panel only)
    annotation_depth_km: float | None = None
    # Override displayed times (e.g. force first panel to 0)
    force_display: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )


# --- Panel file selection (folder under BASE, gouc{step}.txt) ---------------
# Row order = figure rows a–c, d–f
ROWS = [
    PanelRow(
        folder="Fig3a_Pyr_DryML",
        steps=(51, 151, 251),  # model 50 / 150 / 250 → display 0 / 100 / 200
        ml_style="normal",
        annotation="Normal (dry) M.L.",
        annotation_depth_km=110.0,
        force_display=(0.0, None, None),
    ),
    PanelRow(
        folder="Fig3b_50MyrInit_Pyr",
        steps=(51, 151, 251),  # model 50 / 150 / 250 → display 0 / 100 / 200
        ml_style="weak",
        annotation="Weakened (wet) M.L.",
        annotation_depth_km=110.0,
        force_display=(0.0, None, None),
    ),
]

# Legend: (display material index, name) — short / two-line for one-row fit
MATERIAL_LEGEND = [
    (0, "Air"),
    (5, "Upper Crust"),
    (6, "Lower Crust"),
    (12, "Eclogite"),
    (13, "Pyroxenite"),
    (10, "Asthenosphere"),
    (7, "Normal Mantle\nLithosphere"),
    (8, "Weak Mantle\nLithosphere"),
]

# %% ========================================================================
# Style / figure geometry (usually leave alone)
# ============================================================================
# Nature standard: single 89 mm, double 183 mm; page depth ≤ 247 mm
FIG_W_MM = 183.0
FIG_W_IN = FIG_W_MM / 25.4
DPI = 600
N_COLS = 3
HSPACE = 0.02  # same as Fig2
WSPACE = 0.08  # same as Fig2 (3-column)

# Legend swatch: keep width; thickness from width for 9:6
LEGEND_SWATCH_W_PT = 18
LEGEND_SWATCH_H_PT = 12  # W * 6/9
LEGEND_LABEL_GAP = 0.08
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = 8


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
    colors[7] = [104 / 255, 148 / 255, 92 / 255]  # Normal ML (Fig3a Dry)
    colors[8] = [168 / 255, 191 / 255, 143 / 255]  # Weak ML (Fig3b Wet)
    colors[10] = [245 / 255, 210 / 255, 188 / 255]  # Asthenosphere
    colors[12] = [152 / 255, 79 / 255, 49 / 255]  # Eclogite
    colors[13] = [44 / 255, 47 / 255, 59 / 255]  # Pyroxenite
    colors[14] = colors[13]
    colors[15] = colors[12]

    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, colors


def remap_for_display(grid: np.ndarray, ml_style: str) -> np.ndarray:
    """Map model rock IDs onto the unified legend colours."""
    out = np.array(grid, copy=True)
    out = np.where(out == 14, 13, out)
    out = np.where(out == 15, 12, out)
    if ml_style == "normal":
        out = np.where(out == 8, 7, out)
    elif ml_style == "weak":
        out = np.where(out == 7, 8, out)
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_numbers(path: Path) -> np.ndarray:
    return np.fromfile(path, sep=" ", dtype=np.float64)


def read_time_myr(path: Path) -> float:
    """Fast: read only the leading time token (years → Myr)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        chunk = f.read(80)
    return float(chunk.split()[0]) / 1.0e6


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


def nearest_step_by_display(
    folder: Path,
    target_display_myr: float,
) -> tuple[int, float, Path]:
    best = None
    for step, path in list_steps(folder):
        t_model = read_time_myr(path)
        t_disp = t_model - TIME0_MODEL_MYR
        if best is None or abs(t_disp - target_display_myr) < abs(
            best[1] - TIME0_MODEL_MYR - target_display_myr
        ):
            best = (step, t_model, path)
    if best is None:
        raise FileNotFoundError(f"No gouc*.txt in {folder}")
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
    """Same geometry recipe as plot_fig2_strength.py."""
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
        ax.set_xlabel("Width (km)", labelpad=3)
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
        0.03,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="k",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
    )


def time_label(ax, t_disp: float):
    ax.text(
        0.97,
        0.96,
        f"{format_display_time(t_disp)} Myr",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="k",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
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
        fontsize=12,
        color="k",
        fontstyle="italic",
        clip_on=True,
    )


def add_material_legend(leg_ax, colors: np.ndarray):
    """Single-row legend (Chemical Plume plot_fig3.py style); swatch 9:6."""
    n = len(MATERIAL_LEGEND)
    label_gap = LEGEND_LABEL_GAP
    y_top = 0.90

    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    bbox = leg_ax.get_window_extent().transformed(leg_ax.figure.dpi_scale_trans.inverted())
    ax_w_pt = bbox.width * 72.0
    ax_h_pt = bbox.height * 72.0
    sw_w = LEGEND_SWATCH_W_PT / max(ax_w_pt, 1.0)  # keep original width
    # Thickness only: physical height = width * 6/9
    sw_h = (LEGEND_SWATCH_W_PT * 6.0 / 9.0) / max(ax_h_pt, 1.0)
    # Move labels down by swatch thickening (fixed width, height -> W*6/9)
    # delta_h = h_9x6 - h_prev; h_prev taken as flatter W/2 (2:1) with same width
    label_down = sw_h - (LEGEND_SWATCH_W_PT * 0.5) / max(ax_h_pt, 1.0)

    centres = np.linspace(0.5 * sw_w, 1.0 - 0.5 * sw_w, n)

    for k, (idx, name) in enumerate(MATERIAL_LEGEND):
        cx = float(centres[k])
        y0 = y_top - sw_h
        x0 = cx - 0.5 * sw_w
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
    t = read_time_myr(path)
    return step, t, path


def resolve_panel(row: PanelRow, col: int) -> tuple[int, float, Path]:
    """Resolve (step, model_time_Myr, path) from USER CONTROLS above."""
    folder = BASE / row.folder
    step = row.steps[col]
    if step is not None:
        return _step_path(folder, step)
    return nearest_step_by_display(folder, row.t_display[col])


# %% ========================================================================
# Figure
# ============================================================================
def main():
    apply_nature_style()
    cmap_c, norm_c, colors = material_cmap()

    n_rows = len(ROWS)
    n_cols = N_COLS
    # Same side margins as plot_fig2_strength.py
    left, right = 0.10, 0.915

    # Inch-based layout (avoids fraction gaps between panels and legend):
    # [top pad][panels equal-km][Width gap][legend][bottom pad]
    usable_w = FIG_W_IN * (right - left)
    panel_w_in = usable_w / (n_cols + (n_cols - 1) * WSPACE)
    data_asp = (Z_END - Z_BEG) / (X_END - X_BEG)  # Fig2: 300/400 = 0.75
    panel_h_in = panel_w_in * data_asp
    row_gap_in = HSPACE * panel_h_in
    panels_h_in = n_rows * panel_h_in + (n_rows - 1) * row_gap_in

    top_pad_in = 0.08
    width_label_gap_in = 0.55  # tick labels + "Width (km)" — keep clear of legend
    leg_h_in = 0.50
    bottom_pad_in = 0.05
    fig_h = top_pad_in + panels_h_in + width_label_gap_in + leg_h_in + bottom_pad_in
    fig_h = min(fig_h, 247.0 / 25.4)

    fig = plt.figure(figsize=(FIG_W_IN, fig_h))

    # Column x positions (figure fraction)
    col_w = panel_w_in / FIG_W_IN
    col_gap = (WSPACE * panel_w_in) / FIG_W_IN
    x0s = [left + j * (col_w + col_gap) for j in range(n_cols)]

    # Row y positions — stack downward from top
    y_top = 1.0 - top_pad_in / fig_h
    panel_h_frac = panel_h_in / fig_h
    row_gap_frac = row_gap_in / fig_h
    y0s = [
        y_top - (i + 1) * panel_h_frac - i * row_gap_frac for i in range(n_rows)
    ]

    axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j] = fig.add_axes([x0s[j], y0s[i], col_w, panel_h_frac])

    leg_y0 = bottom_pad_in / fig_h
    leg_h_frac = leg_h_in / fig_h
    leg_ax = fig.add_axes([x0s[0], leg_y0, x0s[-1] + col_w - x0s[0], leg_h_frac])

    letters = "abcdef"

    for i, row in enumerate(ROWS):
        for j in range(n_cols):
            ax = axes[i, j]
            last = i == n_rows - 1
            step, t_model, path = resolve_panel(row, j)
            t_yr, grid = decode_composition(path)
            t_model = t_yr / 1.0e6
            t_disp = t_model - TIME0_MODEL_MYR
            if row.force_display[j] is not None:
                t_disp = row.force_display[j]

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
            # One "Width (km)" on the centre panel only — avoids colliding with legend
            style_axis(
                ax,
                show_xlabel=(last and j == 1),
                show_ylabel=(j == 0),
                show_xticklabels=last,
            )
            panel_label(ax, letters[i * n_cols + j])
            time_label(ax, t_disp)

            if j == 0 and row.annotation and row.annotation_depth_km is not None:
                add_ml_annotation(ax, row.annotation, row.annotation_depth_km)

            print(
                f"{row.folder:28s} col={j} step={step:4d} "
                f"model={t_model:7.2f} Myr  display={t_disp:7.2f} Myr  ({path.name})"
            )

    pos = axes[0, 0].get_position()
    z_span = (Z_END - Z_OFFSET) - (Z_BEG - Z_OFFSET)
    x_span = X_END - X_BEG
    phys_aspect = (pos.height * fig_h) / (pos.width * FIG_W_IN)
    print(
        f"Panel aspect (physical h/w) = {phys_aspect:.4f}, "
        f"data z/x = {z_span / x_span:.4f} "
        f"(x={x_span:.0f} km, z={z_span:.0f} km; zoom same as Fig2)"
    )

    pos_l = axes[-1, 0].get_position()
    print(
        f"Legend y0={leg_y0:.3f}, panel bottom y0={pos_l.y0:.3f}, "
        f"gap={pos_l.y0 - (leg_y0 + leg_h_frac):.3f}"
    )
    add_material_legend(leg_ax, colors)

    fig.canvas.draw()
    for lab in axes[-1, -1].get_xticklabels():
        if lab.get_text().strip() == "600":
            lab.set_ha("right")
    for lab in axes[-1, 0].get_xticklabels():
        if lab.get_text().strip() == "200":
            lab.set_ha("left")

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
