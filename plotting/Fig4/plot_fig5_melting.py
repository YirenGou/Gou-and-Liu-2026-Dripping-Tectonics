"""
Fig. 5 — Partial melting driven by the eclogite/pyroxenite drip.

a  model initiation (model 50 Myr shown as 0 Myr), zoom on the dense body
b  step 105, composition + melt-fraction contours
c  step 200, composition + melt-fraction contours
d  P–T paths of one tracked pyroxenite marker and one tracked asthenosphere
   marker on the solidus/liquidus grid.  The asthenosphere marker rises across
   the wet solidus and then turns back toward higher pressure.

Melt fractions follow vis_cto3_Melt_Wet.m; solidus/liquidus curves follow
PlotP-T-New.py.  Marker paths come from the Lagrangian marker files of the same
run (see build_marker_pt.py).  Style matches plot_fig3_existence.py (Nature
double column).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from scipy.ndimage import gaussian_filter

import fig5_data as fd

# %% ========================================================================
# USER CONTROLS
# ============================================================================
BASE = Path(__file__).resolve().parent
OUT_PNG = BASE / "Fig5_partial_melting.png"
OUT_PDF = BASE / "Fig5_partial_melting.pdf"
OUT_JPG = BASE / "Fig5_partial_melting.jpg"
MARKER_NPZ = BASE / "marker_pt.npz"
MARKER_NPZ_ASTH = BASE / "marker_pt_asth.npz"

STEP_A = 53  # model 50.000 Myr → displayed as 0 Myr
STEP_B = 115  # strongest pyroxenite melting within +/-10 steps of 105
STEP_C = 205  # strongest asthenosphere melting within +/-10 steps of 200
TIME0_MODEL_MYR = 50.0

# Common window for panels a/b/c (km, plotted depth)
SEC_XLIM = (250.0, 550.0)
SEC_ZLIM = (-10.0, 400.0)
SEC_XTICKS = [300, 400, 500]
SEC_ZTICKS = [0, 100, 200, 300, 400]

# Melt-fraction contours + translucent fills
MELT_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.20]
MELT_VMAX = 0.20
MELT_SMOOTH_SIGMA = 1.4  # cells of the strided grid (~0.5 km) — keeps thin films
MELT_LW = 0.7
MELT_HALO_LW = 1.7  # white outline so lines read on both the dark drip and the mantle
MELT_FILL_ALPHA = 0.55

# Grid strides
STRIDE_DISPLAY = 2
STRIDE_MELT = 3

# Panel d — tracked markers (ids refer to the i2vis marker array)
MARKER_PYR = 378201  # pyroxenite marker, starts at (404.6, 86.7) km
MARKER_UPW = 204703  # asthenosphere: decompresses across Solidus_wet, then returns
PT_PRESSURE_LIM = (0.0, 13.0)  # ~400 km, the depth range of panels a-c
PT_TEMP_LIM = (600.0, 1900.0)
PT_MAX_DEPTH_KM = 520.0  # stop the drip paths before they pile up on the model base
# pyroxenite path: initiation → panel c.  upwelling path continues a little
# past panel c so the return (repressurization) is visible.
# Heads on each marker path, including the one at the end.  Placed at equal
# screen-arc intervals (see add_direction_arrows).
PT_ARROW_N_PYR = 10
PT_ARROW_N_UPW = 5
# "-|>" head length and width are both 0.4 * mutation_scale (points).
PT_ARROW_SIZE = 15  # former pyroxenite heads; every direction head uses this
UPW_RETURN_GPA = 0.55  # keep this much P increase after Pmin, then stop
UPW_COLOR = "#C2185B"

# %% ========================================================================
# Style / geometry
# ============================================================================
FIG_W_MM = 183.0
FIG_W_IN = FIG_W_MM / 25.4
DPI = 600

TICK_LABELSIZE = 8
AXIS_LABELSIZE = 9
PANEL_LABELSIZE = 11
TIME_LABELSIZE = 9
LEGEND_FONTSIZE = 7

# Legend swatch: keep width; thickness from width for 9:6
LEGEND_SWATCH_W_PT = 18
LEGEND_SWATCH_H_PT = 12  # W * 6/9
LEGEND_LABEL_GAP = 0.10

MATERIAL_LEGEND = [
    (0, "Air"),
    (5, "Upper Crust"),
    (6, "Lower Crust"),
    (12, "Eclogite"),
    (13, "Pyroxenite"),
    (10, "Asthenosphere"),
    (8, "Weak Mantle\nLithosphere"),
]

ECL_COLOR = "#984F31"  # 152, 79, 49
PYR_COLOR = "#2C2F3B"  # 44, 47, 59


def material_cmap():
    colors = np.ones((39, 3), dtype=float)
    colors[0] = [1.0, 1.0, 1.0]  # Air
    colors[5] = [236 / 255, 235 / 255, 194 / 255]  # Upper crust
    colors[6] = [202 / 255, 178 / 255, 114 / 255]  # Lower crust
    colors[7] = [104 / 255, 148 / 255, 92 / 255]  # Normal mantle lithosphere
    colors[8] = [168 / 255, 191 / 255, 143 / 255]  # Weak mantle lithosphere
    colors[10] = [245 / 255, 210 / 255, 188 / 255]  # Asthenosphere
    colors[12] = [152 / 255, 79 / 255, 49 / 255]  # Eclogite
    colors[13] = [44 / 255, 47 / 255, 59 / 255]  # Pyroxenite
    colors[14] = colors[13]
    colors[15] = colors[12]

    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1.0), cmap.N)
    return cmap, norm, colors


def melt_cmap():
    """Dark violet (trace melt) → yellow (high melt); a white halo carries contrast."""
    return LinearSegmentedColormap.from_list(
        "melt", ["#3B0F70", "#8C2981", "#DE4968", "#FE9F6D", "#FDE725"]
    )


def melt_level_colors():
    norm = Normalize(0.0, MELT_VMAX)
    cmap = melt_cmap()
    return [cmap(norm(lv)) for lv in MELT_LEVELS]


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TICK_LABELSIZE,
            "axes.labelsize": AXIS_LABELSIZE,
            "axes.titlesize": AXIS_LABELSIZE,
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
        }
    )


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def panel_label(ax, label, x=0.015, y=0.97):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABELSIZE,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
    )


def time_label(ax, t_disp):
    txt = "0" if t_disp < 0.05 else f"{t_disp:.1f}"
    ax.text(
        0.985,
        0.97,
        f"{txt} Myr",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=TIME_LABELSIZE,
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
    )


def draw_composition(ax, data, xlim, zlim, stride):
    comp = data["comp"][::stride, ::stride]
    x = data["xc"][::stride]
    z = data["zc"][::stride] - fd.Z_OFFSET

    ix = (x >= xlim[0] - 2) & (x <= xlim[1] + 2)
    iz = (z >= zlim[0] - 2) & (z <= zlim[1] + 2)
    cmap, norm, _ = material_cmap()
    ax.pcolormesh(
        x[ix],
        z[iz],
        comp[np.ix_(iz, ix)],
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
    )


def draw_melt_contours(ax, data, xlim, zlim):
    x, zdep, m, _ = fd.melt_on_composition_grid(data, stride=STRIDE_MELT)
    if MELT_SMOOTH_SIGMA > 0:
        m = gaussian_filter(m, MELT_SMOOTH_SIGMA, mode="nearest")

    ix = (x >= xlim[0]) & (x <= xlim[1])
    iz = (zdep >= zlim[0]) & (zdep <= zlim[1])
    xs, zs, ms = x[ix], zdep[iz], m[np.ix_(iz, ix)]

    colors = melt_level_colors()
    ax.contourf(
        xs,
        zs,
        ms,
        levels=MELT_LEVELS + [1.0],
        colors=colors,
        alpha=MELT_FILL_ALPHA,
        zorder=2,
        antialiased=True,
    )
    cs = ax.contour(
        xs,
        zs,
        ms,
        levels=MELT_LEVELS,
        colors=colors,
        linewidths=MELT_LW,
        zorder=3,
    )
    cs.set(
        path_effects=[pe.withStroke(linewidth=MELT_HALO_LW, foreground="white", alpha=0.85)]
    )
    print(f"    melt max = {ms.max():.3f}")


def add_melt_legend(ax):
    """Discrete melt key in the lower-right of the panel."""
    handles = [
        Patch(
            facecolor=(*c[:3], MELT_FILL_ALPHA),
            edgecolor=c,
            linewidth=0.6,
            label=f"{lv * 100:g}",
        )
        for lv, c in zip(MELT_LEVELS, melt_level_colors())
    ]
    leg = ax.legend(
        handles=handles,
        title="Melt fraction (%)",
        loc="lower right",
        ncol=len(MELT_LEVELS),
        fontsize=5.5,
        title_fontsize=TICK_LABELSIZE + 1,
        frameon=True,
        framealpha=0.88,
        borderpad=0.28,
        handlelength=0.85,
        handleheight=0.75,
        handletextpad=0.3,
        columnspacing=0.5,
        borderaxespad=0.3,
    )
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("0.6")
    leg.set_zorder(6)


def add_wet_solidus_label(ax):
    ax.text(
        0.03,
        0.025,
        "Wet solidus",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=TICK_LABELSIZE,
    )


def style_section_axis(ax, xlim, zlim, xticks, zticks, *, xlabel, ylabel):
    ax.set_xlim(*xlim)
    ax.set_ylim(zlim[1], zlim[0])
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.tick_params(which="both", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_axisbelow(False)
    if xlabel:
        ax.set_xlabel("Width (km)", labelpad=2)
    if ylabel:
        ax.set_ylabel("Depth (km)", labelpad=2)
    labels = ax.set_xticklabels([f"{v:g}" for v in xticks])
    if xticks[0] <= xlim[0]:
        labels[0].set_ha("left")
    if xticks[-1] >= xlim[1]:
        labels[-1].set_ha("right")


# ---------------------------------------------------------------------------
# Panel d
# ---------------------------------------------------------------------------
def _keep_upwelling_return(P):
    """True up to the repressurization hook; drop the later parked tail."""
    keep = np.ones(P.size, dtype=bool)
    i_pmin = int(np.argmin(P))
    after = P[i_pmin:] - P[i_pmin]
    hit = np.flatnonzero(after >= UPW_RETURN_GPA)
    if hit.size:
        i_end = min(i_pmin + int(hit[0]) + 3, P.size - 1)
        keep[i_end + 1 :] = False
    return keep


def load_paths():
    """P–T–t paths of the tracked markers over the interval spanned by panels b/c."""
    sources = {
        "pyr": (MARKER_NPZ, MARKER_PYR, fd.solidus_pyroxenite),
        "upw": (MARKER_NPZ_ASTH, MARKER_UPW, fd.solidus_wet),
    }
    cache = {}
    out = {}
    for name, (npz, mid, sol) in sources.items():
        d = cache.setdefault(npz, np.load(npz))
        k = int(np.where(d["ids"] == mid)[0][0])
        z = d["depth"][k]
        t = d["time_myr"] - TIME0_MODEL_MYR
        if name == "upw":
            keep = _keep_upwelling_return(d["P"][k])
        else:
            t_end = float(t[np.searchsorted(d["steps"], STEP_C)])
            keep = t <= t_end
        # the drip markers heat isobarically once they reach the model base
        deep = np.flatnonzero(z > PT_MAX_DEPTH_KM)
        if deep.size:
            keep[deep[0] :] = False
        T, P = d["T"][k][keep], d["P"][k][keep]
        out[name] = {
            "id": int(mid),
            "T": T,
            "P": P,
            "t": t[keep],
            "z": z[keep],
            "x": d["x"][k][keep],
            "supra": T > sol(P),
        }
    return out


def _arrow(ax, p0, p1, color, size):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=size,
            color=color,
            lw=0,
            shrinkA=0,
            shrinkB=0,
            zorder=9,
        )
    )


def _screen_arc(ax, T, P):
    """Cumulative path length in inches on the axes, so heads can be spaced evenly."""
    pos = ax.get_position()
    w = pos.width * ax.figure.get_figwidth()
    h = pos.height * ax.figure.get_figheight()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx = w / abs(x1 - x0)
    sy = h / abs(y1 - y0)
    seg = np.hypot(np.diff(T) * sx, np.diff(P) * sy)
    return np.r_[0.0, np.cumsum(seg)]


def add_direction_arrows(ax, T, P, color, n=PT_ARROW_N_PYR, size=PT_ARROW_SIZE,
                         terminal_span=0.04):
    """*n* arrowheads at equal screen-arc intervals, including one at the end.

    The tip of each head sits on the interval boundary.  Direction is taken
    from the preceding bit of the path so a short wiggle does not turn the head.
    """
    s = _screen_arc(ax, T, P)
    if s[-1] <= 0 or n < 1:
        return
    sn = s / s[-1]
    used = set()
    for k in range(1, int(n) + 1):
        i_tip = int(np.searchsorted(sn, min(k / n, 1.0)))
        i_tip = min(max(i_tip, 1), len(T) - 1)
        if i_tip in used:
            continue
        used.add(i_tip)
        i_tail = int(np.searchsorted(sn, max(0.0, sn[i_tip] - terminal_span)))
        i_tail = min(i_tail, i_tip - 1)
        _arrow(
            ax,
            (float(T[i_tail]), float(P[i_tail])),
            (float(T[i_tip]), float(P[i_tip])),
            color,
            size,
        )


def marker_legend_handle(color):
    """Solid line with a terminal arrowhead the same size as the path heads.

    A ``-|>`` head at ``mutation_scale`` S is 0.4*S points long and wide.
    The ``>`` marker's box is ``markersize`` points on a side, so the two match
    when ``markersize == 0.4 * PT_ARROW_SIZE``.
    """
    return Line2D(
        [0, 1],
        [0, 0],
        color=color,
        lw=1.95,
        solid_capstyle="butt",
        marker=">",
        markersize=0.4 * PT_ARROW_SIZE,
        markevery=[1],
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.0,
    )


def draw_pt_panel(ax, paths):
    p = np.linspace(0.4, PT_PRESSURE_LIM[1], 600)
    curves = [
        (fd.liquidus(p), "#9467bd", "-", "Liquidus"),
        (fd.solidus_dry(p), "#2ca02c", "--", "Solidus$_{dry}$"),
        (fd.solidus_wet(p), "#ff7f0e", "--", "Solidus$_{wet}$"),
        (fd.solidus_saturated(p), "#1f77b4", "--", "Solidus$_{sat}$"),
        (fd.solidus_pyroxenite(p), "k", "--", "Solidus$_{ecl,pyx}$"),
    ]
    for T, c, ls, lab in curves:
        ax.plot(T, p, ls, color=c, lw=1.35, label=lab, zorder=2)

    ax.set_xlim(*PT_TEMP_LIM)
    ax.set_ylim(PT_PRESSURE_LIM[1], PT_PRESSURE_LIM[0])

    for name, color, n in (
        ("pyr", PYR_COLOR, PT_ARROW_N_PYR),
        ("upw", UPW_COLOR, PT_ARROW_N_UPW),
    ):
        pth = paths[name]
        ax.plot(pth["T"], pth["P"], "-", color=color, lw=1.95, solid_capstyle="butt",
                zorder=7)
        add_direction_arrows(ax, pth["T"], pth["P"], color, n=n, size=PT_ARROW_SIZE)
        ax.plot(pth["T"][0], pth["P"][0], "o", ms=3.8, color=color,
                mec="white", mew=0.7, zorder=8)

    ax.set_xticks([800, 1000, 1200, 1400, 1600, 1800])
    ax.set_yticks([0, 5, 10])
    ax.set_xlabel("Temperature (°C)", labelpad=2)
    ax.set_ylabel("Pressure (GPa)", labelpad=2)
    ax.tick_params(which="both", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    handles, labels = ax.get_legend_handles_labels()
    lut = dict(zip(labels, handles))
    hh = [
        marker_legend_handle(PYR_COLOR),
        marker_legend_handle(UPW_COLOR),
        lut["Liquidus"],
        lut["Solidus$_{dry}$"],
        lut["Solidus$_{wet}$"],
        lut["Solidus$_{sat}$"],
        lut["Solidus$_{ecl,pyx}$"],
    ]
    ll = [
        "Pyroxenite marker",
        "Olivine marker",
        "Liquidus",
        "Solidus$_{dry}$",
        "Solidus$_{wet}$",
        "Solidus$_{sat}$",
        "Solidus$_{ecl,pyx}$",
    ]
    leg = ax.legend(
        hh,
        ll,
        loc="upper right",
        ncol=2,
        fontsize=TICK_LABELSIZE,
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0.4,
    )
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("0.6")


# ---------------------------------------------------------------------------
# Material legend row
# ---------------------------------------------------------------------------
def add_material_legend(leg_ax, colors):
    n = len(MATERIAL_LEGEND)
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    bbox = leg_ax.get_window_extent().transformed(leg_ax.figure.dpi_scale_trans.inverted())
    sw_w = LEGEND_SWATCH_W_PT / max(bbox.width * 72.0, 1.0)  # keep original width
    # Thickness only: physical height = width * 6/9
    sw_h = (LEGEND_SWATCH_W_PT * 6.0 / 9.0) / max(bbox.height * 72.0, 1.0)
    # Move labels down by swatch thickening (fixed width, height -> W*6/9)
    # delta_h = h_9x6 - h_prev; h_prev taken as flatter W/2 (2:1) with same width
    label_down = sw_h - (LEGEND_SWATCH_W_PT * 0.5) / max(bbox.height * 72.0, 1.0)

    y_top = 0.92
    # slot centres (not swatch centres) so the widest label still fits on the row
    centres = (np.arange(n) + 0.5) / n
    for (idx, name), cx in zip(MATERIAL_LEGEND, centres):
        face = colors[idx]
        edge = "0.35" if float(np.mean(face)) > 0.92 else "0.2"
        leg_ax.add_patch(
            Rectangle(
                (cx - 0.5 * sw_w, y_top - sw_h),
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
            y_top - sw_h - LEGEND_LABEL_GAP - label_down,
            name,
            transform=leg_ax.transAxes,
            ha="center",
            va="top",
            fontsize=LEGEND_FONTSIZE,
            linespacing=1.1,
            clip_on=False,
        )


# %% ========================================================================
# Figure
# ============================================================================
def main():
    apply_style()
    _, _, colors = material_cmap()

    # --- inch-based layout ---------------------------------------------------
    # row 1: panels a, b, c share one section window; row 2: panel d, full width
    left_pad = 0.42
    right_pad = 0.06
    col_gap = 0.16  # depth ticks are labelled on panel a only
    top_pad = 0.06

    w_sec = (FIG_W_IN - left_pad - right_pad - 2 * col_gap) / 3.0
    h_sec = w_sec * (SEC_ZLIM[1] - SEC_ZLIM[0]) / (SEC_XLIM[1] - SEC_XLIM[0])

    row_gap = 0.52  # x label of row 1 + breathing room
    w_d = FIG_W_IN - left_pad - right_pad
    h_d = 2.50

    xlabel_h = 0.34
    leg_h = 0.42
    bottom_pad = 0.04
    fig_h = top_pad + h_sec + row_gap + h_d + xlabel_h + leg_h + bottom_pad

    fig = plt.figure(figsize=(FIG_W_IN, fig_h))

    def rect(x0_in, y_top_in, w_in, h_in):
        """Axes rectangle from inches measured from the top-left of the canvas."""
        return [x0_in / FIG_W_IN, (fig_h - y_top_in - h_in) / fig_h, w_in / FIG_W_IN, h_in / fig_h]

    xs = [left_pad + k * (w_sec + col_gap) for k in range(3)]
    ax_a, ax_b, ax_c = (fig.add_axes(rect(x, top_pad, w_sec, h_sec)) for x in xs)
    ax_d = fig.add_axes(rect(left_pad, top_pad + h_sec + row_gap, w_d, h_d))
    leg_ax = fig.add_axes(
        [
            left_pad / FIG_W_IN,
            bottom_pad / fig_h,
            (FIG_W_IN - left_pad - right_pad) / FIG_W_IN,
            leg_h / fig_h,
        ]
    )

    # --- panels a, b, c ------------------------------------------------------
    for ax, step, letter in ((ax_a, STEP_A, "a"), (ax_b, STEP_B, "b"), (ax_c, STEP_C, "c")):
        print(f"panel {letter}: step {step}")
        d = fd.load_step(step)
        draw_composition(ax, d, SEC_XLIM, SEC_ZLIM, stride=STRIDE_DISPLAY)
        if letter != "a":
            draw_melt_contours(ax, d, SEC_XLIM, SEC_ZLIM)
        style_section_axis(
            ax,
            SEC_XLIM,
            SEC_ZLIM,
            SEC_XTICKS,
            SEC_ZTICKS,
            xlabel=True,
            ylabel=letter == "a",
        )
        if letter != "a":
            ax.set_yticklabels([])
        panel_label(ax, letter)
        time_label(ax, d["time_myr"] - TIME0_MODEL_MYR)
        add_wet_solidus_label(ax)
        if letter != "a":
            add_melt_legend(ax)
        print(f"    t = {d['time_myr']:.3f} Myr")
        del d

    # --- panel d -------------------------------------------------------------
    paths = load_paths()
    for name, sol in (
        ("pyr", fd.solidus_pyroxenite),
        ("upw", fd.solidus_wet),
    ):
        p = paths[name]
        over = p["T"] - sol(p["P"])
        sup = p["supra"]
        rng = (
            f"(P {p['P'][sup].min():.1f}-{p['P'][sup].max():.1f} GPa)" if sup.any() else ""
        )
        print(
            f"panel d {name}: marker {p['id']} n={len(p['T'])} "
            f"t={p['t'][0]:.2f}-{p['t'][-1]:.2f} Myr  "
            f"z={p['z'][0]:.0f}-{p['z'][-1]:.0f} km  "
            f"P={p['P'][0]:.1f}-{p['P'][-1]:.1f} GPa  "
            f"supra-solidus for {int(sup.sum())} steps, max +{over.max():.0f} C {rng}"
        )
    draw_pt_panel(ax_d, paths)
    panel_label(ax_d, "d", x=0.03, y=0.975)

    # --- legend --------------------------------------------------------------
    fig.canvas.draw()
    add_material_legend(leg_ax, colors)

    fig.savefig(OUT_PNG, dpi=DPI)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_JPG, dpi=DPI, pil_kwargs={"quality": 95})
    print(f"Saved {OUT_PNG.name}, {OUT_PDF.name} and {OUT_JPG.name}")
    print(
        f"Figure size: {FIG_W_IN * 25.4:.1f} × {fig_h * 25.4:.1f} mm "
        f"(Nature double column = {FIG_W_MM:.0f} mm)"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
