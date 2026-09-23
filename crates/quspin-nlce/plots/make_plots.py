"""Plot NLCE partial sums from export_csv output against exact references."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
# Ordinal blue ramp, steps 250/350/450/550/650 (validated: monotone, gaps >= 0.06).
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def load(name):
    data = defaultdict(lambda: defaultdict(list))
    with open(DIR / name) as f:
        for row in csv.DictReader(f):
            o = int(row["order"])
            for k, v in row.items():
                if k != "order":
                    data[o][k].append(float(v))
    return {o: {k: np.array(v) for k, v in d.items()} for o, d in data.items()}


def free_fermions(t):
    k = 2 * np.pi * (np.arange(4096) + 0.5) / 4096
    eps = np.cos(k)[None, :]
    b = 1.0 / np.asarray(t)[:, None]
    f = 1.0 / (np.exp(b * eps) + 1.0)
    e = (eps * f).mean(axis=1)
    lnz = np.log1p(np.exp(-b * eps)).mean(axis=1)
    return e, lnz + b[:, 0] * e


def onsager(t):
    j = 0.25
    k2 = 2 * j / np.asarray(t)
    kap = 2 * np.sinh(k2) / np.cosh(k2) ** 2
    a, g = np.ones_like(kap), np.sqrt(1 - kap**2)
    for _ in range(40):
        a, g = 0.5 * (a + g), np.sqrt(a * g)
    ell = np.pi / (2 * a)
    return -j / np.tanh(k2) * (1 + 2 / np.pi * (2 * np.tanh(k2) ** 2 - 1) * ell)


def order_lines(ax, data, orders, key, transform=lambda o, y: y):
    for color, o in zip(RAMP, orders):
        ax.plot(
            data[o]["T"], transform(o, data[o][key]), color=color, label=f"order {o}"
        )


def reference(ax, t, y, label):
    ax.plot(
        t, y, color=INK, linewidth=1.5, linestyle=(0, (4, 3)), label=label, zorder=5
    )


def finish(fig, path, title, subtitle):
    # Positions in inches from the top, so every figure height gets the same
    # spacing; fig.text (not suptitle) so tight_layout packs the axes below.
    h = fig.get_figheight()
    n_lines = subtitle.count("\n") + 1
    fig.text(
        0.01,
        1 - 0.1 / h,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01, 1 - 0.42 / h, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.42 + 0.19 * n_lines) / h))
    fig.savefig(path, dpi=160)
    print("wrote", path)


def floor(y):
    return np.maximum(np.abs(y), 1e-16)


# --- Heisenberg -------------------------------------------------------------
h = load("heisenberg.csv")
orders = [4, 5, 6, 7, 8]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, key, lab in zip(
    axes, ["energy", "entropy", "specific_heat"], ["E / N", "S / N", "C / N"]
):
    order_lines(ax, h, orders, key)
    ax.set_xscale("log")
    ax.set_xlabel("T / J")
    ax.set_title(lab, loc="left")
t = h[8]["T"]
hi = t >= 1.0
b = 1 / t[hi]
reference(axes[0], t[hi], -3 * b / 8 - 3 * b**2 / 32, "high-T series, O(β²)")
reference(axes[1], t, np.full_like(t, np.log(2)), "ln 2")
axes[0].set_ylim(-0.72, 0.02)
axes[1].set_ylim(0, 0.75)
axes[2].set_ylim(0, 0.62)
axes[2].legend(loc="upper right")
axes[0].legend(handles=axes[0].lines[-1:], loc="lower right")
axes[1].legend(handles=axes[1].lines[-1:], loc="lower right")
finish(
    fig,
    DIR / "heisenberg.png",
    "Square-lattice Heisenberg AFM: bare rectangle-NLCE partial sums",
    "Orders m + n = 4…8 (largest cluster 4×4). Successive orders agree for T ≳ 1 J "
    "and alternate below it, where resummation is needed.",
)

# --- Ising ------------------------------------------------------------------
g = load("ising.csv")
tc = 0.5 / np.log(1 + np.sqrt(2))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
order_lines(axes[0], g, orders, "energy")
t = g[8]["T"]
reference(axes[0], t, onsager(t), "Onsager (exact)")
axes[0].set_ylim(-0.52, 0.02)
axes[0].set_title("E / N", loc="left")
order_lines(axes[1], g, orders, "energy", lambda o, y: floor(y - onsager(g[o]["T"])))
axes[1].set_yscale("log")
axes[1].set_ylim(1e-16, 1)
axes[1].set_title("|E_NLCE − E_Onsager| / N", loc="left")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("T / J")
    ax.axvline(tc, color=INK_2, linewidth=1)
    ax.text(
        tc * 1.04,
        0.97,
        "T_c",
        transform=ax.get_xaxis_transform(),
        va="top",
        fontsize=9,
        color=INK_2,
    )
axes[0].legend(loc="lower right")
finish(
    fig,
    DIR / "ising.png",
    "2D classical Ising model (J = −1, S = ½): convergence to Onsager",
    "Above T_c ≈ 0.567 the error falls steeply with every order (to ~1e-13); "
    "convergence stalls near T_c and is slow in the ordered phase.",
)

# --- XX chain ---------------------------------------------------------------
x = load("xx_chain.csv")
xo = [3, 6, 9, 12, 15]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
order_lines(axes[0], x, xo, "energy")
t = x[15]["T"]
e_ff, s_ff = free_fermions(t)
reference(axes[0], t, e_ff, "free fermions (exact)")
axes[0].set_title("E / N", loc="left")
order_lines(
    axes[1], x, xo, "energy", lambda o, y: floor(y - free_fermions(x[o]["T"])[0])
)
axes[1].set_yscale("log")
axes[1].set_ylim(1e-16, 1)
axes[1].set_title("|E_NLCE − E_exact| / N", loc="left")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("T / J")
axes[0].legend(loc="lower right")
finish(
    fig,
    DIR / "xx_chain.png",
    "1D XX chain: 1 × n clusters vs the free-fermion thermodynamic limit",
    "Orders 3…15 (chains of 2…14 sites). At order 15 the error hits machine "
    "precision (~1e-14) for T ≳ 0.5 J.",
)

# --- Bond expansion vs rectangle expansion -----------------------------------
RECT, BOND = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated pair)
QUANTS = [("energy", "E / N"), ("entropy", "S / N"), ("specific_heat", "C / N")]

if (DIR / "heisenberg_bond.csv").exists():
    hb = load("heisenberg_bond.csv")

    # Partial sums by order, same layout as heisenberg.png.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, lab) in zip(axes, QUANTS):
        order_lines(ax, hb, [8, 9, 10, 11, 12], key)
        ax.set_xscale("log")
        ax.set_xlabel("T / J")
        ax.set_title(lab, loc="left")
    axes[0].set_ylim(-0.72, 0.02)
    axes[1].set_ylim(0, 0.75)
    axes[2].set_ylim(0, 0.62)
    axes[2].legend(loc="upper right")
    finish(
        fig,
        DIR / "heisenberg_bond.png",
        "Square-lattice Heisenberg AFM: bare bond-NLCE partial sums",
        "Orders 8…12 bonds (4,424 topologies, up to 13 sites). Even and odd orders "
        "bracket the result and fan out below T ≈ 1 J.",
    )

    # Highest order of each expansion, and its last-order change.
    rect_last, rect_prev = h[8], h[7]
    bond_last, bond_prev = hb[12], hb[11]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True)
    for col, (key, lab) in enumerate(QUANTS):
        top, bottom = axes[0, col], axes[1, col]
        top.plot(
            rect_last["T"], rect_last[key], color=RECT, label="rectangles, m + n ≤ 8"
        )
        top.plot(bond_last["T"], bond_last[key], color=BOND, label="bonds, ≤ 12")
        top.set_title(lab, loc="left")
        for last, prev, color in [
            (rect_last, rect_prev, RECT),
            (bond_last, bond_prev, BOND),
        ]:
            bottom.plot(last["T"], floor(last[key] - prev[key]), color=color)
        bottom.set_yscale("log")
        bottom.set_ylim(1e-15, 10)
        bottom.set_title(f"|Δ{lab.split()[0]}| between the last two orders", loc="left")
        bottom.set_xlabel("T / J")
        bottom.set_xscale("log")
    axes[0, 0].set_ylim(-0.72, 0.02)
    axes[0, 1].set_ylim(0, 0.75)
    axes[0, 2].set_ylim(0, 0.62)
    axes[0, 0].legend(loc="lower right")
    finish(
        fig,
        DIR / "bond_vs_rect_heisenberg.png",
        "Heisenberg AFM: rectangle vs topological bond expansion (bare sums)",
        "Rectangles: 16 clusters, ≤ 16 sites, ≈35 s. Bonds: 4,424 clusters, ≤ 13 "
        "sites, ≈3–5 min. Bottom: change between the last two orders.\n"
        "The flat bond floor (~1e-8) at high T is eigenvalue rounding amplified by "
        "inclusion–exclusion over 21M embeddings, not truncation error.",
    )

if (DIR / "ising_bond.csv").exists():
    gb = load("ising_bond.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for data, o, color, label in [
        (g, 8, RECT, "rectangles, m + n ≤ 8"),
        (gb, 12, BOND, "bonds, ≤ 12"),
    ]:
        t = data[o]["T"]
        axes[0].plot(t, data[o]["energy"], color=color, label=label)
        axes[1].plot(t, floor(data[o]["energy"] - onsager(t)), color=color)
    t = g[8]["T"]
    reference(axes[0], t, onsager(t), "Onsager (exact)")
    axes[0].set_ylim(-0.52, 0.02)
    axes[0].set_title("E / N", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-16, 1)
    axes[1].set_title("|E_NLCE − E_Onsager| / N", loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("T / J")
        ax.axvline(tc, color=INK_2, linewidth=1)
        ax.text(
            tc * 1.04,
            0.97,
            "T_c",
            transform=ax.get_xaxis_transform(),
            va="top",
            fontsize=9,
            color=INK_2,
        )
    axes[0].legend(loc="lower right")
    finish(
        fig,
        DIR / "bond_vs_rect_ising.png",
        "2D Ising model: rectangle vs topological bond expansion against Onsager",
        "Highest order of each expansion (bare sums). The exact solution gives the "
        "true error, not just the order-to-order change.\n"
        "The bond floor (~1e-9) at high T is rounding amplified by "
        "inclusion–exclusion over 21M embeddings, not truncation error.",
    )


# --- Resummation --------------------------------------------------------------
def load_resummed(name):
    """{(expansion, method): {key: array}} with rows sorted by T."""
    rows = defaultdict(lambda: defaultdict(list))
    with open(DIR / name) as f:
        for r in csv.DictReader(f):
            d = rows[(r["expansion"], r["method"])]
            for k in ("T", "energy", "entropy", "specific_heat"):
                d[k].append(float(r[k]))
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in rows.items()}


BARE_DASH = (0, (3, 2))

if (DIR / "heisenberg_resummed.csv").exists():
    rs = load_resummed("heisenberg_resummed.csv")
    families = {
        # Wynn only: Euler assumes an alternating tail and is biased where
        # the bond series has already converged (≈1e-5 at T = 2 for Euler 3).
        "rect": [m for m in rs if m[0] == "rect" and m[1].startswith("wynn")],
        "bond": [m for m in rs if m[0] == "bond" and m[1].startswith("wynn")],
    }
    rep = {"rect": ("rect", "wynn2"), "bond": ("bond", "wynn4")}
    style = {
        "rect": (RECT, "rectangles, m + n ≤ 8: Wynn 1–3"),
        "bond": (BOND, "bonds, ≤ 12: Wynn 2–5"),
    }
    t = rs[rep["rect"]]["T"]
    keep = (t >= 0.3) & (t <= 5)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True)
    for col, (key, lab) in enumerate(QUANTS):
        top, bottom = axes[0, col], axes[1, col]
        spreads = {}
        for fam, members in families.items():
            color, label = style[fam]
            vals = np.array([rs[m][key] for m in members])[:, keep]
            lo, hi = vals.min(axis=0), vals.max(axis=0)
            spreads[fam] = hi - lo
            top.fill_between(t[keep], lo, hi, color=color, alpha=0.15, linewidth=0)
            top.plot(t[keep], rs[rep[fam]][key][keep], color=color, label=label)
            bottom.plot(t[keep], floor(hi - lo), color=color, label=f"spread, {fam}")
        cross = rs[rep["bond"]][key][keep] - rs[rep["rect"]][key][keep]
        bottom.plot(
            t[keep],
            floor(cross),
            color=INK,
            linewidth=1.5,
            linestyle=(0, (4, 3)),
            label="|bond − rect| (representatives)",
        )
        top.set_title(lab, loc="left")
        bottom.set_title(f"{lab.split()[0]}: spread and disagreement", loc="left")
        bottom.set_yscale("log")
        bottom.set_ylim(1e-9, 1)
        bottom.set_xscale("log")
        bottom.set_xlabel("T / J")
        bottom.set_xticks([0.3, 0.5, 1, 2, 5], ["0.3", "0.5", "1", "2", "5"])
        bottom.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0, 0].set_ylim(-0.7, 0)
    axes[0, 1].set_ylim(0, 0.72)
    axes[0, 2].set_ylim(0, 0.55)
    axes[0, 0].legend(loc="lower right")
    axes[1, 0].legend(loc="upper right")
    finish(
        fig,
        DIR / "resummed_heisenberg.png",
        "Heisenberg AFM: resummed rectangle vs bond expansion",
        "Lines: Wynn 2 cycles (rectangles), Wynn 4 cycles (bonds); bands span all Wynn "
        "cycle counts (Euler agrees at low T but is biased at high T).\n"
        "Bottom: band width within each expansion and the gap between them, a "
        "convergence proxy (there is no exact result).",
    )

if (DIR / "ising_resummed.csv").exists():
    ri = load_resummed("ising_resummed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    t = ri[("rect", "wynn2")]["T"]
    near = (t >= 0.45) & (t <= 1.6)
    for fam, color, label in [
        ("rect", RECT, "rectangles, m + n ≤ 8"),
        ("bond", BOND, "bonds, ≤ 12 (even orders)"),
    ]:
        axes[0].plot(
            t[near], ri[(fam, "wynn2")]["energy"][near], color=color, label=label
        )
        for method, dash, suffix in [
            ("bare", BARE_DASH, "bare"),
            ("wynn2", "-", "Wynn 2"),
        ]:
            err = floor(ri[(fam, method)]["energy"] - onsager(t))
            axes[1].plot(t, err, color=color, linestyle=dash, label=f"{fam}, {suffix}")
    reference(axes[0], t[near], onsager(t[near]), "Onsager (exact)")
    axes[0].set_title("E / N, Wynn 2 cycles", loc="left")
    axes[1].set_title("|E − E_Onsager| / N", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-16, 1)
    axes[0].set_xticks([0.5, 0.6, 0.8, 1, 1.5], ["0.5", "0.6", "0.8", "1", "1.5"])
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("T / J")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.axvline(tc, color=INK_2, linewidth=1)
        ax.text(
            tc * 1.03,
            0.97,
            "T_c",
            transform=ax.get_xaxis_transform(),
            va="top",
            fontsize=9,
            color=INK_2,
        )
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="upper right", fontsize=8.5)
    finish(
        fig,
        DIR / "resummed_ising.png",
        "2D Ising model: Wynn resummation against Onsager",
        "Dashed: bare sums; solid: Wynn ε with 2 cycles (bond series thinned to even "
        "orders, since odd orders vanish).",
    )
