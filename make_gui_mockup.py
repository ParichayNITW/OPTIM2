"""Generate a GUI mockup image for the pipeline optimization app."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import numpy as np

fig = plt.figure(figsize=(20, 12), dpi=100, facecolor="#1E1E2E")

# ── colour palette ────────────────────────────────────────────────────────────
BG       = "#1E1E2E"
SIDEBAR  = "#2A2A3E"
PANEL    = "#2E2E44"
ACCENT   = "#4A90D9"
GREEN    = "#50C878"
ORANGE   = "#FF8C42"
RED      = "#E84393"
YELLOW   = "#FFD700"
WHITE    = "#FFFFFF"
LGRAY    = "#AAAACC"
DGRAY    = "#555577"
PIPE_COL = "#4A90D9"

# ── layout axes ──────────────────────────────────────────────────────────────
# Left sidebar (controls)
ax_side = fig.add_axes([0.0, 0.0, 0.16, 1.0])
ax_side.set_facecolor(SIDEBAR)
ax_side.set_xticks([]); ax_side.set_yticks([])
for sp in ax_side.spines.values(): sp.set_visible(False)

# Main canvas
ax = fig.add_axes([0.16, 0.22, 0.84, 0.78])
ax.set_facecolor(BG)
ax.set_xlim(0, 100); ax.set_ylim(0, 60)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_color(DGRAY)

# Bottom info bar
ax_bot = fig.add_axes([0.16, 0.0, 0.84, 0.22])
ax_bot.set_facecolor(PANEL)
ax_bot.set_xlim(0, 100); ax_bot.set_ylim(0, 1)
ax_bot.set_xticks([]); ax_bot.set_yticks([])
for sp in ax_bot.spines.values(): sp.set_color(DGRAY)

# ── SIDEBAR content ───────────────────────────────────────────────────────────
ax_side.text(0.5, 0.97, "Pipeline\nOptima", transform=ax_side.transAxes,
             ha="center", va="top", fontsize=13, fontweight="bold",
             color=ACCENT, family="monospace")

ax_side.plot([0.05, 0.95], [0.94, 0.94], color=DGRAY, lw=0.8,
             transform=ax_side.transAxes)

menu_items = [
    ("⌂  Dashboard",    0.89, ACCENT),
    ("▦  Pipeline Map", 0.84, WHITE),
    ("⚙  Stations",     0.79, WHITE),
    ("➤  Optimizer",    0.74, WHITE),
    ("▶  Scheduler",    0.69, WHITE),
    ("≡  Results",      0.64, WHITE),
    ("☁  Export",       0.59, WHITE),
]
for label, y, col in menu_items:
    if col == ACCENT:
        rect = FancyBboxPatch((0.04, y-0.022), 0.92, 0.038,
                              boxstyle="round,pad=0.005", linewidth=0,
                              facecolor=ACCENT+"33", transform=ax_side.transAxes)
        ax_side.add_patch(rect)
    ax_side.text(0.1, y, label, transform=ax_side.transAxes,
                 ha="left", va="center", fontsize=9,
                 color=col, family="monospace")

ax_side.text(0.5, 0.12, "Mode", transform=ax_side.transAxes,
             ha="center", va="center", fontsize=8, color=LGRAY)
for i, (lbl, col) in enumerate([("Edit","#FF8C42"),("View","#50C878"),("Run","#E84393")]):
    bx = 0.08 + i*0.3
    rb = FancyBboxPatch((bx, 0.06), 0.25, 0.05,
                        boxstyle="round,pad=0.005", linewidth=1,
                        edgecolor=col, facecolor=col+"33",
                        transform=ax_side.transAxes)
    ax_side.add_patch(rb)
    ax_side.text(bx+0.125, 0.085, lbl, transform=ax_side.transAxes,
                 ha="center", va="center", fontsize=7.5, color=col, fontweight="bold")

# ── TITLE BAR ────────────────────────────────────────────────────────────────
ax.text(50, 57.5, "Pipeline Network — Interactive Map", ha="center", va="center",
        fontsize=14, fontweight="bold", color=WHITE)
ax.text(50, 55.5, "Click any node to edit parameters  |  Drag to pan  |  Scroll to zoom",
        ha="center", va="center", fontsize=8.5, color=LGRAY, style="italic")

# ── PIPELINE GEOMETRY ─────────────────────────────────────────────────────────
# Main line: Origin → S1 → S2 → S3 → Terminal
main_x = [5,  22, 42, 62, 82, 95]
main_y = [30, 30, 32, 28, 31, 30]

# Draw pipe segments (thick line)
for i in range(len(main_x)-1):
    ax.plot([main_x[i], main_x[i+1]], [main_y[i], main_y[i+1]],
            color=PIPE_COL, lw=6, alpha=0.35, solid_capstyle="round", zorder=2)
    ax.plot([main_x[i], main_x[i+1]], [main_y[i], main_y[i+1]],
            color=PIPE_COL, lw=2, alpha=0.9, solid_capstyle="round", zorder=3)

# Branch from S2 → Branch Station → Branch Terminal
bx = [42, 50, 62]
by = [32, 44, 44]
for i in range(len(bx)-1):
    ax.plot([bx[i], bx[i+1]], [by[i], by[i+1]],
            color=ORANGE, lw=6, alpha=0.3, solid_capstyle="round", zorder=2)
    ax.plot([bx[i], bx[i+1]], [by[i], by[i+1]],
            color=ORANGE, lw=2, alpha=0.9, solid_capstyle="round", zorder=3,
            linestyle="--")

# ── STATION NODE DEFINITIONS ──────────────────────────────────────────────────
stations = [
    # (x, y, label, sublabel, color, has_pump, has_dra)
    (5,  30, "Origin",    "Q=850 m³/h",     GREEN,  False, False),
    (22, 30, "St. A",     "Pump+DRA\nSDH=420m", ACCENT, True,  True),
    (42, 32, "St. B",     "Pump+DRA\nSDH=380m", ACCENT, True,  True),
    (62, 28, "St. C",     "DRA Only\nSDH=—",    YELLOW, False, True),
    (82, 31, "St. D",     "Pump Only\nSDH=310m",ACCENT, True,  False),
    (95, 30, "Terminal",  "RH=25m",          RED,    False, False),
    # Branch
    (50, 44, "Br. St.",   "Pump+DRA\nSDH=200m", ORANGE, True, True),
    (62, 44, "Br. Term.", "RH=10m",          RED,    False, False),
]

def draw_station(ax, x, y, label, sublabel, color, has_pump, has_dra, selected=False):
    r = 2.2
    circle = Circle((x, y), r, facecolor=color+"22", edgecolor=color,
                    linewidth=2.5 if not selected else 4, zorder=5)
    ax.add_patch(circle)
    if selected:
        glow = Circle((x, y), r+0.6, facecolor="none", edgecolor=color,
                      linewidth=1, alpha=0.4, zorder=4)
        ax.add_patch(glow)
    ax.text(x, y+0.2, label, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=color, zorder=6)
    # Sub-icons for pump / DRA
    icons = []
    if has_pump: icons.append(("⚡", YELLOW))
    if has_dra:  icons.append(("DRA", GREEN))
    for ii, (ico, icol) in enumerate(icons):
        ox = x - 1.0 + ii * 2.2
        oy = y - r - 1.8
        ax.text(ox, oy, ico, ha="center", va="center", fontsize=7,
                color=icol, fontweight="bold", zorder=6)
    # Sublabel (above node)
    ax.text(x, y + r + 1.2, sublabel, ha="center", va="bottom",
            fontsize=7, color=LGRAY, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL, edgecolor=DGRAY, lw=0.7))

for i, (x, y, lbl, sub, col, pump, dra) in enumerate(stations):
    selected = (lbl == "St. B")   # show one node as "selected / being edited"
    draw_station(ax, x, y, lbl, sub, col, pump, dra, selected)

# Flow arrows along main pipe
arrow_xs = [(5+22)//2, (22+42)//2, (42+62)//2, (62+82)//2, (82+95)//2]
arrow_ys = [30, 31, 30, 29.5, 30.5]
for ax_pt, ay_pt in zip(arrow_xs, arrow_ys):
    ax.annotate("", xy=(ax_pt+2, ay_pt), xytext=(ax_pt-2, ay_pt),
                arrowprops=dict(arrowstyle="-|>", color=PIPE_COL, lw=1.5),
                zorder=7)
# Branch flow arrow
ax.annotate("", xy=(56, 44), xytext=(52, 44),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5), zorder=7)

# KP markers
for kp, xp, yp in [(0,5,22),(85,22,22),(210,42,22),(370,62,22),(510,82,22),(620,95,22)]:
    ax.text(xp, yp, f"KP {kp}", ha="center", va="center", fontsize=6.5,
            color=DGRAY, style="italic")

# Legend top-right
legend_x, legend_y = 76, 54
ax.text(legend_x, legend_y, "Legend", fontsize=8, color=WHITE, fontweight="bold")
for ii, (sym, col, desc) in enumerate([
    ("●", ACCENT, "Pump Station"),
    ("●", YELLOW, "DRA Only"),
    ("●", GREEN,  "Origin"),
    ("●", RED,    "Terminal"),
    ("▬", ORANGE, "Branch Line"),
]):
    ax.text(legend_x,     legend_y-1.8*(ii+1), sym,  fontsize=9, color=col)
    ax.text(legend_x+1.5, legend_y-1.8*(ii+1), desc, fontsize=7.5, color=LGRAY)

# ── SELECTED NODE POPUP (St. B) ───────────────────────────────────────────────
px, py = 55, 36
popup = FancyBboxPatch((px, py), 22, 17,
                       boxstyle="round,pad=0.3", linewidth=1.5,
                       edgecolor=ACCENT, facecolor="#1A1A30EE", zorder=10)
ax.add_patch(popup)
ax.text(px+11, py+16, "St. B  —  Edit Parameters", ha="center", va="top",
        fontsize=9, fontweight="bold", color=ACCENT, zorder=11)
ax.axhline(y=py+14.5, xmin=(px)/100, xmax=(px+22)/100, color=DGRAY, lw=0.7, zorder=11)

fields = [
    ("Elevation (m)",      "32.0"),
    ("Min Residual (m)",   "20.0"),
    ("Max DR (%)",         "70"),
    ("Pump Type A – avail","2"),
    ("Pump Type B – avail","1"),
    ("Min / Max Pumps",    "1  /  3"),
]
for fi, (fname, fval) in enumerate(fields):
    fy = py + 13.0 - fi*2.1
    ax.text(px+0.8, fy, fname, ha="left", va="center",
            fontsize=7.5, color=LGRAY, zorder=11)
    fbox = FancyBboxPatch((px+12, fy-0.7), 8.5, 1.4,
                          boxstyle="round,pad=0.15", linewidth=0.8,
                          edgecolor=ACCENT+"88", facecolor=PANEL, zorder=11)
    ax.add_patch(fbox)
    ax.text(px+16.25, fy, fval, ha="center", va="center",
            fontsize=8, color=WHITE, fontweight="bold", zorder=12)

# Save / Cancel buttons
for bx2, blbl, bcol in [(px+2, "Save", GREEN), (px+13, "Cancel", RED)]:
    btn = FancyBboxPatch((bx2, py+0.5), 7, 1.8,
                         boxstyle="round,pad=0.2", linewidth=1,
                         edgecolor=bcol, facecolor=bcol+"33", zorder=11)
    ax.add_patch(btn)
    ax.text(bx2+3.5, py+1.4, blbl, ha="center", va="center",
            fontsize=8, color=bcol, fontweight="bold", zorder=12)

# Dashed connector from St. B node to popup
ax.annotate("", xy=(px, py+8), xytext=(42+2.2, 32),
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1,
                            linestyle="dashed", alpha=0.6), zorder=9)

# ── BOTTOM INFO BAR ───────────────────────────────────────────────────────────
panels_bot = [
    (0,  25,  "Total Pipeline Length", "620 km",        ACCENT),
    (25, 50,  "Active Flow Rate",       "850 m³/h",      GREEN),
    (50, 75,  "Total Head Available",   "1,110 m",       YELLOW),
    (75, 100, "DRA Stations Active",    "3 / 4",         ORANGE),
]
for x0, x1, title, val, col in panels_bot:
    mid = (x0+x1)/2
    vline_x = x1/100
    ax_bot.axvline(x=vline_x, color=DGRAY, lw=0.8, ymin=0.1, ymax=0.9)
    ax_bot.text(mid, 0.72, title, ha="center", va="center",
                fontsize=8, color=LGRAY, transform=ax_bot.transAxes)
    ax_bot.text(mid, 0.38, val, ha="center", va="center",
                fontsize=13, color=col, fontweight="bold", transform=ax_bot.transAxes)

# Bottom toolbar
toolbar_items = ["+ Add Station", "✂ Remove", "↺ Undo", "↻ Redo",
                 "▶ Run Optimizer", "💾 Export"]
for ti, tlbl in enumerate(toolbar_items):
    tx = 0.03 + ti * 0.16
    col = ACCENT if "Run" in tlbl else (GREEN if "Export" in tlbl else LGRAY)
    bkg = ACCENT+"33" if "Run" in tlbl else (GREEN+"22" if "Export" in tlbl else DGRAY+"55")
    tb = FancyBboxPatch((tx, 0.05), 0.13, 0.28,
                        boxstyle="round,pad=0.02", linewidth=0.8,
                        edgecolor=col, facecolor=bkg,
                        transform=ax_bot.transAxes)
    ax_bot.add_patch(tb)
    ax_bot.text(tx+0.065, 0.19, tlbl, ha="center", va="center",
                fontsize=7.5, color=col, fontweight="bold",
                transform=ax_bot.transAxes)

fig.text(0.5, 0.005, "Pipeline Optima — GUI Interface Mockup  |  Interactive Streamlit Map View",
         ha="center", va="bottom", fontsize=8, color=DGRAY, style="italic")

plt.savefig("/home/user/OPTIM2/gui_mockup.png", dpi=100,
            facecolor=BG)
print("Saved gui_mockup.png")
