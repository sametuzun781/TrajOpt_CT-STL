import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
from matplotlib import gridspec

# ============================================================
# Global plotting style
# ============================================================
def set_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 300,        
        "savefig.dpi": 600,
        "font.size": 12,          
        "axes.labelsize": 15,     
        "legend.fontsize": 11,    
        "xtick.labelsize": 12,    
        "ytick.labelsize": 12,
        "figure.autolayout": False, 
        "axes.xmargin": 0,
        "lines.solid_capstyle": 'round',
        "lines.solid_joinstyle": 'round',
    })
    mpl.rc('font', **{'family':'serif','sans-serif':['Computer Modern Roman']}) 
    mpl.rc('text', usetex=True)
    mpl.rcParams.update({'text.latex.preamble': r"\usepackage{bm}"})

# ============================================================
# Helpers
# ============================================================
def compute_speed(x):
    return np.linalg.norm(x[:, 3:6], axis=1)

def compute_distance_to_station(x, p_w):
    return np.linalg.norm(x[:, 0:3] - p_w[None, :], axis=1)

def compute_station_margin(x, p_w, r_w):
    return r_w - compute_distance_to_station(x, p_w)

def first_entry_time_and_index(times, x, p_w, r_w):
    dist = compute_distance_to_station(x, p_w)
    idx = np.where(dist <= r_w)[0]
    if len(idx) == 0:
        return None, None
    return times[idx[0]], idx[0]

# KIRPILMAYI (CLIPPING) ÖNLEMEK İÇİN PAD VE ZOOM AYARLANDI
def set_tight_axes_equal(ax, xyz, p_w, r_w, pad=2.5):
    xmin = min(xyz[:, 0].min(), p_w[0] - r_w) - pad
    xmax = max(xyz[:, 0].max(), p_w[0] + r_w) + pad
    
    ymin = min(xyz[:, 1].min(), p_w[1] - r_w) - pad
    ymax = max(xyz[:, 1].max(), p_w[1] + r_w) + pad

    zmin = min(xyz[:, 2].min(), p_w[2] - r_w) - pad
    zmax = max(xyz[:, 2].max(), p_w[2] + r_w) + pad

    xmin = -8.2
    xmax = 8.2
    ymin = -3.0
    ymax = 1.0
    # zmin = -1.5
    # zmax = 1.5
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    # Pad 2.5 yapıldığı için objeler daha merkezde, zoom=1.35 beyaz boşluğu atar ama objeleri kesmez.
    try:
        ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin), zoom=1.35)
    except TypeError:
        ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        ax.dist = 8.0

def plot_sphere(ax, center, radius, color="#7B2CBF", alpha=0.18, n_u=60, n_v=40):
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True, shade=True, zorder=0)
    ax.plot_wireframe(x, y, z, rstride=6, cstride=6, color=color, linewidth=0.30, alpha=min(0.42, alpha + 0.08))

def make_colored_3d_trajectory_by_speed(ax, xyz, speed, cmap="rainbow", lw=3.0):
    points = xyz.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=np.min(speed), vmax=np.max(speed))
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(speed[:-1])
    lc.set_linewidth(lw)
    ax.add_collection3d(lc)
    return lc, norm

# ============================================================
# Subplot 1: 3D Trajectory
# ============================================================
def plot_trajectory_3d_speed(ax, results, params, cmap="rainbow"):
    x_all = results["x_all"]
    x_nodes = results["x_nmpc_all"][0]
    p_w = np.asarray(params["p_w"])
    r_w = float(params["r_w"])

    xyz = x_all[:, 0:3]
    xyz_nodes = x_nodes[:, 0:3]
    speed = compute_speed(x_all)
    
    display_radius = max(1.0 * r_w, 0.01)

    # Station
    plot_sphere(ax, p_w, display_radius, color="#7B2CBF", alpha=0.09)
    plot_sphere(ax, p_w, 0.70 * display_radius, color="#9D4EDD", alpha=0.18)
    ax.scatter(p_w[0], p_w[1], p_w[2], s=90, c="#5A189A", marker="*", edgecolors="white", linewidths=0.7, zorder=8)

    # Trajectory
    lc, norm = make_colored_3d_trajectory_by_speed(ax, xyz, speed, cmap=cmap, lw=2.6)
    ax.scatter(xyz_nodes[:, 0], xyz_nodes[:, 1], xyz_nodes[:, 2], s=15, c="black", alpha=0.92, zorder=6)

    # Start and Goal
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], s=85, c="#2A9D8F", marker="o", edgecolors="black", linewidths=0.7, zorder=7)
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], s=100, c="#D62828", marker="X", edgecolors="black", linewidths=0.6, zorder=7)
    
    ax.set_xlabel(r"$x$ [m]", labelpad=15)
    ax.set_ylabel(r"$y$ [m]", labelpad=0)
    ax.set_zlabel(r"$z$ [m]", labelpad=-2)

    # ax.view_init(elev=25, azim=-60) 
    ax.view_init(elev=70, azim=-90) 
    
    # Pad=2.5 verileri güvenli bölgeye alır
    set_tight_axes_equal(ax, xyz, p_w, r_w, pad=2.5)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_alpha(0.04)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#2A9D8F", markeredgecolor='black', markersize=8, label='Start'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor="#D62828", markeredgecolor='black', markersize=8, label='Goal'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor="#5A189A", markeredgecolor='white', markersize=10, label='Charging Station'),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(-0.15, 0.80), frameon=True, framealpha=0.95)

    return norm, cmap

# ============================================================
# Subplot 2: Speed Profile
# ============================================================
def plot_speed_profile(ax, results, params):
    t = results["times_all"]
    t_nodes = results["times_nodes"]
    x_all = results["x_all"]
    x_nodes = results["x_nmpc_all"][0]
    p_w = np.asarray(params["p_w"])
    r_w = float(params["r_w"])
    
    v_limit_before = 2.0
    v_limit_after = 6.0

    speed = compute_speed(x_all)
    speed_nodes = compute_speed(x_nodes)
    t_hit, _ = first_entry_time_and_index(t, x_all, p_w, r_w)

    v_limit = np.ones_like(t) * v_limit_after
    if t_hit is not None:
        v_limit[t <= t_hit] = v_limit_before
    else:
        v_limit[:] = v_limit_before 

    ax.fill_between(t, 0, v_limit, step="pre", color="#457B9D", alpha=0.15, lw=0, label="Allowed Region")
    ax.step(t, v_limit, where="pre", color="#1D3557", linestyle="--", lw=2.0, label=r"$v_{\max}(t)$")
    ax.plot(t, speed, color="#C1121F", lw=2.5, label=r"$\|v(t)\|$")
    ax.scatter(t_nodes, speed_nodes, s=24, color="black", zorder=5)

    if t_hit is not None:
        ax.axvline(t_hit, color="0.35", linestyle=(0, (4, 3)), lw=1.5, label=r"$t_{hit}$")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Speed [m/s]")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0.0, 1.1 * max(np.max(speed), v_limit_after))
    ax.grid(True, which="major", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

# ============================================================
# Subplot 3: Station Margin
# ============================================================
def plot_station_margin(ax, results, params):
    t = results["times_all"]
    t_nodes = results["times_nodes"]
    x_all = results["x_all"]
    x_nodes = results["x_nmpc_all"][0]
    p_w = np.asarray(params["p_w"])
    r_w = float(params["r_w"])

    margin = compute_station_margin(x_all, p_w, r_w)
    margin_nodes = compute_station_margin(x_nodes, p_w, r_w)
    t_hit, _ = first_entry_time_and_index(t, x_all, p_w, r_w)

    ax.fill_between(t, 0, max(0.02, 1.05 * np.max(np.maximum(margin, 0.0))), color="#7B2CBF", alpha=0.15, lw=0, label="Inside Station")
    ax.plot(t, margin, color="#6A4C93", lw=2.5, label=r"$g_{c}(t)$")
    ax.scatter(t_nodes, margin_nodes, s=24, color="black", zorder=5)
    ax.axhline(0.0, color="black", lw=1.4, linestyle=(0, (6, 3)))

    if t_hit is not None:
        ax.axvline(t_hit, color="0.35", linestyle=(0, (4, 3)), lw=1.5, label=r"$t_{hit}$")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Margin [m]")
    ax.set_xlim(t[0], t[-1])

    ymin = 1.08 * min(np.min(margin), -0.05)
    ymax = 1.08 * max(np.max(margin), 0.05)
    ax.set_ylim(ymin, ymax)

    ax.grid(True, which="major", alpha=0.4)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)

# ============================================================
# Master Unified Plot Function
# ============================================================
def plot_until_unified_figure(results, params, savepath=None):
    set_paper_style()
    
    fig = plt.figure(figsize=(11, 6.5)) 
    
    # Kırpılmayı önlemek için güvenli absolute positioning (sol 0.0, sağ 0.88)
    ax3d = fig.add_axes([0.0, 0.40, 0.86, 0.65], projection="3d") 
    
    # Colorbar 3B grafiğin sağında, tuvalin sınırları içinde (0.87)
    cax = fig.add_axes([0.78, 0.58, 0.015, 0.30])
    
    ax_speed = fig.add_axes([0.06, 0.08, 0.38, 0.28])
    ax_margin = fig.add_axes([0.55, 0.08, 0.38, 0.28])
    
    norm, cmap = plot_trajectory_3d_speed(ax3d, results, params)
    plot_speed_profile(ax_speed, results, params)
    plot_station_margin(ax_margin, results, params)
    
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
    cbar.set_label(r"Speed $\|v(t)\|$ [m/s]")

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ============================================================
# Standalone Figure Exporter
# ============================================================
def save_individual_until_figures(results, params, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    set_paper_style()

    # 1) 3D Trajectory (Standalone)
    fig1 = plt.figure(figsize=(9.0, 4.5))
    
    # Tuvalin dışına taşmaması için güvenli limitler
    ax1 = fig1.add_axes([0.0, 0.0, 0.85, 1.0], projection="3d") 
    norm, cmap = plot_trajectory_3d_speed(ax1, results, params)
    
    cax = fig1.add_axes([0.84, 0.28, 0.015, 0.45])
    cbar = fig1.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(r"Speed $\|v(t)\|$ [m/s]")
    
    fig1.savefig(os.path.join(save_dir, "until_3d_traj.png"), bbox_inches="tight")
    fig1.savefig(os.path.join(save_dir, "until_3d_traj.pdf"), bbox_inches="tight")
    plt.close(fig1)

    # 2) Speed Profile (Standalone)
    fig2 = plt.figure(figsize=(6.5, 2.8))
    ax2 = fig2.add_subplot(111)
    plot_speed_profile(ax2, results, params)
    fig2.savefig(os.path.join(save_dir, "until_speed_profile.png"), bbox_inches="tight")
    fig2.savefig(os.path.join(save_dir, "until_speed_profile.pdf"), bbox_inches="tight")
    plt.close(fig2)

    # 3) Station Margin (Standalone)
    fig3 = plt.figure(figsize=(6.5, 2.8))
    ax3 = fig3.add_subplot(111)
    plot_station_margin(ax3, results, params)
    fig3.savefig(os.path.join(save_dir, "until_station_margin.png"), bbox_inches="tight")
    fig3.savefig(os.path.join(save_dir, "until_station_margin.pdf"), bbox_inches="tight")
    plt.close(fig3)
