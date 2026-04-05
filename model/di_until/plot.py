import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D

# ============================================================
# Global plotting style
# ============================================================
def set_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.size": 12,          # base
        "axes.labelsize": 16,     # x,y,z labels
        "legend.fontsize": 13,    # legend text
        "xtick.labelsize": 14,    # axis tick labels
        "ytick.labelsize": 14,
    })
    mpl.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']}) ## for Palatino and other serif fonts use: #rc('font',**{'family':'serif','serif':['Palatino']})
    mpl.rc('text', usetex=True)
    mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'axes.xmargin': 0})
    mpl.rcParams.update({'lines.solid_capstyle': 'round'})
    mpl.rcParams.update({'lines.solid_joinstyle': 'round'})
    mpl.rcParams.update({'lines.dash_capstyle': 'round'})
    mpl.rcParams.update({'lines.dash_joinstyle': 'round'})
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

def first_entry_time(times, x, p_w, r_w):
    dist = compute_distance_to_station(x, p_w)
    idx = np.where(dist <= r_w)[0]
    return None if len(idx) == 0 else times[idx[0]]

def set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


def plot_sphere(ax, center, radius, color="#7B2CBF", alpha=0.18, n_u=60, n_v=40):
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    ax.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        color=color,
        alpha=alpha,
        linewidth=0,
        antialiased=True,
        shade=True,
        zorder=0
    )
    ax.plot_wireframe(
        x, y, z,
        rstride=6, cstride=6,
        color=color,
        linewidth=0.30,
        alpha=min(0.42, alpha + 0.08)
    )


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
# 1) 3D trajectory colored by speed
# ============================================================
def plot_trajectory_3d_speed(results, params, save_dir=None,
                             station_scale_for_display=6.0,
                             min_display_radius=0.8,
                             cmap="rainbow"):

    os.makedirs(save_dir, exist_ok=True)
    set_paper_style()

    x_all = results["x_all"]
    x_nodes = results["x_nmpc_all"][0]
    p_w = np.asarray(params["p_w"])
    r_w = float(params["r_w"])

    xyz = x_all[:, 0:3]
    xyz_nodes = x_nodes[:, 0:3]
    speed = compute_speed(x_all)

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    # speed-colored trajectory
    lc, norm = make_colored_3d_trajectory_by_speed(ax, xyz, speed, cmap=cmap, lw=2.4)

    # traj_extent = np.max(np.ptp(xyz, axis=0))
    display_radius = max(station_scale_for_display * r_w, min_display_radius)

    # charging station enlarged for visibility
    plot_sphere(ax, p_w, display_radius, color="#7B2CBF", alpha=0.09)
    plot_sphere(ax, p_w, 0.70 * display_radius, color="#9D4EDD", alpha=0.18)

    # exact center marker
    ax.scatter(
        p_w[0], p_w[1], p_w[2],
        s=90, c="#5A189A", marker="*",
        edgecolors="white", linewidths=0.7,
        depthshade=False, zorder=8
    )

    # nodes
    ax.scatter(
        xyz_nodes[:, 0], xyz_nodes[:, 1], xyz_nodes[:, 2],
        s=15, c="black", alpha=0.92, depthshade=False, zorder=6
    )

    # start / goal
    ax.scatter(
        xyz[0, 0], xyz[0, 1], xyz[0, 2],
        s=85, c="#2A9D8F", marker="o",
        edgecolors="black", linewidths=0.7, depthshade=False, zorder=7
    )
    ax.scatter(
        xyz[-1, 0], xyz[-1, 1], xyz[-1, 2],
        s=100, c="#D62828", marker="X",
        edgecolors="black", linewidths=0.6, depthshade=False, zorder=7
    )

    ax.set_xlabel(r"$x$ [m]", labelpad=5)
    ax.set_ylabel(r"$y$ [m]", labelpad=0)
    ax.set_zlabel(r"$z$ [m]", labelpad=0)

    ax.view_init(elev=45, azim=270)
    ax.grid(False)
    ax.set_box_aspect((1.25, 1.25, 0.1), zoom=1.30)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_alpha(0.04)

    all_xyz = np.vstack([xyz, xyz_nodes, p_w[None, :]])
    mins = np.min(all_xyz, axis=0)
    maxs = np.max(all_xyz, axis=0)
    mins[0] = -12.5
    mins[1] = -12.5
    mins[2] = -1

    maxs[0] = 12.5
    maxs[1] = 12.5
    maxs[2] = 1

    pad = 0.05 * max(np.max(maxs - mins), 1.0)

    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)

    set_axes_equal_3d(ax)
    ax.set_zlim(mins[2], maxs[2])
    ax.set_zticks([mins[2], 0, maxs[2]])

    # --- 2D grid on xy-plane (z = 0), spacing = 2.5 m ---
    grid_spacing = 2.5
    z_grid = 0.0

    x_lines = np.arange(np.floor(mins[0] / grid_spacing) * grid_spacing,
                        np.ceil(maxs[0] / grid_spacing) * grid_spacing + grid_spacing,
                        grid_spacing)
    y_lines = np.arange(np.floor(mins[1] / grid_spacing) * grid_spacing,
                        np.ceil(maxs[1] / grid_spacing) * grid_spacing + grid_spacing,
                        grid_spacing)

    for x in x_lines:
        ax.plot([x, x], [mins[0], maxs[0]], [z_grid, z_grid],
                color='0.8', linewidth=0.6, zorder=0)

    for y in y_lines:
        ax.plot([mins[1], maxs[1]], [y, y], [z_grid, z_grid],
                color='0.8', linewidth=0.6, zorder=0)

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, fraction=0.032, pad=0.125
    )
    cbar.set_label(r"Speed $\|v(t)\|$ [m/s]")

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Start',
               markerfacecolor="#2A9D8F", markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Goal',
               markerfacecolor="#D62828", markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Charging station',
               markerfacecolor="#5A189A", markeredgecolor='white', markersize=10),
        Line2D([0], [0], marker='o', color='black', label='Nodes',
               markerfacecolor='black', markersize=4, linewidth=0),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.95),
        frameon=True,
        framealpha=0.95
    )

    plt.subplots_adjust(left=0.00, right=0.92, bottom=0.02, top=0.98)
    plt.savefig(os.path.join(save_dir, "until_traj.png"), bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "until_traj.pdf"), bbox_inches="tight")
    plt.show()


# ============================================================
# 2) Speed profile with constant color
# ============================================================
def plot_speed_profile(results, params, save_dir=None):
    set_paper_style()

    t = results["times_all"]
    t_nodes = results["times_nodes"]
    x_all = results["x_all"]
    x_nodes = results["x_nmpc_all"][0]
    p_w = np.asarray(params["p_w"])
    r_w = float(params["r_w"])
    v_safe = float(params["spd_lim"])

    speed = compute_speed(x_all)
    speed_nodes = compute_speed(x_nodes)
    t_hit = first_entry_time(t, x_all, p_w, r_w)

    fig, ax = plt.subplots(figsize=(7.3, 3.7))

    ax.axhspan(0.0, v_safe, color="#457B9D", alpha=0.10, lw=0)

    ax.plot(
        t, speed,
        color="#C1121F",
        lw=2.5,
        label=r"$\|v(t)\|$"
    )

    ax.scatter(
        t_nodes, speed_nodes,
        s=24, color="black", zorder=5,
        label="Nodes"
    )

    ax.axhline(
        y=v_safe,
        color="#1D3557",
        linestyle=(0, (6, 3)),
        lw=1.7,
        label=r"$v_{\mathrm{safe}}$"
    )

    if t_hit is not None:
        ax.axvline(
            t_hit,
            color="0.35",
            linestyle=(0, (4, 3)),
            lw=1.5,
            label="Station entry"
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Speed [m/s]")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0.0, 1.08 * max(np.max(speed), v_safe))
    ax.grid(True, which="major")

    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "until_speed.png"), bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "until_speed.pdf"), bbox_inches="tight")
    plt.show()


# ============================================================
# 3) Charging-station margin plot
# ============================================================
def plot_station_margin(results, params, save_dir=None):
    set_paper_style()

    t = results["times_all"]
    t_nodes = results["times_nodes"]
    x_all = results["x_all"]
    x_nodes = results["x_nmpc_all"][0]
    p_w = np.asarray(params["p_w"])
    r_w = float(params["r_w"])

    margin = compute_station_margin(x_all, p_w, r_w)
    margin_nodes = compute_station_margin(x_nodes, p_w, r_w)
    t_hit = first_entry_time(t, x_all, p_w, r_w)

    fig, ax = plt.subplots(figsize=(7.3, 3.7))

    ax.axhspan(0.0, max(0.02, 1.05 * np.max(np.maximum(margin, 0.0))),
               color="#7B2CBF", alpha=0.10, lw=0)

    ax.plot(
        t, margin,
        color="#6A4C93",
        lw=2.5,
        label=r"$g_{c}(t)=r_w-\|r(t)-p_w\|$"
    )

    ax.scatter(
        t_nodes, margin_nodes,
        s=24, color="black", zorder=5,
        label="Nodes"
    )

    ax.axhline(
        0.0,
        color="black",
        lw=1.4,
        linestyle=(0, (6, 3)),
        label=r"$g_{c}(t)=0$"
    )

    if t_hit is not None:
        ax.axvline(
            t_hit,
            color="0.35",
            linestyle=(0, (4, 3)),
            lw=1.5,
            label="Station entry"
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Charging-station margin [m]")
    ax.set_xlim(t[0], t[-1])

    ymin = 1.08 * min(np.min(margin), -0.05)
    ymax = 1.08 * max(np.max(margin), 0.05)
    ax.set_ylim(ymin, ymax)

    ax.grid(True, which="major")
    ax.legend(loc="best", frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "until_margin.png"), bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "until_margin.pdf"), bbox_inches="tight")
    plt.show()
