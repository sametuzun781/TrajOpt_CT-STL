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
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.labelsize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    mpl.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
    mpl.rc('text', usetex=True)
    mpl.rcParams.update({'text.latex.preamble': r"\usepackage{bm}"})


# ============================================================
# Helpers
# ============================================================
def compute_speed(x):
    return np.linalg.norm(x[:, 3:6], axis=1)

def compute_distance(x, p_w):
    return np.linalg.norm(x[:, 0:3] - np.asarray(p_w)[None, 0:3], axis=1)

def first_entry_index(x, p_w, r_w):
    dist = compute_distance(x, p_w)
    idx = np.where(dist <= r_w)[0]
    return int(idx[0]) if len(idx) > 0 else None

def plot_sphere(ax, center, radius, color="#7B2CBF", alpha=0.18, n_u=60, n_v=40):
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    ax.plot_surface(
        x, y, z, rstride=1, cstride=1,
        color=color, alpha=alpha, linewidth=0,
        antialiased=True, shade=True, zorder=0
    )
    ax.plot_wireframe(
        x, y, z, rstride=6, cstride=6,
        color=color, linewidth=0.30, alpha=min(0.42, alpha + 0.08)
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


# ============================================================
# 1) 3D Trajectory for EVENTUALLY
# ============================================================
def plot_eventually_trajectory_3d(ax, results, params, waypoints, cmap="rainbow"):
    x_all = np.asarray(results["x_all"])
    x_nodes = np.asarray(results["x_nmpc_all"][0])
    xyz = x_all[:, 0:3]
    xyz_nodes = x_nodes[:, 0:3]
    speed = compute_speed(x_all)

    # Plot the colored trajectory and nodes
    lc, norm = make_colored_3d_trajectory_by_speed(ax, xyz, speed, cmap=cmap, lw=2.4)
    ax.scatter(xyz_nodes[:, 0], xyz_nodes[:, 1], xyz_nodes[:, 2], s=12, c="black", alpha=0.92, zorder=6)

    # Plot start and goal
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], s=85, c="#1D3557", marker="o", edgecolors="black", zorder=8)
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], s=100, c="#D62828", marker="X", edgecolors="black", zorder=8)

    # Plot the 3 waypoints and mark the exact satisfaction entries
    for wp in waypoints:
        p_w = wp["pos"]
        r_w = wp["r"]
        color = wp["color"]
        
        # Draw the sphere
        display_radius = max(1.0 * r_w, 0.8)
        plot_sphere(ax, p_w, display_radius, color=color, alpha=0.09)
        plot_sphere(ax, p_w, 0.70 * display_radius, color=color, alpha=0.18)
        ax.scatter(p_w[0], p_w[1], p_w[2], s=90, c=color, marker="*", edgecolors="white", linewidths=0.7, zorder=8)

    ax.set_xlabel(r"$x$ [m]", labelpad=12)
    ax.set_ylabel(r"$y$ [m]", labelpad=8)
    ax.set_zlabel(r"$z$ [m]", labelpad=6)

    # From your code
    # ax.view_init(elev=90, azim=270)
    # ax.view_init(elev=90-45, azim=270+30-15)
    ax.view_init(elev=50, azim=280) # Adjusted angle for a wide trajectory
    ax.grid(True)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_alpha(0.04)

    set_axes_equal_3d(ax)

    # Custom legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor="#1D3557", markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Goal', markerfacecolor="#D62828", markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Target Waypoints', markerfacecolor="gray", markeredgecolor='white', markersize=10),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.95)
    
    return lc, norm


# ============================================================
# 2) 2D Distances Plot for EVENTUALLY
# ============================================================
def plot_eventually_distances(ax, results, params, waypoints):
    t = np.asarray(results["times_all"])
    t_nodes = np.asarray(results["times_nodes"])
    x_all = np.asarray(results["x_all"])
    x_nodes = np.asarray(results["x_nmpc_all"][0])

    r_w = float(params["r_w"])

    # Shade the satisfaction threshold region
    ax.axhspan(-0.05, r_w, color="#2A9D8F", alpha=0.15, lw=0, label="Satisfaction Zone ($d \leq r_w$)")

    max_dist = 0.0 # Track the maximum distance to dynamically scale the y-axis tightly

    for i, wp in enumerate(waypoints):
        p_w = wp["pos"]
        color = wp["color"]
        name = wp["name"]
        
        # Calculate continuous distance and discrete node distances
        dist = compute_distance(x_all, p_w)
        dist_nodes = compute_distance(x_nodes, p_w)
        
        # Update dynamic maximum distance
        max_dist = max(max_dist, np.max(dist))

        # Plot continuous line
        ax.plot(t, dist, color=color, lw=2.5, label=fr"$\|r(t) - p_{{w,{i+1}}}\|$ ({name})")
        
        # Plot NMPC nodes
        ax.scatter(t_nodes, dist_nodes, s=20, color=color, edgecolors="black", linewidths=0.5, zorder=5)

        # Mark satisfaction entry time
        t_hit_idx = first_entry_index(x_all, p_w, r_w)
        if t_hit_idx is not None:
            t_hit = t[t_hit_idx]
            ax.axvline(t_hit, color=color, linestyle=(0, (4, 3)), lw=1.2, alpha=0.7)

    # Threshold line
    ax.axhline(r_w, color="black", lw=1.5, linestyle="--", label=r"Threshold ($r_w$)")
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.5)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Distance to Waypoint [m]")
    ax.set_xlim(t[0], t[-1])
    
    # DYNAMIC Y-LIMIT: Tightly fits exactly 5% above the maximum distance calculated
    ax.set_ylim(-0.05, max_dist * 1.05)
    
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)


# ============================================================
# Combined Paper Figure for EVENTUALLY
# ============================================================
def plot_eventually_demo_figure(results, params, savepath=None):
    set_paper_style()

    waypoints = [
        {"pos": params["p_w"],   "r": params["r_w"], "color": "#7B2CBF", "name": "WP 1"},
        {"pos": params["p_w_2"], "r": params["r_w"], "color": "#2A9D8F", "name": "WP 2"},
        {"pos": params["p_w_3"], "r": params["r_w"], "color": "#F4A261", "name": "WP 3"}
    ]

    fig = plt.figure(figsize=(12.8, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1.0], wspace=0.28)

    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax_dist = fig.add_subplot(gs[1])

    # Plot 3D trajectory
    lc, norm = plot_eventually_trajectory_3d(ax3d, results, params, waypoints)
    
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="rainbow"), ax=ax3d, fraction=0.035, pad=0.08)
    cbar.set_label(r"Speed $\|v(t)\|$ [m/s]")

    # Plot 2D Distances
    plot_eventually_distances(ax_dist, results, params, waypoints)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ============================================================
# Standalone Figure Exporter
# ============================================================
def save_individual_eventually_figures(results, params, save_dir):
    """
    Saves the 3D trajectory and the Waypoint Distances plot 
    as two separate, high-quality files.
    """
    os.makedirs(save_dir, exist_ok=True)
    set_paper_style()

    waypoints = [
        {"pos": params["p_w"],   "r": params["r_w"], "color": "#7B2CBF", "name": "WP 1"},
        {"pos": params["p_w_2"], "r": params["r_w"], "color": "#2A9D8F", "name": "WP 2"},
        {"pos": params["p_w_3"], "r": params["r_w"], "color": "#F4A261", "name": "WP 3"}
    ]

    # 1) 3D Trajectory (Standalone)
    fig1 = plt.figure(figsize=(7.5, 6.0))
    ax1 = fig1.add_subplot(111, projection="3d")
    
    lc, norm = plot_eventually_trajectory_3d(ax1, results, params, waypoints)
    
    cbar = fig1.colorbar(ScalarMappable(norm=norm, cmap="rainbow"), ax=ax1, fraction=0.035, pad=0.01)
    cbar.set_label(r"Speed $\|v(t)\|$ [m/s]")
    
    fig1.savefig(os.path.join(save_dir, "eventually_3d_traj.png"), bbox_inches="tight")
    fig1.savefig(os.path.join(save_dir, "eventually_3d_traj.pdf"), bbox_inches="tight")
    plt.close(fig1)

    # 2) Distances Plot (Standalone)
    fig2 = plt.figure(figsize=(6.5, 4.5))
    ax2 = fig2.add_subplot(111)
    
    plot_eventually_distances(ax2, results, params, waypoints)
    
    fig2.savefig(os.path.join(save_dir, "eventually_distances.png"), bbox_inches="tight")
    fig2.savefig(os.path.join(save_dir, "eventually_distances.pdf"), bbox_inches="tight")
    plt.close(fig2)
    
    print(f"Saved individual 'Eventually' figures to: {save_dir}")
