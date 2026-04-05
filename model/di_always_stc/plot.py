import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
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
# Math & Geometry Helpers
# ============================================================
def compute_speed(x):
    return np.linalg.norm(x[:, 3:6], axis=1)

def dist_to_rectangle(x_pts, y_pts, xmin, xmax, ymin, ymax):
    """Computes the exact Euclidean distance from points to a 2D rectangle."""
    dx = np.maximum(0, np.maximum(xmin - x_pts, x_pts - xmax))
    dy = np.maximum(0, np.maximum(ymin - y_pts, y_pts - ymax))
    return np.sqrt(dx**2 + dy**2)

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

def _box_faces(x1, x2, y1, y2, z1, z2):
    return [
        [[x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1]], # Bottom
        [[x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]], # Top
        [[x1, y1, z1], [x2, y1, z1], [x2, y1, z2], [x1, y1, z2]], # Front
        [[x1, y2, z1], [x2, y2, z1], [x2, y2, z2], [x1, y2, z2]], # Back
        [[x1, y1, z1], [x1, y2, z1], [x1, y2, z2], [x1, y1, z2]], # Left
        [[x2, y1, z1], [x2, y2, z1], [x2, y2, z2], [x2, y1, z2]], # Right
    ]

def add_box(ax, x1, x2, y1, y2, z1, z2, facecolor='red', edgecolor='darkred', alpha=0.2):
    poly = Poly3DCollection(_box_faces(x1, x2, y1, y2, z1, z2), 
                            facecolors=facecolor, edgecolors=edgecolor, 
                            linewidths=0.5, alpha=alpha)
    ax.add_collection3d(poly)

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
# 1) 3D Trajectory for ALWAYS
# ============================================================
def plot_always_trajectory_3d(ax, results, params, cmap="rainbow"):
    x_all = np.asarray(results["x_all"])
    x_nodes = np.asarray(results["x_nmpc_all"][0])

    xyz = x_all[:, 0:3]
    xyz_nodes = x_nodes[:, 0:3]
    speed = compute_speed(x_all)

    # Plot the colored trajectory and nodes
    lc, norm = make_colored_3d_trajectory_by_speed(ax, xyz, speed, cmap=cmap, lw=2.8)
    ax.scatter(xyz_nodes[:, 0], xyz_nodes[:, 1], xyz_nodes[:, 2], s=12, c="black", alpha=0.92, zorder=6)

    # Define Obstacles based on your specs
    # Obstacle 1: infinity to -2 in y axis
    obs1_x1, obs1_x2 = params['x_stc_1'], params['x_stc_2']
    obs1_y1, obs1_y2 = params['y_stc_1'], 7.000  # Extend to +15 (practical infinity for plot)
    
    # Obstacle 2: -infinity to 2 in y axis
    obs2_x1, obs2_x2 = params['x_stc_3'], params['x_stc_4']
    obs2_y1, obs2_y2 = -7.000, params['y_stc_2'] # Extend to -15 (practical infinity for plot)

    zmin = xyz[:, 2].min() - 1.0
    zmax = xyz[:, 2].max() + 1.0

    # Draw Obstacle 1 (Danger Zone)
    add_box(ax, obs1_x1, obs1_x2, obs1_y1, obs1_y2, zmin, zmax, facecolor='#D62828', edgecolor='black', alpha=0.15)
    
    # Draw Obstacle 2 (Danger Zone)
    add_box(ax, obs2_x1, obs2_x2, obs2_y1, obs2_y2, zmin, zmax, facecolor='#D62828', edgecolor='black', alpha=0.15)

    # Start and Goal markers
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], s=85, c="#1D3557", marker="o", edgecolors="black", zorder=8)
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], s=100, c="#D62828", marker="X", edgecolors="black", zorder=8)

    ax.set_xlabel(r"$x$ [m]", labelpad=12)
    ax.set_ylabel(r"$y$ [m]", labelpad=8)
    ax.set_zlabel(r"$z$ [m]", labelpad=6)
    
    ax.set_xlim(-7.00, 7.00)
    ax.set_ylim(-7.00, 7.00)
    ax.set_zlim(zmin, zmax)
    
    ax.view_init(elev=-90, azim=0)
    ax.grid(True)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_alpha(0.04)

    set_axes_equal_3d(ax)

    # Custom legend
    legend_handles = [
        # Line2D([0], [0], marker='s', color='w', label=r'Avoidance Region ($\neg O$)', markerfacecolor="#D62828", markeredgecolor='black', alpha=0.4, markersize=12),
        Line2D([0], [0], marker='s', color='w', label=r'Avoidance Region', markerfacecolor="#D62828", markeredgecolor='black', alpha=0.4, markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor="#1D3557", markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Goal', markerfacecolor="#D62828", markeredgecolor='black', markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.95)
    
    return lc, norm

# ============================================================
# 2) 2D Distances Plot for ALWAYS
# ============================================================
def plot_always_distances(ax, results, params):
    t = np.asarray(results["times_all"])
    x_all = np.asarray(results["x_all"])
    x_nodes = np.asarray(results["x_nmpc_all"][0])
    
    # Coordinates over time
    x_pts, y_pts = x_all[:, 0], x_all[:, 1]
    
    # Calculate continuous distance to boundaries
    # Note: 15.0 and -15.0 are used as the "infinite" bounds
    dist_obs1 = dist_to_rectangle(x_pts, y_pts, params['x_stc_1'], params['x_stc_2'], params['y_stc_1'], 15.0)
    dist_obs2 = dist_to_rectangle(x_pts, y_pts, params['x_stc_3'], params['x_stc_4'], -15.0, params['y_stc_2'])

    # Shade the SAFE region (Distance > 0)
    ax.axhspan(0.0, max(np.max(dist_obs1), np.max(dist_obs2)) * 1.05, color="#2A9D8F", alpha=0.10, lw=0, label="Safe Zone ($d > 0$)")
    
    # Shade the COLLISION region (Distance == 0)
    ax.axhspan(-0.5, 0.0, color="#D62828", alpha=0.20, lw=0, label="Collision Zone")

    # Plot continuous lines
    ax.plot(t, dist_obs1, color="#1D3557", lw=2.5, label=r"Distance to $O_1$")
    ax.plot(t, dist_obs2, color="#E9C46A", lw=2.5, label=r"Distance to $O_2$")

    # Threshold line (Must remain > 0)
    ax.axhline(0.0, color="black", lw=1.5, linestyle="--")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Obstacle Clearance [m]")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.5, max(np.max(dist_obs1), np.max(dist_obs2)) * 1.05)
    
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

# ============================================================
# Combined Paper Figure for ALWAYS
# ============================================================
def plot_always_demo_figure(results, params, savepath=None):
    set_paper_style()

    fig = plt.figure(figsize=(16.8, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1.0], wspace=0.25)

    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax_dist = fig.add_subplot(gs[1])

    # Plot 3D trajectory
    lc, norm = plot_always_trajectory_3d(ax3d, results, params)
    
    # Add colorbar for the 3D plot
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="rainbow"), ax=ax3d, fraction=0.02, pad=-0.05)
    cbar.set_label(r"Speed $\|v(t)\|$ [m/s]")

    # Plot 2D Distances
    plot_always_distances(ax_dist, results, params)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ============================================================
# Standalone Figure Exporter
# ============================================================
def save_individual_always_figures(results, params, save_dir):
    """
    Saves the 3D trajectory and the Obstacle Clearance plot 
    as two separate, high-quality files.
    """
    os.makedirs(save_dir, exist_ok=True)
    set_paper_style()

    # 1) 3D Trajectory (Standalone)
    fig1 = plt.figure(figsize=(7.5, 7.5)) # Perfect square for top-down view
    
    # Force layout to eliminate white space
    ax1 = fig1.add_axes([0.0, 0.0, 0.85, 1.0], projection="3d")
    cax = fig1.add_axes([0.78, 0.20, 0.02, 0.60])
    
    lc, norm = plot_always_trajectory_3d(ax1, results, params)
    
    cbar = fig1.colorbar(ScalarMappable(norm=norm, cmap="rainbow"), cax=cax)
    cbar.set_label(r"Speed $\|v(t)\|$ [m/s]")
    
    fig1.savefig(os.path.join(save_dir, "always_3d_traj.png"), bbox_inches="tight")
    fig1.savefig(os.path.join(save_dir, "always_3d_traj.pdf"), bbox_inches="tight")
    plt.close(fig1)

    # 2) Distances Plot (Standalone)
    fig2 = plt.figure(figsize=(6.5, 4.0)) # Shorter height physically
    ax2 = fig2.add_subplot(111)
    
    plot_always_distances(ax2, results, params)
    
    fig2.savefig(os.path.join(save_dir, "always_distances.png"), bbox_inches="tight")
    fig2.savefig(os.path.join(save_dir, "always_distances.pdf"), bbox_inches="tight")
    plt.close(fig2)
    
    print(f"Saved individual 'Always' figures to: {save_dir}")
