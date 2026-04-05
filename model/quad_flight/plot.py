import os
import glob
import numpy as np

import matplotlib as mpl
from cycler import cycler

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 120 
mpl.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']}) ## for Palatino and other serif fonts use: #rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)
mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:red', 'k', 'tab:green', 'tab:blue', 'tab:grey'])
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams.update({'font.size': 12})
mpl.rcParams.update({'legend.fontsize': 11})
mpl.rcParams.update({'axes.xmargin': 0})
mpl.rcParams.update({'lines.solid_capstyle': 'round'})
mpl.rcParams.update({'lines.solid_joinstyle': 'round'})
mpl.rcParams.update({'lines.dash_capstyle': 'round'})
mpl.rcParams.update({'lines.dash_joinstyle': 'round'})
mpl.rcParams.update({'text.latex.preamble': r"\usepackage{bm}"})


# ============================================================
# Rotation matrix from Euler angles [phi, theta, psi]
# Standard ZYX convention:
# R = Rz(psi) @ Ry(theta) @ Rx(phi)
# This maps body-frame vectors to world/inertial frame.
# ============================================================
def rotation_matrix(euler_angles):
    phi, theta, psi = np.asarray(euler_angles, dtype=float).reshape(-1)[:3]

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cphi, -sphi],
        [0.0, sphi, cphi],
    ])

    Ry = np.array([
        [cth, 0.0, sth],
        [0.0, 1.0, 0.0],
        [-sth, 0.0, cth],
    ])

    Rz = np.array([
        [cpsi, -spsi, 0.0],
        [spsi, cpsi, 0.0],
        [0.0, 0.0, 1.0],
    ])

    return Rz @ Ry @ Rx


# ============================================================
# Quadrotor drawing
# ============================================================
def plt_drone_fcn(ax, center, z_dir, length_drone, head_angle):
    def cyl(ax, p0, p1, rad_drone, clr=None, clr2=None):
        v = p1 - p0
        mag = np.linalg.norm(v + 1e-6)
        v = v / mag

        not_v = np.array([1.0, 0.0, 0.0])
        if np.allclose(v, not_v):
            not_v = np.array([0.0, 1.0, 0.0])

        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1 + 1e-6)
        n2 = np.cross(v, n1)

        t = np.linspace(0.0, mag, 2)
        theta = np.linspace(0.0, 2.0 * np.pi, 100)
        rsample = np.linspace(0.0, rad_drone, 2)

        t, theta2 = np.meshgrid(t, theta)
        rsample, theta = np.meshgrid(rsample, theta)

        X, Y, Z = [
            p0[i] + v[i] * t + rad_drone * np.sin(theta2) * n1[i]
            + rad_drone * np.cos(theta2) * n2[i]
            for i in [0, 1, 2]
        ]

        X2, Y2, Z2 = [
            p0[i] + rsample * np.sin(theta) * n1[i]
            + rsample * np.cos(theta) * n2[i]
            for i in [0, 1, 2]
        ]

        X3, Y3, Z3 = [
            p0[i] + v[i] * mag + rsample * np.sin(theta) * n1[i]
            + rsample * np.cos(theta) * n2[i]
            for i in [0, 1, 2]
        ]

        ax.plot_surface(X, Y, Z, color=clr, zorder=9)
        ax.plot_surface(X2, Y2, Z2, color=clr, zorder=9)
        ax.plot_surface(X3, Y3, Z3, color=clr, zorder=9)

        if clr2 is not None:
            phi = np.linspace(0.0, 2.0 * np.pi, 50)
            theta = np.linspace(0.0, np.pi, 25)

            dx = 3.0 * rad_drone * np.outer(np.cos(phi), np.sin(theta))
            dy = 3.0 * rad_drone * np.outer(np.sin(phi), np.sin(theta))
            dz = 3.0 * rad_drone * np.outer(np.ones(np.size(phi)), np.cos(theta))

            ax.plot_surface(p1[0] + dx, p1[1] + dy, p1[2] + dz,
                            cstride=1, rstride=1, color=clr2, zorder=10)
            ax.plot_surface(p0[0] + dx, p0[1] + dy, p0[2] + dz,
                            cstride=1, rstride=1, color=clr2, zorder=10)

    def rotate_vec(v, d, alpha):
        return (
            v * np.cos(alpha)
            + np.cross(d, v) * np.sin(alpha)
            + d * np.dot(d, v) * (1.0 - np.cos(alpha))
        )

    center = np.asarray(center, dtype=float)
    z_dir = np.asarray(z_dir, dtype=float)
    head_angle = np.asarray(head_angle, dtype=float)

    z_dir = z_dir / np.linalg.norm(z_dir + 1e-6)
    head_angle = head_angle / np.linalg.norm(head_angle + 1e-6)

    rad_drone = length_drone * 0.02

    l1_axis = rotate_vec(head_angle, z_dir, np.pi / 4.0)
    p0 = center - l1_axis * length_drone / 2.0
    p1 = center + l1_axis * length_drone / 2.0

    l2_axis = rotate_vec(l1_axis, z_dir, np.pi / 2.0)
    p2 = center - l2_axis * length_drone / 2.0
    p3 = center + l2_axis * length_drone / 2.0

    cyl(ax, p0, p1, rad_drone, clr='black', clr2='yellow')
    cyl(ax, p2, p3, rad_drone, clr='black', clr2='yellow')

    p6 = center
    p7 = center + head_angle * length_drone / 4.0
    cyl(ax, p6, p7, rad_drone / 1.5, clr='gray')

    p8 = center
    p9 = center + z_dir * length_drone / 2.0
    cyl(ax, p8, p9, rad_drone * 0.8, clr='red')


# ============================================================
# Small geometry helpers
# ============================================================
def _box_faces(x1, x2, y1, y2, z1, z2):
    p000 = [x1, y1, z1]
    p001 = [x1, y1, z2]
    p010 = [x1, y2, z1]
    p011 = [x1, y2, z2]
    p100 = [x2, y1, z1]
    p101 = [x2, y1, z2]
    p110 = [x2, y2, z1]
    p111 = [x2, y2, z2]

    return [
        [p000, p100, p110, p010],
        [p001, p101, p111, p011],
        [p000, p100, p101, p001],
        [p010, p110, p111, p011],
        [p000, p010, p011, p001],
        [p100, p110, p111, p101],
    ]


def add_box(ax, x1, x2, y1, y2, z1, z2,
            facecolor='gray', edgecolor='black',
            alpha=0.15, linewidth=0.5):
    if x2 <= x1 or y2 <= y1 or z2 <= z1:
        return

    poly = Poly3DCollection(
        _box_faces(x1, x2, y1, y2, z1, z2),
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(poly)


def add_sphere(ax, center, radius, facecolor='limegreen',
               edgecolor='green', alpha=0.25, wire=False):
    center = np.asarray(center, dtype=float)

    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)

    X = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    Y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    Z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    if wire:
        ax.plot_wireframe(
            X, Y, Z,
            rstride=2, cstride=2,
            color=edgecolor,
            linewidth=0.8,
            alpha=max(alpha, 0.5),
        )
    else:
        ax.plot_surface(
            X, Y, Z,
            color=facecolor,
            edgecolor='none',
            alpha=alpha,
            shade=True,
        )
        ax.plot_wireframe(
            X, Y, Z,
            rstride=4, cstride=4,
            color=edgecolor,
            linewidth=0.35,
            alpha=min(alpha + 0.2, 0.9),
        )


def contiguous_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    segments = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = i
            prev = i
    segments.append((start, prev))
    return segments


# ============================================================
# Time helpers
# ============================================================
def get_state_times(results):
    x_all = np.asarray(results['x_all'], dtype=float)

    if 'times_all' in results:
        t = np.asarray(results['times_all'], dtype=float).reshape(-1)
        if len(t) == x_all.shape[0]:
            return t

    return x_all[:, 18] * 10.0


def get_control_series(results, control_idx):
    t_state = get_state_times(results)
    u_all = np.asarray(results['u_all'], dtype=float)

    if u_all.ndim == 1:
        return np.linspace(t_state[0], t_state[-1], len(u_all)), u_all

    if u_all.ndim == 2:
        if u_all.shape[1] >= 4:
            return np.linspace(t_state[0], t_state[-1], u_all.shape[0]), u_all[:, control_idx]
        if u_all.shape[0] >= 4:
            return np.linspace(t_state[0], t_state[-1], u_all.shape[1]), u_all[control_idx, :]

    if u_all.ndim == 3:
        # common case: (Nseg, nu, nsub)
        if u_all.shape[1] >= 4:
            y = u_all[:, control_idx, :].reshape(-1, order='F')

            if u_all.shape[0] == len(t_state) - 1 and u_all.shape[2] == 2:
                t_u = np.dstack((t_state[:-1], t_state[1:])).reshape(-1)
                return t_u, y

            return np.linspace(t_state[0], t_state[-1], len(y)), y

        # fallback: (nu, Nseg, nsub)
        if u_all.shape[0] >= 4:
            y = u_all[control_idx, :, :].reshape(-1, order='F')
            return np.linspace(t_state[0], t_state[-1], len(y)), y

    raise ValueError("Could not infer control array shape from results['u_all'].")


# ============================================================
# Problem geometry from your moving-wall / moving-sphere setup
# ============================================================
def build_problem_geometry(results, params):
    x_all = np.asarray(results['x_all'], dtype=float)

    x_traj = x_all[:, 0]
    y_traj = x_all[:, 1]
    z_traj = x_all[:, 2]
    time = x_all[:, 18] * 10.0

    delta_p = params['y_stc_range'] * (np.sin(time - np.pi / 2.0) + 1.0)

    obstacles = [
        dict(name='obs4', x1=float(params['x_stc_7']),  x2=float(params['x_stc_8']),
             y0=float(params['y_stc_4']), occupied='above', sign=-1.0),

        dict(name='obs1', x1=float(params['x_stc_1']),  x2=float(params['x_stc_2']),
             y0=float(params['y_stc_1']), occupied='below', sign=-1.0),

        dict(name='obs2', x1=float(params['x_stc_3']),  x2=float(params['x_stc_4']),
             y0=float(params['y_stc_2']), occupied='above', sign=-1.0),

        dict(name='obs5', x1=float(params['x_stc_9']),  x2=float(params['x_stc_10']),
             y0=float(params['y_stc_5']), occupied='below', sign=-1.0),

        dict(name='obs6', x1=float(params['x_stc_11']), x2=float(params['x_stc_12']),
             y0=float(params['y_stc_6']), occupied='above', sign=+1.0),

        dict(name='obs3', x1=float(params['x_stc_5']),  x2=float(params['x_stc_6']),
             y0=float(params['y_stc_3']), occupied='below', sign=+1.0),
    ]

    for ob in obstacles:
        ob['y_t'] = ob['y0'] + ob['sign'] * delta_p
        ob['y_min'] = float(np.min(ob['y_t']))
        ob['y_max'] = float(np.max(ob['y_t']))

    groups = {}
    for ob in obstacles:
        key = (round(ob['x1'], 10), round(ob['x2'], 10))
        groups.setdefault(key, []).append(ob)

    doors = []
    for _, grp in groups.items():
        lower = next((g for g in grp if g['occupied'] == 'below'), None)
        upper = next((g for g in grp if g['occupied'] == 'above'), None)

        if lower is None or upper is None:
            continue

        x1 = grp[0]['x1']
        x2 = grp[0]['x2']
        xc = 0.5 * (x1 + x2)

        inside = np.where((x_traj >= x1) & (x_traj <= x2))[0]
        if len(inside) > 0:
            k = inside[np.argmin(np.abs(x_traj[inside] - xc))]
        else:
            k = int(np.argmin(np.abs(x_traj - xc)))

        y_low = float(lower['y_t'][k])
        y_high = float(upper['y_t'][k])

        if y_low > y_high:
            y_low, y_high = y_high, y_low

        doors.append(dict(
            x1=x1,
            x2=x2,
            xc=xc,
            k=k,
            t=float(time[k]),
            y_low=y_low,
            y_high=y_high,
            lower=lower,
            upper=upper,
        ))

    doors = sorted(doors, key=lambda d: d['xc'])

    p_w_1 = np.asarray(params['p_w_1'], dtype=float)
    p_w_2 = np.asarray(params['p_w_2'], dtype=float)

    y_loc_1 = params['pw1_base'] + params['pw_range'] * 0.5 * (
        np.sin(np.pi / 5.655 * time - np.pi / 2.0) + 1.0
    )
    y_loc_2 = params['pw2_base'] - params['pw_range'] * 0.5 * (
        np.sin(np.pi / 4.38 * time - np.pi / 2.0) + 1.0
    )

    spheres = [
        dict(
            name='sphere 1',
            x=float(p_w_1[0]),
            z=float(p_w_1[2]),
            r=float(params['r_w_1']),
            y_t=y_loc_1,
            edgecolor='green',
            facecolor='limegreen',
        ),
        dict(
            name='sphere 2',
            x=float(p_w_2[0]),
            z=float(p_w_2[2]),
            r=float(params['r_w_2']),
            y_t=y_loc_2,
            edgecolor='purple',
            facecolor='violet',
        ),
    ]

    for sph in spheres:
        sph['center_t'] = np.column_stack([
            np.full_like(time, sph['x']),
            sph['y_t'],
            np.full_like(time, sph['z']),
        ])
        sph['dist_sq'] = sph['r']**2 - (
            (x_traj - sph['x'])**2 +
            (y_traj - sph['y_t'])**2 +
            (z_traj - sph['z'])**2
        )
        sph['visit'] = sph['dist_sq'] > 0.0

    sphere_snapshots = []
    for sph in spheres:
        for seg_id, (a, b) in enumerate(contiguous_true_segments(sph['visit']), start=1):
            idx = np.arange(a, b + 1)
            k = int(idx[np.argmax(sph['dist_sq'][idx])])

            sphere_snapshots.append(dict(
                name=sph['name'],
                seg_id=seg_id,
                k=k,
                t=float(time[k]),
                center=sph['center_t'][k].copy(),
                r=sph['r'],
                edgecolor=sph['edgecolor'],
                facecolor=sph['facecolor'],
            ))

    return {
        'x_all': x_all,
        'time': time,
        'obstacles': obstacles,
        'doors': doors,
        'spheres': spheres,
        'sphere_snapshots': sphere_snapshots,
    }


def get_plot_limits(geom, params=None, pad_xyz=(2.0, 2.0, 2.0)):
    x_all = geom['x_all']
    obstacles = geom['obstacles']
    spheres = geom['spheres']

    pad_x, pad_y, pad_z = pad_xyz

    x_vals = [x_all[:, 0].min(), x_all[:, 0].max()]
    y_vals = [x_all[:, 1].min(), x_all[:, 1].max()]
    z_vals = [x_all[:, 2].min(), x_all[:, 2].max()]

    for ob in obstacles:
        x_vals.extend([ob['x1'], ob['x2']])
        y_vals.extend([ob['y_min'], ob['y_max']])

    for sph in spheres:
        x_vals.extend([sph['x'] - sph['r'], sph['x'] + sph['r']])
        y_vals.extend([np.min(sph['y_t']) - sph['r'], np.max(sph['y_t']) + sph['r']])
        z_vals.extend([sph['z'] - sph['r'], sph['z'] + sph['r']])

    xmin = min(x_vals) - pad_x
    xmax = max(x_vals) + pad_x
    ymin = min(y_vals) - pad_y
    ymax = max(y_vals) + pad_y
    zmin = min(z_vals) - pad_z
    zmax = max(z_vals) + pad_z

    if params is not None:
        if 'add_min_alt' in params and params['add_min_alt'] and 'w_min_alt' in params:
            zmin = min(zmin, float(params['w_min_alt']) - pad_z)
        if 'add_max_alt' in params and params['add_max_alt'] and 'w_max_alt' in params:
            zmax = max(zmax, float(params['w_max_alt']) + pad_z)

    return xmin, xmax, ymin, ymax, zmin, zmax


def style_3d_axes_plt(ax, xmin, xmax, ymin, ymax, zmin, zmax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.set_xlabel(r'$r_x$ [m]', size=16)
    ax.set_ylabel(r'$r_y$ [m]', size=16)
    ax.set_zlabel(r'$r_z$ [m]', size=16)

    ax.xaxis.labelpad = 18
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 4

    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.zaxis.set_tick_params(labelsize=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    try:
        ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin), zoom=0.88)
    except Exception:
        pass

    ax.view_init(elev=45, azim=220)


def draw_obstacle_shadows_3d_plt(ax, obstacles, zmin, zmax):
    for ob in obstacles:
        add_box(
            ax,
            ob['x1'], ob['x2'],
            ob['y_min'], ob['y_max'],
            zmin, zmax,
            facecolor='gray',
            edgecolor='none',
            alpha=0.10,
            linewidth=0.0,
        )

# ============================================================
# Static state/control figure
# ============================================================
def plot_states_controls(results, params, save_path=None):
    t_state = get_state_times(results)
    x_all = np.asarray(results['x_all'], dtype=float)

    t_u_T, u_T = get_control_series(results, 3)
    t_u_tx, u_tx = get_control_series(results, 0)
    t_u_ty, u_ty = get_control_series(results, 1)
    t_u_tz, u_tz = get_control_series(results, 2)

    spd_norm = np.linalg.norm(x_all[:, 3:6], axis=1)

    fig, axs = plt.subplots(
        4, 3,
        gridspec_kw={'height_ratios': [3, 3, 3, 3]},
        figsize=(8 * 0.95, 10 * 0.95)
    )

    xy_fs = 14
    c_up_lim = 'green'
    c_plt = 'blue'

    # Row 1
    axa = axs[0, 0]
    axa.plot(t_u_T, u_T, c='blue')
    if 'T_max' in params:
        axa.axhline(y=params['T_max'], c=c_up_lim, linestyle='dashed')
    if 'T_min' in params:
        axa.axhline(y=params['T_min'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('Thrust, $T$ [N]', fontsize=xy_fs)

    axa = axs[0, 1]
    axa.plot(t_state, spd_norm, c=c_plt)
    if 'vehicle_v_max' in params:
        axa.axhline(y=params['vehicle_v_max'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('Speed, $v$ [m s$^{-1}$]', fontsize=xy_fs)

    axa = axs[0, 2]
    axa.plot(t_state, x_all[:, 2], c=c_plt)
    if 'min_alt' in params:
        axa.axhline(y=params['min_alt'], c=c_up_lim, linestyle='dashed')
    if 'add_max_alt' in params and params['add_max_alt'] and 'w_max_alt' in params:
        axa.axhline(y=params['w_max_alt'], c=c_up_lim, linestyle='dashed')
    if 'add_min_alt' in params and params['add_min_alt'] and 'w_min_alt' in params:
        axa.axhline(y=params['w_min_alt'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('Altitude, $r_z$ [m]', fontsize=xy_fs)

    # Row 2
    axa = axs[1, 0]
    axa.plot(t_u_tx, u_tx, c='blue')
    if 'tau_max' in params:
        axa.axhline(y=params['tau_max'], c=c_up_lim, linestyle='dashed')
        axa.axhline(y=-params['tau_max'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$\\tau_x$ [N m]', fontsize=xy_fs)

    axa = axs[1, 1]
    axa.plot(t_u_ty, u_ty, c='blue')
    if 'tau_max' in params:
        axa.axhline(y=params['tau_max'], c=c_up_lim, linestyle='dashed')
        axa.axhline(y=-params['tau_max'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$\\tau_y$ [N m]', fontsize=xy_fs)

    axa = axs[1, 2]
    axa.plot(t_u_tz, u_tz, c='blue')
    if 'tau_max' in params:
        axa.axhline(y=params['tau_max'], c=c_up_lim, linestyle='dashed')
        axa.axhline(y=-params['tau_max'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$\\tau_z$ [N m]', fontsize=xy_fs)

    # Row 3
    axa = axs[2, 0]
    axa.plot(t_state, x_all[:, 9] * 180.0 / np.pi, c='blue')
    if 'phi_rate' in params:
        bd = params['phi_rate'] * 180.0 / np.pi
        axa.axhline(y=bd, c=c_up_lim, linestyle='dashed')
        axa.axhline(y=-bd, c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$p$ [deg s$^{-1}$]', fontsize=xy_fs)

    axa = axs[2, 1]
    axa.plot(t_state, x_all[:, 10] * 180.0 / np.pi, c='blue')
    if 'theta_rate' in params:
        bd = params['theta_rate'] * 180.0 / np.pi
        axa.axhline(y=bd, c=c_up_lim, linestyle='dashed')
        axa.axhline(y=-bd, c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$q$ [deg s$^{-1}$]', fontsize=xy_fs)

    axa = axs[2, 2]
    axa.plot(t_state, x_all[:, 11] * 180.0 / np.pi, c='blue')
    if 'yaw_rate' in params:
        bd = params['yaw_rate'] * 180.0 / np.pi
        axa.axhline(y=bd, c=c_up_lim, linestyle='dashed')
        axa.axhline(y=-bd, c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$r$ [deg s$^{-1}$]', fontsize=xy_fs)

    # Row 4
    axa = axs[3, 0]
    axa.plot(t_state, x_all[:, 6] * 180.0 / np.pi, c='blue')
    if 'phi_bd' in params:
        bd = params['phi_bd'] * 180.0 / np.pi
        axa.axhline(y=bd, c=c_up_lim, linestyle='dashed')
        axa.axhline(y=-bd, c=c_up_lim, linestyle='dashed')
    axa.set_xlabel('Time [s]', fontsize=xy_fs)
    axa.set_ylabel('$\\phi$ [deg]', fontsize=xy_fs)

    axa = axs[3, 1]
    axa.plot(t_state, x_all[:, 7] * 180.0 / np.pi, c='blue')
    if 'theta_bd' in params:
        bd = params['theta_bd'] * 180.0 / np.pi
        axa.axhline(y=bd, c=c_up_lim, linestyle='dashed')
        axa.axhline(y=-bd, c=c_up_lim, linestyle='dashed')
    axa.set_xlabel('Time [s]', fontsize=xy_fs)
    axa.set_ylabel('$\\theta$ [deg]', fontsize=xy_fs)

    axa = axs[3, 2]
    axa.plot(t_state, x_all[:, 8] * 180.0 / np.pi, c='blue')
    axa.axhline(y=180.0, c=c_up_lim, linestyle='dashed')
    axa.axhline(y=-180.0, c=c_up_lim, linestyle='dashed')
    axa.set_xlabel('Time [s]', fontsize=xy_fs)
    axa.set_ylabel('$\\psi$ [deg]', fontsize=xy_fs)

    for row in axs:
        for axa in row:
            axa.grid(alpha=0.25)
            axa.tick_params(labelsize=11)
            axa.set_xlim(t_state[0], t_state[-1])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.show()

# ============================================================
# Static 3D plot + states plot
# ============================================================
def qf_plot(results, params, file_name, xylim=None):
    os.makedirs(file_name, exist_ok=True)

    geom = build_problem_geometry(results, params)
    x_all = geom['x_all']
    time = geom['time']
    obstacles = geom['obstacles']
    doors = geom['doors']
    spheres = geom['spheres']
    sphere_snapshots = geom['sphere_snapshots']

    spd_norm = np.linalg.norm(x_all[:, 3:6], axis=1)
    seg_speed = 0.5 * (spd_norm[:-1] + spd_norm[1:]) if len(spd_norm) >= 2 else spd_norm

    xmin, xmax, ymin, ymax, zmin, zmax = get_plot_limits(geom, params, pad_xyz=(2.0, 2.0, 2.0))
    if xylim is not None:
        xmin, xmax = xylim[0]
        ymin, ymax = xylim[1]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    if x_all.shape[0] >= 2:
        points = x_all[:, 0:3].reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = Line3DCollection(
            segments,
            cmap=plt.cm.rainbow,
            norm=plt.Normalize(0.0, max(spd_norm.max(), 1e-9)),
            array=seg_speed,
            linewidth=3.0,
        )
        ax.add_collection(lc)

        cbar = fig.colorbar(lc, ax=ax, shrink=0.60, pad=0.02)
        cbar.set_label(r"Speed [m s$^{-1}$]", fontsize=14)

    draw_obstacle_shadows_3d_plt(ax, obstacles, zmin, zmax)

    for i, d in enumerate(doors, start=1):
        add_box(
            ax,
            d['x1'], d['x2'],
            d['y_low'], d['y_high'],
            zmin, zmax,
            facecolor='deepskyblue',
            edgecolor='blue',
            alpha=0.16,
            linewidth=0.45,
        )

        ax.scatter(
            [x_all[d['k'], 0]],
            [x_all[d['k'], 1]],
            [x_all[d['k'], 2]],
            s=60,
            c='gold',
            edgecolors='black',
            zorder=12,
        )

        ax.text(
            d['xc'],
            0.5 * (d['y_low'] + d['y_high']),
            zmax,
            f'door {i}, t={d["t"]:.2f}',
            fontsize=10,
            color='blue',
        )

    for sph in spheres:
        ax.plot(
            np.full_like(time, sph['x']),
            sph['y_t'],
            np.full_like(time, sph['z']),
            linestyle=':',
            linewidth=1.5,
            alpha=0.6,
            color=sph['edgecolor'],
        )

    for snap in sphere_snapshots:
        add_sphere(
            ax,
            center=snap['center'],
            radius=snap['r'],
            facecolor=snap['facecolor'],
            edgecolor=snap['edgecolor'],
            alpha=0.16,
            wire=False,
        )

        k = snap['k']
        ax.scatter(
            [x_all[k, 0]],
            [x_all[k, 1]],
            [x_all[k, 2]],
            s=55,
            c='yellow',
            edgecolors='black',
            zorder=12,
        )

        ax.text(
            snap['center'][0],
            snap['center'][1],
            snap['center'][2] + snap['r'] + 0.4,
            f'{snap["name"]}, t={snap["t"]:.2f}',
            fontsize=10,
            color=snap['edgecolor'],
        )

    drone_scale = np.linalg.norm(x_all[:, 0:3].max(axis=0) - x_all[:, 0:3].min(axis=0))
    # length_drone = 0.05 * max(drone_scale, 1.0)
    length_drone = 0.08 * max(drone_scale, 1.0)

    for k in [0, x_all.shape[0] - 1]:
        R = rotation_matrix(x_all[k, 6:9])
        z_dir = (R @ np.array([0.0, 0.0, 1.0])).ravel()
        head_angle = (R @ np.array([1.0, 0.0, 0.0])).ravel()

        plt_drone_fcn(
            ax=ax,
            center=x_all[k, 0:3],
            z_dir=z_dir,
            length_drone=length_drone,
            head_angle=head_angle,
        )

    style_3d_axes_plt(ax, xmin, xmax, ymin, ymax, zmin, zmax)
    # ax.set_title('Quadrotor trajectory, moving walls, doors, and moving visit spheres', fontsize=15)

    save_path_3d = os.path.join(file_name, 'qf_traj.png')
    save_path_3d_pdf = os.path.join(file_name, 'qf_traj.pdf')
    plt.tight_layout()
    plt.savefig(save_path_3d, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_3d_pdf, bbox_inches='tight')
    plt.show()

    save_path_states = os.path.join(file_name, 'qf_states')
    plot_states_controls(results, params, save_path=save_path_states)

    print(f'Saved static 3D figure to: {save_path_3d}')
    print(f'Saved states/controls figure to: {save_path_states}')

# ============================================================
# ENLARGED FONT 3D AXIS SETTINGS
# ============================================================
def style_3d_axes(ax, xmin, xmax, ymin, ymax, zmin, zmax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Fonts made huge (18)
    ax.set_xlabel(r'$r_x$ [m]', size=18)
    ax.set_ylabel(r'$r_y$ [m]', size=18)
    ax.set_zlabel(r'$r_z$ [m]', size=18)
    
    ax.xaxis.labelpad = 18
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 8

    # Axis ticks enlarged (14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.zaxis.set_tick_params(labelsize=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    # ax.view_init(elev=30, azim=200)
    # ax.view_init(elev=45, azim=220)
    # ax.view_init(elev=45, azim=240)
    ax.view_init(elev=45, azim=210)
    # ax.view_init(elev=35, azim=220)


def draw_obstacle_shadows_3d(ax, obstacles, zmin, zmax):
    for ob in obstacles:
        add_box(ax, ob['x1'], ob['x2'], ob['y_min'], ob['y_max'], zmin, zmax, facecolor='gray', edgecolor='none', alpha=0.10, linewidth=0.0)

def add_vertical_plane(ax, x1, x2, y, z1, z2, facecolor='dimgray', edgecolor='black', alpha=0.18, linewidth=0.8):
    verts = [[[x1, y, z1], [x2, y, z1], [x2, y, z2], [x1, y, z2]]]
    poly = Poly3DCollection(verts, facecolors=facecolor, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(poly)

def draw_current_obstacle_planes_3d(ax, obstacles, k, zmin, zmax):
    for ob in obstacles:
        y_now = float(ob['y_t'][k])
        add_vertical_plane(ax, ob['x1'], ob['x2'], y_now, zmin, zmax, facecolor='dimgray', edgecolor='black', alpha=0.20, linewidth=0.7)
        ax.plot([ob['x1'], ob['x2']], [y_now, y_now], [zmin, zmin], color='black', linewidth=0.8, alpha=0.8)
        ax.plot([ob['x1'], ob['x2']], [y_now, y_now], [zmax, zmax], color='black', linewidth=0.8, alpha=0.8)
        ax.plot([ob['x1'], ob['x1']], [y_now, y_now], [zmin, zmax], color='black', linewidth=0.8, alpha=0.8)
        ax.plot([ob['x2'], ob['x2']], [y_now, y_now], [zmin, zmax], color='black', linewidth=0.8, alpha=0.8)

def draw_current_doors_3d(ax, doors, k, zmin, zmax):
    for d in doors:
        y_low = float(d['lower']['y_t'][k])
        y_high = float(d['upper']['y_t'][k])
        if y_low > y_high: y_low, y_high = y_high, y_low
        if y_high > y_low:
            add_box(ax, d['x1'], d['x2'], y_low, y_high, zmin, zmax, facecolor='deepskyblue', edgecolor='blue', alpha=0.16, linewidth=0.4)

# ============================================================
# ENLARGED FONT AND LINE 2D PLOT SETTINGS
# ============================================================
def plot_running_series(ax, t, y, current_t, ylabel, line_color='blue', bounds=None):
    t = np.asarray(t).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = t <= current_t

    if np.any(mask):
        ax.plot(t[mask], y[mask], color=line_color, linewidth=2.5) # Lines thickened
        ax.scatter([t[mask][-1]], [y[mask][-1]], s=35, color='black', zorder=5) # Scatter point enlarged

    if bounds is not None:
        for bd in bounds: ax.axhline(y=bd, color='green', linestyle='dashed', linewidth=1.5, alpha=0.8)

    ax.axvline(x=current_t, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.set_xlim(t[0], t[-1])
    
    # Fonts adjusted for clarity
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel('Time [s]', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.25)


# ============================================================
# ANIMATION (PERFECT GRIDSPEC & ZOOM TRICK)
# ============================================================
def animation(results, params, file_name, delt=2):
    os.makedirs(file_name, exist_ok=True)

    # for f in glob.glob(os.path.join(file_name, 'anim_*.png')):
    #     os.remove(f)

    geom = build_problem_geometry(results, params)
    x_all = geom['x_all']
    time = geom['time']
    obstacles = geom['obstacles']
    doors = geom['doors']
    spheres = geom['spheres']

    # BASIS FOR ZOOM TRICK: Adding a massive 5-meter artificial padding to the boundaries.
    # This way, when we zoom, this padding area gets clipped instead of the objects (spheres)!
    xmin, xmax, ymin, ymax, zmin, zmax = get_plot_limits(
        geom, params, pad_xyz=(5.0, 5.0, 2.0)
    )

    drone_scale = np.linalg.norm(
        x_all[:, 0:3].max(axis=0) - x_all[:, 0:3].min(axis=0)
    )
    # length_drone = 0.05 * max(drone_scale, 1.0)
    # length_drone = 0.08 * max(drone_scale, 1.0)
    length_drone = 0.1 * max(drone_scale, 1.0)

    spd_full = np.linalg.norm(x_all[:, 3:6], axis=1)

    t_state = get_state_times(results)
    t_u_T, u_T = get_control_series(results, 3)
    t_u_tx, u_tx = get_control_series(results, 0)
    t_u_ty, u_ty = get_control_series(results, 1)
    t_u_tz, u_tz = get_control_series(results, 2)

    for k in tqdm(range(0, x_all.shape[0], delt)):
        # Canvas size wide and tall
        fig = plt.figure(figsize=(24, 13))

        # ====================================================
        # A SINGLE SAFE GRIDSPEC 
        # ====================================================
        gs = gridspec.GridSpec(
            nrows=5, ncols=6,
            left=0.04, right=0.98, top=0.96, bottom=0.07,
            wspace=0.28, hspace=0.15, # Spacing increased to prevent text from overlapping
            height_ratios=[1.0, 1.0, 1.0, 1.0, 1.3], # BOTTOM CONTROLS 30% TALLER
            width_ratios=[1.4, 1.4, 1.4, 1.4, 1.1, 1.1] # MASSIVE 3D AREA, State plots proportional
        )

        ax3d = fig.add_subplot(gs[0:4, 0:4], projection='3d')

        ax_alt   = fig.add_subplot(gs[0, 4])
        ax_speed = fig.add_subplot(gs[0, 5])
        ax_p     = fig.add_subplot(gs[1, 4])
        ax_phi   = fig.add_subplot(gs[1, 5])
        ax_q     = fig.add_subplot(gs[2, 4])
        ax_theta = fig.add_subplot(gs[2, 5])
        ax_r     = fig.add_subplot(gs[3, 4])
        ax_psi   = fig.add_subplot(gs[3, 5])

        ax_T  = fig.add_subplot(gs[4, 0])
        ax_tx = fig.add_subplot(gs[4, 1])
        ax_ty = fig.add_subplot(gs[4, 2])
        ax_tz = fig.add_subplot(gs[4, 3])

        # ====================================================
        # 3D Scene Drawing
        # ====================================================
        if k >= 1:
            points = x_all[:k+1, 0:3].reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            seg_spd = 0.5 * (spd_full[:k] + spd_full[1:k+1])

            lc = Line3DCollection(
                segments, cmap=plt.cm.rainbow,
                norm=plt.Normalize(0.0, max(spd_full.max(), 1e-9)),
                array=seg_spd, linewidth=3.2, zorder=3,
            )
            ax3d.add_collection(lc)

        draw_obstacle_shadows_3d(ax3d, obstacles, zmin, zmax)
        draw_current_obstacle_planes_3d(ax3d, obstacles, k, zmin, zmax)
        draw_current_doors_3d(ax3d, doors, k, zmin, zmax)

        title_tags = []
        for sph in spheres:
            center_now = sph['center_t'][k]
            inside_now = bool(sph['visit'][k])

            ax3d.plot(
                np.full(k + 1, sph['x']), sph['y_t'][:k+1], np.full(k + 1, sph['z']),
                linestyle=':', linewidth=1.8, alpha=0.55, color=sph['edgecolor'],
            )
            add_sphere(
                ax3d, center=center_now, radius=sph['r'],
                facecolor=sph['facecolor'], edgecolor=sph['edgecolor'],
                alpha=0.18 if inside_now else 0.10, wire=not inside_now,
            )
            if inside_now: title_tags.append(f'inside {sph["name"]}')

        R = rotation_matrix(x_all[k, 6:9])
        z_dir = (R @ np.array([0.0, 0.0, 1.0])).ravel()
        head_angle = (R @ np.array([1.0, 0.0, 0.0])).ravel()
        plt_drone_fcn(ax=ax3d, center=x_all[k, 0:3], z_dir=z_dir, length_drone=length_drone, head_angle=head_angle)

        ax3d.scatter([x_all[k, 0]], [x_all[k, 1]], [x_all[k, 2]], s=45, c='gold', edgecolors='black', zorder=12)
        style_3d_axes(ax3d, xmin, xmax, ymin, ymax, zmin, zmax)

        # ====================================================
        # WHITE SPACE AND CLIPPING SOLUTION
        # ====================================================
        # Thanks to padding=5.0, a 1.20 zoom only swallows the empty spaces, spheres remain safe!
        # Slightly extending the Z axis with (zmax-zmin)*1.2 to prevent flattening.
        try:
            ax3d.set_box_aspect((xmax - xmin, ymax - ymin, (zmax - zmin)*1.2), zoom=1.20)
            # ax3d.set_box_aspect((xmax - xmin, ymax - ymin, (zmax - zmin)*1.2), zoom=1.50)
        except TypeError:
            ax3d.set_box_aspect((xmax - xmin, ymax - ymin, (zmax - zmin)*1.2))
            ax3d.dist = 7.5

        title = f't = {time[k]:.3f}'
        if len(title_tags) > 0: title += '   |   ' + ', '.join(title_tags)
        ax3d.set_title(title, fontsize=18, pad=8)

        # ====================================================
        # 2D Plots (State & Control)
        # ====================================================
        current_t = t_state[k]

        altitude_bounds = []
        if 'min_alt' in params: altitude_bounds.append(params['min_alt'])
        if 'add_max_alt' in params and params['add_max_alt'] and 'w_max_alt' in params: altitude_bounds.append(params['w_max_alt'])
        if 'add_min_alt' in params and params['add_min_alt'] and 'w_min_alt' in params: altitude_bounds.append(params['w_min_alt'])

        plot_running_series(ax_alt, t_state, x_all[:, 2], current_t, r'$r_z$ [m]', bounds=altitude_bounds if len(altitude_bounds) > 0 else None)
        plot_running_series(ax_speed, t_state, spd_full, current_t, r'$v$ [m/s]', bounds=[params['vehicle_v_max']] if 'vehicle_v_max' in params else None)

        p_bounds = [-params['phi_rate'] * 180.0 / np.pi, params['phi_rate'] * 180.0 / np.pi] if 'phi_rate' in params else None
        q_bounds = [-params['theta_rate'] * 180.0 / np.pi, params['theta_rate'] * 180.0 / np.pi] if 'theta_rate' in params else None
        r_bounds = [-params['yaw_rate'] * 180.0 / np.pi, params['yaw_rate'] * 180.0 / np.pi] if 'yaw_rate' in params else None

        plot_running_series(ax_p, t_state, x_all[:, 9] * 180.0 / np.pi, current_t, r'$p$ [deg/s]', bounds=p_bounds)
        plot_running_series(ax_q, t_state, x_all[:, 10] * 180.0 / np.pi, current_t, r'$q$ [deg/s]', bounds=q_bounds)
        plot_running_series(ax_r, t_state, x_all[:, 11] * 180.0 / np.pi, current_t, r'$r$ [deg/s]', bounds=r_bounds)

        phi_bounds = [-params['phi_bd'] * 180.0 / np.pi, params['phi_bd'] * 180.0 / np.pi] if 'phi_bd' in params else None
        theta_bounds = [-params['theta_bd'] * 180.0 / np.pi, params['theta_bd'] * 180.0 / np.pi] if 'theta_bd' in params else None

        plot_running_series(ax_phi,   t_state, x_all[:, 6] * 180.0 / np.pi, current_t, r'$\phi$ [deg]', bounds=phi_bounds)
        plot_running_series(ax_theta, t_state, x_all[:, 7] * 180.0 / np.pi, current_t, r'$\theta$ [deg]', bounds=theta_bounds)
        plot_running_series(ax_psi,   t_state, x_all[:, 8] * 180.0 / np.pi, current_t, r'$\psi$ [deg]', bounds=[-180.0, 180.0])

        tau_bounds = [-params['tau_max'], params['tau_max']] if 'tau_max' in params else None

        plot_running_series(ax_T, t_u_T, u_T, current_t, r'$T$ [N]', bounds=[params['T_min'], params['T_max']] if ('T_min' in params and 'T_max' in params) else None)
        plot_running_series(ax_tx, t_u_tx, u_tx, current_t, r'$\tau_x$ [Nm]', bounds=tau_bounds)
        plot_running_series(ax_ty, t_u_ty, u_ty, current_t, r'$\tau_y$ [Nm]', bounds=tau_bounds)
        plot_running_series(ax_tz, t_u_tz, u_tz, current_t, r'$\tau_z$ [Nm]', bounds=tau_bounds)

        # Label Padding adjustment (To prevent text from sticking too close to the plot)
        for ax in [ax_alt, ax_speed, ax_p, ax_phi, ax_q, ax_theta, ax_r, ax_psi, ax_T, ax_tx, ax_ty, ax_tz]:
            ax.yaxis.labelpad = 4 

        # Hide x-axis (Time) labels on upper rows to save space
        for ax in [ax_alt, ax_speed, ax_p, ax_phi, ax_q, ax_theta]:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)

        save_path = os.path.join(file_name, f'anim_{str(k).zfill(5)}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f'Saved animation frames to: {file_name}')
