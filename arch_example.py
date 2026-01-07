from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from truss_analysis import TrussAnalysis, TrussElement

EQUIV_THICKNESS = (1.0 / 6.0) ** 0.5  # I=1/12 with A=1 for two chords


def _estimate_initial_ds(analysis, psi, target_dlam):
    num_nodes = len(analysis.nodes)
    dof_per_node = len(analysis.nodes[0])
    total_dofs = num_nodes * dof_per_node

    free_dofs = []
    for i in range(num_nodes):
        for j in range(dof_per_node):
            if (i, j) not in analysis.constrained_dofs:
                free_dofs.append(i * dof_per_node + j)

    F_ext_ref = np.zeros(total_dofs)
    for (n_idx, a_idx), val in analysis.loads.items():
        idx = n_idx * dof_per_node + a_idx
        F_ext_ref[idx] = val

    F_ext_ref_free = F_ext_ref[free_dofs]
    fext_norm_sq = float(np.dot(F_ext_ref_free, F_ext_ref_free))

    K_tan = np.zeros((total_dofs, total_dofs))
    for elem in analysis.elements:
        _, _, k_sub, _, _, _ = elem.get_force_and_stiffness(analysis.current_nodes)
        idx1 = elem.node1_idx * dof_per_node
        idx2 = elem.node2_idx * dof_per_node
        K_tan[idx1 : idx1 + dof_per_node, idx1 : idx1 + dof_per_node] += k_sub
        K_tan[idx2 : idx2 + dof_per_node, idx2 : idx2 + dof_per_node] += k_sub
        K_tan[idx1 : idx1 + dof_per_node, idx2 : idx2 + dof_per_node] -= k_sub
        K_tan[idx2 : idx2 + dof_per_node, idx1 : idx1 + dof_per_node] -= k_sub

    K_free = K_tan[np.ix_(free_dofs, free_dofs)]
    try:
        u_F = np.linalg.solve(K_free, F_ext_ref_free)
    except np.linalg.LinAlgError:
        u_F = np.linalg.lstsq(K_free, F_ext_ref_free, rcond=None)[0]

    uF_norm = float(np.linalg.norm(u_F))
    ds = target_dlam * np.sqrt(uF_norm * uF_norm + (psi**2) * fext_norm_sq)
    if ds <= 0.0 or not np.isfinite(ds):
        ds = 1.0e-4

    return ds, uF_norm, fext_norm_sq


def build_arch_truss(
    radius=100.0,
    rise=40.0,
    half_span=80.0,
    thickness=EQUIV_THICKNESS,
    n_segments=40,
    area=1.0,
    youngs=1.0e7,
    sig_y0=1.0e20,
    hardening=1.0,
    add_diagonals=True,
    brace_area_scale=1.0,
    load_mag=100.0,
    imperfection=1.0e-4,
):
    center_x = 0.0
    center_y = rise - radius

    theta_left = np.arctan2(0.0 - center_y, -half_span - center_x)
    theta_right = np.arctan2(0.0 - center_y, half_span - center_x)

    if n_segments % 2 != 0:
        n_segments += 1

    r_inner = radius - thickness / 2.0
    r_outer = radius + thickness / 2.0

    inner_nodes = []
    outer_nodes = []
    for i in range(n_segments + 1):
        t = i / n_segments
        theta = theta_left + (theta_right - theta_left) * t

        inner_nodes.append(
            [
                center_x + r_inner * np.cos(theta),
                center_y + r_inner * np.sin(theta),
            ]
        )
        outer_nodes.append(
            [
                center_x + r_outer * np.cos(theta),
                center_y + r_outer * np.sin(theta),
            ]
        )

    nodes = np.array(inner_nodes + outer_nodes, dtype=float)
    analysis = TrussAnalysis(nodes)

    def add_member(n1, n2, area_override=None):
        member_area = area if area_override is None else area_override
        analysis.add_element(
            TrussElement(n1, n2, youngs, member_area, sig_y0, hardening)
        )

    def inner_idx(i):
        return i

    def outer_idx(i):
        return len(inner_nodes) + i

    for i in range(n_segments):
        add_member(inner_idx(i), inner_idx(i + 1))
        add_member(outer_idx(i), outer_idx(i + 1))

    for i in range(n_segments + 1):
        add_member(inner_idx(i), outer_idx(i))

    diag_area = area * brace_area_scale if brace_area_scale > 0.0 else 0.0
    if add_diagonals or diag_area > 0.0:
        for i in range(n_segments):
            if i % 2 == 0:
                add_member(inner_idx(i), outer_idx(i + 1), area_override=diag_area)
            else:
                add_member(outer_idx(i), inner_idx(i + 1), area_override=diag_area)

    support_nodes = [
        inner_idx(0),
        outer_idx(0),
        inner_idx(n_segments),
        outer_idx(n_segments),
    ]
    for node_idx in support_nodes:
        analysis.add_constraint(node_idx, 0, 0.0)
        analysis.add_constraint(node_idx, 1, 0.0)

    mid_idx = n_segments // 2
    mid_inner = inner_idx(mid_idx)
    mid_outer = outer_idx(mid_idx)

    ref_load = -float(load_mag)
    analysis.add_load(mid_inner, 1, ref_load / 2.0)
    analysis.add_load(mid_outer, 1, ref_load / 2.0)

    if imperfection:
        analysis.add_load(mid_outer, 0, ref_load * imperfection)

    return analysis, mid_inner, mid_outer, ref_load


def run_arch_simulation(
    radius=100.0,
    rise=40.0,
    half_span=80.0,
    thickness=EQUIV_THICKNESS,
    n_segments=40,
    area=1.0,
    youngs=1.0e7,
    target_disp=80.0,
    target_dlam=0.5,
    fixed_ds=None,
    max_steps=600,
    tol=1.0e-6,
    max_iter=50,
    adaptive_ds=True,
    load_mag=100.0,
    imperfection=1.0e-4,
    add_diagonals=True,
    brace_area_scale=None,
    use_extrapolated_predictor=True,
    record_history=False,
):
    if brace_area_scale is None:
        brace_area_scale = 1.0 if add_diagonals else 0.01
        if not add_diagonals:
            print(
                "add_diagonals=False -> using brace_area_scale=0.01 "
                "for stability; set brace_area_scale=0.0 to disable."
            )
    analysis, mid_inner, mid_outer, ref_load = build_arch_truss(
        radius=radius,
        rise=rise,
        half_span=half_span,
        thickness=thickness,
        n_segments=n_segments,
        area=area,
        youngs=youngs,
        load_mag=load_mag,
        imperfection=imperfection,
        add_diagonals=add_diagonals,
        brace_area_scale=brace_area_scale,
    )

    if fixed_ds is None:
        ds, uF_norm, fext_norm_sq = _estimate_initial_ds(analysis, 0.0, target_dlam)
        ds_max = ds * 10.0
        ds_min = ds / 4096.0
        print(
            f"Initial estimate: uF_norm={uF_norm:.6e}, "
            f"fext_norm={np.sqrt(fext_norm_sq):.6e}, ds={ds:.6e}"
        )
    else:
        ds = float(fixed_ds)
        ds_max = ds
        ds_min = ds
        print(f"Fixed arc-length: ds={ds:.6e}")

    disp_history = []
    load_history = []
    nodes_history = []

    prev_du = None
    step = 0
    converged_prev = False

    while step < max_steps:
        success, _, lam, du = analysis.solve_arc_length_step_crisfield(
            ds,
            tol=tol,
            max_iter=max_iter,
            use_extrapolated_predictor=use_extrapolated_predictor,
        )
        if not success:
            failure = getattr(analysis, "_last_arc_failure", None)
            if failure:
                print(f"Arc-length failure: {failure}")
            if adaptive_ds:
                ds = max(ds * 0.5, ds_min)
                if ds <= ds_min:
                    print(f"Arc-length failed near step {step}")
                    break
                converged_prev = False
                continue
            break

        prev_du = du
        step += 1
        if adaptive_ds and converged_prev:
            ds = min(ds * 1.2, ds_max)
        converged_prev = True

        u_inner = analysis.current_nodes[mid_inner, 1] - analysis.nodes[mid_inner, 1]
        u_outer = analysis.current_nodes[mid_outer, 1] - analysis.nodes[mid_outer, 1]
        u_mid = 0.5 * (u_inner + u_outer)
        disp_history.append(-u_mid)
        load_history.append(lam * ref_load)
        if record_history:
            nodes_history.append(analysis.current_nodes.copy())

        if target_disp is not None and disp_history[-1] >= target_disp:
            break

    return (
        np.array(disp_history),
        np.array(load_history),
        analysis,
        nodes_history,
        mid_inner,
        mid_outer,
    )


def _plot_truss(ax, analysis, nodes_override=None, style="k-", alpha=1.0):
    nodes = analysis.current_nodes if nodes_override is None else nodes_override
    for element in analysis.elements:
        n1 = nodes[element.node1_idx]
        n2 = nodes[element.node2_idx]
        ax.plot([n1[0], n2[0]], [n1[1], n2[1]], style, linewidth=0.6, alpha=alpha)


def _select_snapshot_indices(disp, targets):
    indices = []
    if len(disp) == 0:
        return indices
    for d_target in targets:
        idx = int(np.argmin((disp - d_target) ** 2))
        if idx not in indices:
            indices.append(idx)
    return indices


def plot_arch_results(
    disp,
    load,
    analysis,
    nodes_history,
    mid_inner,
    mid_outer,
    output_path=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    ax0 = axes[0]
    _plot_truss(ax0, analysis, nodes_override=analysis.nodes, style="k-")
    crown = 0.5 * (analysis.nodes[mid_inner] + analysis.nodes[mid_outer])
    ax0.plot(crown[0], crown[1], "rv", markersize=8)
    ax0.set_title("(a) Initial Arch")
    ax0.set_xlabel("X (mm)")
    ax0.set_ylabel("Y (mm)")
    ax0.axis("equal")
    ax0.grid(True)

    ax1 = axes[1]
    ax1.plot(disp, -load, "k-")
    ax1.set_title("(b) Load-Displacement")
    ax1.set_xlabel("Central vertical displacement (mm)")
    ax1.set_ylabel("Downward load (N)")
    ax1.grid(True)

    ax2 = axes[2]
    if nodes_history:
        max_disp = float(np.max(disp)) if len(disp) else 0.0
        targets = [0.25 * max_disp, 0.5 * max_disp, 0.75 * max_disp, max_disp]
        indices = _select_snapshot_indices(disp, targets)
        for idx in indices:
            _plot_truss(
                ax2,
                analysis,
                nodes_override=nodes_history[idx],
                style="k-",
                alpha=0.7,
            )
        ax2.set_title("(c) Deformed shapes (snapshots)")
    else:
        ax2.text(0.5, 0.5, "Simulation Failed", ha="center")
        ax2.set_title("(c) Deformed shapes (snapshots)")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.axis("equal")
    ax2.grid(True)

    ax3 = axes[3]
    if nodes_history:
        _plot_truss(ax3, analysis, nodes_override=nodes_history[-1], style="k-")
        ax3.set_title("(d) Final deformed shape")
    else:
        ax3.text(0.5, 0.5, "Simulation Failed", ha="center")
        ax3.set_title("(d) Final deformed shape")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.axis("equal")
    ax3.grid(True)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def run_and_save(output_path):
    disp, load, analysis, hist, mid_inner, mid_outer = run_arch_simulation(
        record_history=True,
        add_diagonals=True,
    )
    if len(disp) > 0:
        print(
            f"Arch: steps={len(disp)}, disp_end={disp[-1]:.3f}, "
            f"load_max={np.max(-load):.3f}, load_min={np.min(-load):.3f}"
        )
    else:
        print("Arch: no converged steps.")

    plot_arch_results(
        disp,
        load,
        analysis,
        hist,
        mid_inner,
        mid_outer,
        output_path=output_path,
    )


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent
    run_and_save(output_dir / "figure_3_10_arch.pdf")
