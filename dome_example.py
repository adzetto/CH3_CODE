from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from truss_analysis import TrussAnalysis, TrussElement


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


def build_shallow_dome(
    outer_radius=50.0,
    inner_radius=25.0,
    outer_z=0.0,
    inner_z=6.216,
    apex_z=8.216,
    n_sectors=24,
    area=1.0,
    youngs=8.0e7,
    sig_y0=1.0e20,
    hardening=1.0,
    load_mag=1.0,
    imperfection=1.0e-4,
):
    if n_sectors % 2 != 0:
        n_sectors += 1

    angles_outer = np.linspace(0.0, 2.0 * np.pi, n_sectors, endpoint=False)
    angles_inner = angles_outer + (np.pi / n_sectors)

    outer_nodes = np.column_stack(
        (
            outer_radius * np.cos(angles_outer),
            outer_radius * np.sin(angles_outer),
            np.full(n_sectors, outer_z),
        )
    )
    inner_nodes = np.column_stack(
        (
            inner_radius * np.cos(angles_inner),
            inner_radius * np.sin(angles_inner),
            np.full(n_sectors, inner_z),
        )
    )
    apex_node = np.array([[0.0, 0.0, apex_z]])

    nodes = np.vstack([outer_nodes, inner_nodes, apex_node])
    analysis = TrussAnalysis(nodes)

    def add_member(n1, n2):
        analysis.add_element(TrussElement(n1, n2, youngs, area, sig_y0, hardening))

    def outer_idx(i):
        return i % n_sectors

    def inner_idx(i):
        return n_sectors + (i % n_sectors)

    apex_idx = 2 * n_sectors

    for i in range(n_sectors):
        add_member(outer_idx(i), outer_idx(i + 1))
        add_member(inner_idx(i), inner_idx(i + 1))

    for i in range(n_sectors):
        add_member(inner_idx(i), outer_idx(i))
        add_member(inner_idx(i), outer_idx(i + 1))

    for i in range(n_sectors):
        add_member(apex_idx, inner_idx(i))

    for i in range(n_sectors):
        analysis.add_constraint(outer_idx(i), 0, 0.0)
        analysis.add_constraint(outer_idx(i), 1, 0.0)
        analysis.add_constraint(outer_idx(i), 2, 0.0)

    ref_load = -float(load_mag)
    analysis.add_load(apex_idx, 2, ref_load)
    if imperfection:
        analysis.add_load(apex_idx, 0, ref_load * imperfection)

    return analysis, apex_idx, ref_load


def run_dome_simulation(
    outer_radius=50.0,
    inner_radius=25.0,
    outer_z=0.0,
    inner_z=6.216,
    apex_z=8.216,
    n_sectors=24,
    area=1.0,
    youngs=8.0e7,
    load_mag=1.0,
    target_disp=50.0,
    target_dlam=0.1,
    fixed_ds=None,
    max_steps=800,
    tol=1.0e-6,
    max_iter=50,
    adaptive_ds=True,
    imperfection=1.0e-4,
    use_extrapolated_predictor=True,
    record_history=False,
):
    analysis, apex_idx, ref_load = build_shallow_dome(
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        outer_z=outer_z,
        inner_z=inner_z,
        apex_z=apex_z,
        n_sectors=n_sectors,
        area=area,
        youngs=youngs,
        load_mag=load_mag,
        imperfection=imperfection,
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

    step = 0
    converged_prev = False

    while step < max_steps:
        success, _, lam, _ = analysis.solve_arc_length_step_crisfield(
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

        step += 1
        if adaptive_ds and converged_prev:
            ds = min(ds * 1.2, ds_max)
        converged_prev = True

        u_apex = (
            analysis.current_nodes[apex_idx, 2] - analysis.nodes[apex_idx, 2]
        )
        disp_history.append(-u_apex)
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
        apex_idx,
    )


def _plot_truss_projection(ax, analysis, nodes_override=None, axis="xz", alpha=1.0):
    nodes = analysis.current_nodes if nodes_override is None else nodes_override
    axis_map = {"x": 0, "y": 1, "z": 2}
    a = axis_map[axis[0]]
    b = axis_map[axis[1]]
    for element in analysis.elements:
        n1 = nodes[element.node1_idx]
        n2 = nodes[element.node2_idx]
        ax.plot([n1[a], n2[a]], [n1[b], n2[b]], "k-", linewidth=0.5, alpha=alpha)


def _select_snapshot_indices(disp, targets):
    indices = []
    if len(disp) == 0:
        return indices
    for d_target in targets:
        idx = int(np.argmin((disp - d_target) ** 2))
        if idx not in indices:
            indices.append(idx)
    return indices


def plot_dome_results(
    disp,
    load,
    analysis,
    nodes_history,
    output_path=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    ax0 = axes[0]
    _plot_truss_projection(ax0, analysis, nodes_override=analysis.nodes, axis="xz")
    ax0.set_title("(a) Initial Dome (X-Z projection)")
    ax0.set_xlabel("X (mm)")
    ax0.set_ylabel("Z (mm)")
    ax0.axis("equal")
    ax0.grid(True)

    ax1 = axes[1]
    ax1.plot(disp, -load, "k-")
    ax1.set_title("(b) Load-Displacement")
    ax1.set_xlabel("Apex vertical displacement (mm)")
    ax1.set_ylabel("Downward load (N)")
    ax1.grid(True)

    ax2 = axes[2]
    if nodes_history:
        max_disp = float(np.max(disp)) if len(disp) else 0.0
        targets = [0.25 * max_disp, 0.5 * max_disp, 0.75 * max_disp, max_disp]
        indices = _select_snapshot_indices(disp, targets)
        for idx in indices:
            _plot_truss_projection(
                ax2, analysis, nodes_override=nodes_history[idx], axis="xz", alpha=0.7
            )
        ax2.set_title("(c) Deformed shapes (snapshots)")
    else:
        ax2.text(0.5, 0.5, "Simulation Failed", ha="center")
        ax2.set_title("(c) Deformed shapes (snapshots)")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Z (mm)")
    ax2.axis("equal")
    ax2.grid(True)

    ax3 = axes[3]
    if nodes_history:
        _plot_truss_projection(
            ax3, analysis, nodes_override=nodes_history[-1], axis="xz"
        )
        ax3.set_title("(d) Final deformed shape")
    else:
        ax3.text(0.5, 0.5, "Simulation Failed", ha="center")
        ax3.set_title("(d) Final deformed shape")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.axis("equal")
    ax3.grid(True)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def run_and_save(output_path):
    disp, load, analysis, hist, _ = run_dome_simulation(
        fixed_ds=0.1,
        adaptive_ds=False,
        target_disp=12.0,
        max_steps=400,
        record_history=True,
    )
    if len(disp) > 0:
        print(
            f"Dome: steps={len(disp)}, disp_end={disp[-1]:.3f}, "
            f"load_max={np.max(-load):.3f}, load_min={np.min(-load):.3f}"
        )
    else:
        print("Dome: no converged steps.")

    plot_dome_results(disp, load, analysis, hist, output_path=output_path)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent
    run_and_save(output_dir / "figure_3_11_shallow_dome.pdf")
