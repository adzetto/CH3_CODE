from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

from truss_analysis import TrussAnalysis, TrussElement

PYIMP_ROOT = (
    Path.home() / "Desktop" / "kadapa" / "ArcLengthMethod" / "ArcLengthMethod-pyimp"
)
PYIMP_INPUT = (
    Path.home()
    / "Desktop"
    / "kadapa"
    / "ArcLengthMethod"
    / "oc"
    / "input_LeeFrame-nelem20.txt"
)
FLAGSHYP_ROOT = (
    Path.home()
    / "Desktop"
    / "CH3_CODE"
    / "Book_Codes"
    / "FLagSHyP_MatLab_Zipped"
    / "FLagSHyp_MatLab_on_the_Web"
    / "job_folder"
)
FLAGSHYP_TRUSS_ELASTIC = (
    FLAGSHYP_ROOT / "trussed_frame_elastic" / "trussed_frame_elastic.dat"
)
FLAGSHYP_TRUSS_PLASTIC = (
    FLAGSHYP_ROOT / "trussed_frame_plastic" / "trussed_frame_plastic.dat"
)


def generate_trussed_frame(
    clamped=False,
    right_support="x",
    support_mode="book",
    member_area=1.0,
):
    L = 120.0
    H = 120.0
    D = 2.0
    spacing = 2.0

    n_vertical = int(H / spacing)
    n_horizontal = int(L / spacing)

    nodes = []
    elements = []
    node_map = {}

    def get_node_idx(x, y):
        key = (round(x, 4), round(y, 4))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append([x, y])
        return node_map[key]

    # Vertical leg nodes
    for i in range(n_vertical + 1):
        y = i * spacing
        get_node_idx(0.0, y)
        get_node_idx(D, y)

    # Horizontal leg nodes
    for i in range(n_horizontal + 1):
        x = i * spacing
        get_node_idx(x, H - D)
        get_node_idx(x, H)

    # Vertical leg elements
    for i in range(n_vertical):
        y_bot = i * spacing
        y_top = (i + 1) * spacing

        n1 = get_node_idx(0.0, y_bot)
        n2 = get_node_idx(D, y_bot)
        n3 = get_node_idx(D, y_top)
        n4 = get_node_idx(0.0, y_top)

        elements.append((n1, n4))
        elements.append((n2, n3))
        elements.append((n1, n2))
        if i == n_vertical - 1:
            elements.append((n4, n3))
        elements.append((n1, n3))
        elements.append((n2, n4))

    # Horizontal leg elements
    start_idx = int(D / spacing)
    for i in range(start_idx, n_horizontal):
        x_left = i * spacing
        x_right = (i + 1) * spacing

        n1 = get_node_idx(x_left, H - D)
        n2 = get_node_idx(x_left, H)
        n3 = get_node_idx(x_right, H)
        n4 = get_node_idx(x_right, H - D)

        elements.append((n2, n3))
        elements.append((n1, n4))
        elements.append((n4, n3))
        elements.append((n1, n3))
        elements.append((n2, n4))

    # Load node
    load_node_idx = get_node_idx(24.0, 120.0)

    analysis = TrussAnalysis(nodes)

    # Material properties
    E = 210000.0
    A = member_area
    sig_y0 = 2500.0
    H_mod = 1.0

    for el_nodes in elements:
        el = TrussElement(el_nodes[0], el_nodes[1], E, A, sig_y0, H_mod)
        analysis.add_element(el)

    # Boundary conditions
    if support_mode == "book":
        bottom_nodes = [get_node_idx(0.0, 0.0)]
        right_nodes = [get_node_idx(L, H)]
    elif support_mode == "edge":
        bottom_nodes = [get_node_idx(0.0, 0.0), get_node_idx(D, 0.0)]
        right_nodes = [get_node_idx(L, H - D), get_node_idx(L, H)]
    else:
        raise ValueError("support_mode must be 'book' or 'edge'")

    for node_idx in bottom_nodes:
        analysis.add_constraint(node_idx, 0, 0.0)
        analysis.add_constraint(node_idx, 1, 0.0)

    if clamped:
        for node_idx in right_nodes:
            analysis.add_constraint(node_idx, 0, 0.0)
            analysis.add_constraint(node_idx, 1, 0.0)
    else:
        if right_support == "x":
            for node_idx in right_nodes:
                analysis.add_constraint(node_idx, 0, 0.0)
        elif right_support == "y":
            for node_idx in right_nodes:
                analysis.add_constraint(node_idx, 1, 0.0)
        else:
            raise ValueError("right_support must be 'x' or 'y'")

    return analysis, load_node_idx


def _boundary_code_to_dofs(code, ndime):
    mapping = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [0, 1, 0],
        3: [1, 1, 0],
        4: [0, 0, 1],
        5: [1, 0, 1],
        6: [0, 1, 1],
        7: [1, 1, 1],
    }
    dofs = mapping.get(code, [0, 0, 0])
    return dofs[:ndime]


def _material_property_count(mat_type):
    if mat_type == 2:
        return 6
    if mat_type == 4:
        return 4
    if mat_type == 17:
        return 5
    return 3


def parse_flagshyp_truss_input(path):
    path = Path(path)
    content = path.read_text(errors="ignore").replace("\x00", "")
    raw_lines = content.splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]

    idx = 0
    title = lines[idx]
    idx += 1
    element_type = lines[idx]
    idx += 1

    n_nodes = int(lines[idx].split()[0])
    idx += 1

    if element_type == "truss2":
        ndime = 3
        n_face_nodes_elem = 0
    else:
        ndime = 2
        n_face_nodes_elem = 0

    node_ids = []
    bc_codes = []
    nodes = []
    for _ in range(n_nodes):
        parts = lines[idx].split()
        node_id = int(parts[0])
        code = int(parts[1])
        coords = [float(val) for val in parts[2 : 2 + ndime]]
        node_ids.append(node_id)
        bc_codes.append(code)
        nodes.append(coords)
        idx += 1

    n_elem = int(lines[idx].split()[0])
    idx += 1

    elements = []
    for _ in range(n_elem):
        parts = lines[idx].split()
        if not parts:
            idx += 1
            continue
        elem_id = int(parts[0])
        mat_id = int(parts[1])
        n1 = int(parts[2])
        n2 = int(parts[3])
        elements.append((elem_id, mat_id, n1, n2))
        idx += 1

    tail_tokens = " ".join(lines[idx:]).split()
    tidx = 0

    n_mats = int(tail_tokens[tidx])
    tidx += 1
    materials = {}
    for _ in range(n_mats):
        mat_id = int(tail_tokens[tidx])
        mat_type = int(tail_tokens[tidx + 1])
        tidx += 2
        prop_count = _material_property_count(mat_type)
        props = [float(tail_tokens[tidx + i]) for i in range(prop_count)]
        tidx += prop_count
        materials[mat_id] = {"type": mat_type, "props": props}

    n_point_loads = int(tail_tokens[tidx])
    tidx += 1
    n_prescribed = int(tail_tokens[tidx])
    tidx += 1
    n_pressure = int(tail_tokens[tidx])
    tidx += 1
    gravity = [float(tail_tokens[tidx + i]) for i in range(ndime)]
    tidx += ndime

    point_loads = []
    for _ in range(n_point_loads):
        node_id = int(tail_tokens[tidx])
        tidx += 1
        comps = [float(tail_tokens[tidx + i]) for i in range(ndime)]
        tidx += ndime
        point_loads.append((node_id, comps))

    prescribed = []
    for _ in range(n_prescribed):
        node_id = int(tail_tokens[tidx])
        local_dof = int(tail_tokens[tidx + 1])
        value = float(tail_tokens[tidx + 2])
        tidx += 3
        prescribed.append((node_id, local_dof, value))

    pressure = []
    if n_pressure and n_face_nodes_elem:
        for _ in range(n_pressure):
            elem_id = int(tail_tokens[tidx])
            tidx += 1
            face_nodes = [int(tail_tokens[tidx + i]) for i in range(n_face_nodes_elem)]
            tidx += n_face_nodes_elem
            value = float(tail_tokens[tidx])
            tidx += 1
            pressure.append((elem_id, face_nodes, value))

    control = {
        "nincr": int(tail_tokens[tidx]),
        "xlmax": float(tail_tokens[tidx + 1]),
        "dlamb": float(tail_tokens[tidx + 2]),
        "miter": int(tail_tokens[tidx + 3]),
        "cnorm": float(tail_tokens[tidx + 4]),
        "searc": float(tail_tokens[tidx + 5]),
        "arcln": float(tail_tokens[tidx + 6]),
        "incout": float(tail_tokens[tidx + 7]),
        "itarget": float(tail_tokens[tidx + 8]),
        "nwant": int(tail_tokens[tidx + 9]),
        "iwant": int(tail_tokens[tidx + 10]),
    }

    return {
        "title": title,
        "element_type": element_type,
        "ndime": ndime,
        "node_ids": node_ids,
        "bc_codes": bc_codes,
        "nodes": np.array(nodes, dtype=float),
        "elements": elements,
        "materials": materials,
        "point_loads": point_loads,
        "prescribed": prescribed,
        "gravity": gravity,
        "pressure": pressure,
        "control": control,
    }


def build_analysis_from_flagshyp(path, clamped=False):
    data = parse_flagshyp_truss_input(path)
    nodes = data["nodes"]
    analysis = TrussAnalysis(nodes)

    for _, mat_id, n1, n2 in data["elements"]:
        props = data["materials"][mat_id]["props"]
        E = props[1]
        A = props[3]
        sig_y0 = props[4]
        H_mod = props[5]
        el = TrussElement(n1 - 1, n2 - 1, E, A, sig_y0, H_mod)
        analysis.add_element(el)

    for idx, code in enumerate(data["bc_codes"]):
        dofs = _boundary_code_to_dofs(code, data["ndime"])
        for axis, fixed in enumerate(dofs):
            if fixed:
                analysis.add_constraint(idx, axis, 0.0)

    if clamped:
        coords = nodes
        min_y = np.min(coords[:, 1])
        max_x = np.max(coords[:, 0])
        tol = 1.0e-6
        for idx, (x, y, *_rest) in enumerate(coords):
            if abs(y - min_y) <= tol or abs(x - max_x) <= tol:
                analysis.add_constraint(idx, 0, 0.0)
                analysis.add_constraint(idx, 1, 0.0)

    for node_id, dof, value in data["prescribed"]:
        analysis.set_displacement(node_id - 1, dof - 1, value)

    for node_id, comps in data["point_loads"]:
        for axis, val in enumerate(comps):
            if val != 0.0:
                analysis.add_load(node_id - 1, axis, val)

    output_node = data["control"]["nwant"] - 1
    output_axis = data["control"]["iwant"] - 1

    return analysis, output_node, output_axis, data


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


def run_frame_simulation(
    case_type="Elastic",
    clamped=False,
    method="arc_length",
    record_history=False,
    load_mag=100.0,
    psi=0.0,
    target_disp=98.0,
    target_dlam=0.01,
    bootstrap_disp=0.0,
    max_steps=1500,
    tol=1e-6,
    max_iter=60,
    right_support="x",
    support_mode="book",
    member_area=1.0,
    fixed_ds=None,
    adaptive_ds=True,
    input_path=None,
    use_control=True,
    arc_length_scheme="auto",
    force_limit=None,
    use_extrapolated_predictor=True,
):
    import gc

    gc.collect()
    print(f"Running Frame Simulation: {case_type}, Clamped={clamped}, Method={method}")
    if input_path is None:
        analysis, load_node = generate_trussed_frame(
            clamped,
            right_support=right_support,
            support_mode=support_mode,
            member_area=member_area,
        )
        load_axis = 1
        control = {}
    else:
        analysis, load_node, load_axis, data = build_analysis_from_flagshyp(
            input_path, clamped=clamped
        )
        control = data["control"]
        if use_control:
            if max_steps == 1500:
                max_steps = int(control["nincr"]) * 5
            max_iter = int(control["miter"])
            tol = float(control["cnorm"])
            if case_type == "Plastic":
                if fixed_ds is None:
                    target_dlam = float(control["dlamb"])
            else:
                if fixed_ds is None:
                    fixed_ds = float(control["arcln"])
            analysis._arc_dlamb = float(control["dlamb"])
            if force_limit is None:
                force_limit = 700.0

    for el in analysis.elements:
        if case_type == "Elastic":
            el.sig_y0 = 1e20

    res_force = []
    res_disp = []
    nodes_history = []
    ref_load = analysis.loads.get((load_node, load_axis), -load_mag)

    if method == "arc_length":
        if bootstrap_disp > 0.0:
            analysis.set_displacement(load_node, load_axis, -bootstrap_disp)
            success, _ = analysis.solve_step(tol=1e-6, max_iter=50)
            analysis.constrained_dofs.pop((load_node, load_axis), None)
            if not success:
                print("Bootstrap displacement step failed.")
                return np.array(res_disp), np.array(res_force), analysis, nodes_history

        if input_path is None:
            analysis.add_load(load_node, load_axis, -load_mag)

        if fixed_ds is None:
            ds, uF_norm, fext_norm_sq = _estimate_initial_ds(analysis, psi, target_dlam)
            ds_max = ds * 10.0
            ds_min = ds / 4096.0
        else:
            ds = float(fixed_ds)
            fext_norm_sq = float(
                np.dot(
                    np.array(list(analysis.loads.values())),
                    np.array(list(analysis.loads.values())),
                )
            )
            if input_path is not None and use_control and adaptive_ds:
                ds_max = ds * 10.0
                ds_min = ds / 100.0
            else:
                ds_max = ds
                ds_min = ds
        prev_du = None
        step = 0
        converged_prev = False
        if fixed_ds is None:
            print(
                f"Initial estimate: uF_norm={uF_norm:.6e}, "
                f"fext_norm={np.sqrt(fext_norm_sq):.6e}, ds={ds:.6e}"
            )
        else:
            print(f"Fixed arc-length: ds={ds:.6e}")

        while step < max_steps:
            if arc_length_scheme == "auto":
                use_crisfield = input_path is not None
            else:
                use_crisfield = arc_length_scheme == "crisfield"

            if use_crisfield:
                success, F_int, lam, du = analysis.solve_arc_length_step_crisfield(
                    ds,
                    tol=tol,
                    max_iter=max_iter,
                    use_extrapolated_predictor=use_extrapolated_predictor,
                )
            else:
                success, F_int, lam, du = analysis.solve_arc_length_step(
                    ds,
                    prev_du=prev_du,
                    psi=psi,
                    tol=tol,
                    max_iter=max_iter,
                    use_extrapolated_predictor=use_extrapolated_predictor,
                )

            if not success:
                failure = getattr(analysis, "_last_arc_failure", None)
                details = getattr(analysis, "_last_arc_details", None)
                if failure:
                    msg = f"Arc-length failure: {failure}"
                    if details:
                        msg += f" {details}"
                    print(msg)
                if adaptive_ds:
                    if input_path is not None and use_control:
                        ds = max(ds / 10.0, ds_min)
                    elif converged_prev:
                        ds = max(ds * 0.5, ds_min)
                    else:
                        ds = max(ds * 0.25, ds_min)
                    if ds <= ds_min:
                        print(f"Arc-Length failed near step {step}")
                        break
                    converged_prev = False
                    continue
                print(f"Arc-Length failed near step {step}")
                break

            prev_du = du
            step += 1
            if adaptive_ds:
                if use_crisfield:
                    new_ds = getattr(analysis, "_arc_ds", None)
                    if new_ds and new_ds > 0.0:
                        ds = new_ds
                if input_path is not None and use_control:
                    itarget = control.get("itarget", None)
                    last_iters = getattr(analysis, "_last_arc_iters", None)
                    if itarget and last_iters:
                        ds = ds * (float(itarget) / float(last_iters)) ** 0.7
                        ds = min(max(ds, ds_min), ds_max)
                elif converged_prev:
                    ds = min(ds * 1.2, ds_max)
            converged_prev = True

            current_u_y = (
                analysis.current_nodes[load_node, load_axis]
                - analysis.nodes[load_node, load_axis]
            )
            current_disp = -current_u_y

            res_force.append(lam * ref_load)
            res_disp.append(current_disp)
            if record_history:
                nodes_history.append(analysis.current_nodes.copy())

            if force_limit is not None and abs(res_force[-1]) > force_limit:
                res_force.pop()
                res_disp.pop()
                if record_history and nodes_history:
                    nodes_history.pop()
                break
            if target_disp is not None and current_disp >= target_disp:
                break

    else:
        steps = 50
        max_disp = -100.0
        displacements = np.linspace(0, max_disp, steps)

        for i, disp in enumerate(displacements):
            if i == 0:
                continue

            analysis.set_displacement(load_node, load_axis, disp)
            success, F_int = analysis.solve_step(tol=1e-3, max_iter=20)

            if not success:
                print(f"Convergence failed at step {i}, displacement {disp:.2f}")
                break

            fy = F_int[load_node * len(analysis.nodes[0]) + load_axis]
            res_force.append(-fy)
            res_disp.append(-disp)
            if record_history:
                nodes_history.append(analysis.current_nodes.copy())

    return np.array(res_disp), np.array(res_force), analysis, nodes_history


def select_snapshot_indices(disp, force_plot, targets):
    indices = []
    if len(disp) == 0:
        return indices
    for d_target, f_target in targets:
        dist = (disp - d_target) ** 2 + (force_plot - f_target) ** 2
        indices.append(int(np.argmin(dist)))
    return indices


def plot_truss_on_axis(
    ax, analysis, title, style="b-", nodes_override=None, nodes_style=None
):
    nodes = analysis.current_nodes if nodes_override is None else nodes_override
    coords = nodes[:, :2] if nodes.shape[1] > 2 else nodes
    for element in analysis.elements:
        n1 = coords[element.node1_idx]
        n2 = coords[element.node2_idx]
        ax.plot([n1[0], n2[0]], [n1[1], n2[1]], style, linewidth=0.5)

    if title:
        ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.axis("equal")
    ax.grid(True)


def plot_results(
    d_el,
    f_el,
    analysis_el,
    hist_el,
    d_pl,
    f_pl,
    analysis_pl,
    hist_pl,
    support_mode="book",
    analysis_initial=None,
    load_node_idx=None,
    output_path=None,
):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    if analysis_initial is None:
        analysis_initial, load_node_idx = generate_trussed_frame(
            support_mode=support_mode
        )
    plot_truss_on_axis(
        plt.gca(), analysis_initial, "(a) Initial Configuration", style="k-"
    )
    if load_node_idx is not None:
        ln = analysis_initial.nodes[load_node_idx]
        plt.plot(ln[0], ln[1], "rv", markersize=10, label="Load")
        plt.legend()
    plt.gca().set_xlim(-10, 130)
    plt.gca().set_ylim(-10, 130)

    plt.subplot(2, 2, 2)
    if len(d_el) > 0:
        plt.plot(d_el, -f_el, "k--", label="Elastic")
    if len(d_pl) > 0:
        plt.plot(d_pl, -f_pl, "k-", label="Plastic")

    plt.title("(b) Load-Displacement")
    plt.xlabel("Downward Displacement (mm)")
    plt.ylabel("Downward Load (N)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 100)
    plt.ylim(-400, 700)

    plt.subplot(2, 2, 3)
    if len(d_el) > 0 and hist_el:
        elastic_targets = [(21.0, 402.0), (50.0, 556.0), (58.0, -285.0), (94.0, 592.0)]
        elastic_indices = select_snapshot_indices(d_el, -f_el, elastic_targets)
        for idx in elastic_indices:
            plot_truss_on_axis(
                plt.gca(), analysis_el, "", style="k-", nodes_override=hist_el[idx]
            )
        plt.title("(c) Deformed Elastic (snapshots)")
    else:
        plt.text(0.5, 0.5, "Simulation Failed", ha="center")
    plt.gca().set_xlim(-10, 130)
    plt.gca().set_ylim(-10, 130)

    plt.subplot(2, 2, 4)
    if len(d_pl) > 0 and hist_pl:
        plastic_targets = [(32.0, 222.0)]
        plastic_indices = select_snapshot_indices(d_pl, -f_pl, plastic_targets)
        idx = plastic_indices[0] if plastic_indices else -1
        plot_truss_on_axis(
            plt.gca(),
            analysis_pl,
            "(d) Deformed Plastic (snapshot)",
            style="k-",
            nodes_override=hist_pl[idx],
        )
    else:
        plt.text(0.5, 0.5, "Simulation Failed", ha="center")
    plt.gca().set_xlim(-10, 130)
    plt.gca().set_ylim(-10, 130)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


def _import_pyimp(pyimp_root):
    if "pyimp" in sys.modules:
        return sys.modules["pyimp"]
    pyimp_root = Path(pyimp_root)
    if not pyimp_root.exists():
        raise FileNotFoundError(f"ArcLengthMethod-pyimp not found: {pyimp_root}")
    spec = importlib.util.spec_from_file_location(
        "pyimp",
        pyimp_root / "__init__.py",
        submodule_search_locations=[str(pyimp_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pyimp"] = module
    spec.loader.exec_module(module)
    return module


def _apply_clamped_rotations(model):
    fixed = set(range(model.neq)) - set(model.free_dofs.tolist())
    support_nodes = {dof // model.ndof for dof in fixed}
    for node_id in support_nodes:
        fixed.add(node_id * model.ndof + 2)
    free = np.array([idx for idx in range(model.neq) if idx not in fixed], dtype=int)
    return replace(model, free_dofs=free)


def run_pyimp_frame(
    input_path=PYIMP_INPUT,
    clamped=False,
    load_scale=100.0,
    max_iters=20,
    tol=1.0e-6,
    quiet=True,
):
    _import_pyimp(PYIMP_ROOT)
    from pyimp.io import load_model
    from pyimp.arc_length import run_arclength
    from pyimp.solver import ArcLengthSettings

    model = load_model(input_path)
    if clamped:
        model = _apply_clamped_rotations(model)

    settings = ArcLengthSettings(tol=tol)
    result = run_arclength(
        model, max_iters=max_iters, settings=settings, verbose=not quiet
    )

    disp_history = np.array(result.disp_history)
    load_factors = np.array(result.load_factors)
    output_history = np.array(result.output_history)

    if output_history.shape[1] < 2:
        raise ValueError("Expected at least two output DOFs for Lee frame.")

    ref_load = float(np.sum(model.fext[model.force_dofs]))
    if ref_load == 0.0:
        ref_load = 1.0

    disp = -output_history[:, 1]
    load = -load_factors * ref_load * load_scale

    return disp, load, model, disp_history, load_factors


def plot_pyimp_results(disp, load, model, disp_history, title_suffix=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def plot_model(ax, coords, elem_conn, title, nodes=None, style="k-"):
        pts = coords if nodes is None else nodes
        for elem in elem_conn:
            n1 = int(elem[2])
            n2 = int(elem[3])
            x = [pts[n1, 0], pts[n2, 0]]
            y = [pts[n1, 1], pts[n2, 1]]
            ax.plot(x, y, style, linewidth=0.6)
        if title:
            ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.axis("equal")
        ax.grid(True)

    plot_model(
        axes[0, 0],
        model.coords,
        model.elem_conn,
        f"(a) Initial Configuration{title_suffix}",
    )

    axes[0, 1].plot(disp, load, "k-")
    axes[0, 1].set_title(f"(b) Load-Displacement{title_suffix}")
    axes[0, 1].set_xlabel("Downward Displacement (mm)")
    axes[0, 1].set_ylabel("Downward Load (N)")
    axes[0, 1].grid(True)
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].set_ylim(-400, 700)

    elastic_targets = [(21.0, 402.0), (50.0, 556.0), (58.0, -285.0), (94.0, 592.0)]
    if len(disp) > 0:
        elastic_indices = select_snapshot_indices(disp, load, elastic_targets)
        for idx in elastic_indices:
            disp_nodes = disp_history[idx].reshape((model.nnode, model.ndof))
            nodes = model.coords.copy()
            nodes[:, 0] += disp_nodes[:, 0]
            nodes[:, 1] += disp_nodes[:, 1]
            plot_model(axes[1, 0], model.coords, model.elem_conn, "", nodes=nodes)
        axes[1, 0].set_title(f"(c) Deformed Elastic (snapshots){title_suffix}")
    else:
        axes[1, 0].text(0.5, 0.5, "Simulation Failed", ha="center")

    axes[1, 1].text(0.5, 0.5, "Plastic not available in pyimp model", ha="center")
    axes[1, 1].set_title(f"(d) Deformed Plastic{title_suffix}")
    axes[1, 1].set_xlabel("X (mm)")
    axes[1, 1].set_ylabel("Y (mm)")
    axes[1, 1].grid(True)
    axes[1, 1].set_xlim(-10, 130)
    axes[1, 1].set_ylim(-10, 130)

    plt.tight_layout()
    plt.show()


def run_and_plot(
    right_support="x",
    clamped=False,
    support_mode="book",
    member_area=1.0,
    input_elastic=None,
    input_plastic=None,
    use_control=True,
    output_path=None,
):
    d_el, f_el, analysis_el, hist_el = run_frame_simulation(
        "Elastic",
        clamped=clamped,
        method="arc_length",
        record_history=True,
        right_support=right_support,
        support_mode=support_mode,
        member_area=member_area,
        input_path=input_elastic,
        use_control=use_control,
        adaptive_ds=False,
    )
    d_pl, f_pl, analysis_pl, hist_pl = run_frame_simulation(
        "Plastic",
        clamped=clamped,
        method="arc_length",
        record_history=True,
        right_support=right_support,
        support_mode=support_mode,
        member_area=member_area,
        input_path=input_plastic,
        use_control=use_control,
        target_disp=32.0,
        max_steps=400,
    )
    if len(d_el) > 0:
        print(
            f"Elastic: steps={len(d_el)}, disp_end={d_el[-1]:.3f}, "
            f"load_max={np.max(-f_el):.3f}, load_min={np.min(-f_el):.3f}"
        )
    else:
        print("Elastic: no converged steps.")
    if len(d_pl) > 0:
        print(
            f"Plastic: steps={len(d_pl)}, disp_end={d_pl[-1]:.3f}, "
            f"load_max={np.max(-f_pl):.3f}, load_min={np.min(-f_pl):.3f}"
        )
    else:
        print("Plastic: no converged steps.")
    analysis_initial = None
    load_node_idx = None
    if input_elastic is not None:
        analysis_initial, load_node_idx, _, _ = build_analysis_from_flagshyp(
            input_elastic, clamped=clamped
        )

    plot_results(
        d_el,
        f_el,
        analysis_el,
        hist_el,
        d_pl,
        f_pl,
        analysis_pl,
        hist_pl,
        support_mode=support_mode,
        analysis_initial=analysis_initial,
        load_node_idx=load_node_idx,
        output_path=output_path,
    )


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent
    run_and_plot(
        clamped=False,
        input_elastic=FLAGSHYP_TRUSS_ELASTIC,
        input_plastic=FLAGSHYP_TRUSS_PLASTIC,
        use_control=True,
        output_path=output_dir / "figure_3_9_trussed_frame.pdf",
    )
    run_and_plot(
        clamped=True,
        input_elastic=FLAGSHYP_TRUSS_ELASTIC,
        input_plastic=FLAGSHYP_TRUSS_PLASTIC,
        use_control=True,
        output_path=output_dir / "figure_3_9_trussed_frame_clamped.pdf",
    )
