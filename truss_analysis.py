import numpy as np


class TrussElement:
    def __init__(self, node1_idx, node2_idx, E, A, sig_y0, H):
        self.node1_idx = node1_idx
        self.node2_idx = node2_idx
        self.E = E
        self.A = A
        self.sig_y0 = sig_y0
        self.H = H

        # Initial coordinates
        self.X1 = None
        self.X2 = None
        self.L = None
        self.V = None

        # History variables
        self.ep_n = 0.0  # Plastic strain
        self.alpha_n = 0.0  # Accumulated plastic strain

    def initialize(self, nodes):
        self.X1 = nodes[self.node1_idx]
        self.X2 = nodes[self.node2_idx]
        self.L = np.linalg.norm(self.X2 - self.X1)
        self.V = self.A * self.L

    def get_force_and_stiffness(self, current_nodes, update_history=False):
        # 1. Geometric Update
        x1 = current_nodes[self.node1_idx]
        x2 = current_nodes[self.node2_idx]

        dx = x2 - x1
        l = np.linalg.norm(dx)
        min_len = self.L * 1.0e-8
        if not np.isfinite(l) or l <= min_len:
            raise ValueError("Invalid element length during update.")
        n = dx / l

        # Logarithmic strain
        eps_n1 = np.log(l / self.L)

        # 2. Elastic Predictor
        tau_trial = self.E * (eps_n1 - self.ep_n)
        phi_trial = abs(tau_trial) - (self.sig_y0 + self.H * self.alpha_n)

        # 3. Plastic Corrector
        if phi_trial <= 0:
            # Elastic Case
            tau_n1 = tau_trial
            E_alg = self.E
            ep_n1 = self.ep_n
            alpha_n1 = self.alpha_n
        else:
            # Plastic Case
            delta_gamma = phi_trial / (self.E + self.H)
            sign_tau = np.sign(tau_trial)

            tau_n1 = tau_trial - self.E * delta_gamma * sign_tau
            ep_n1 = self.ep_n + delta_gamma * sign_tau
            alpha_n1 = self.alpha_n + delta_gamma

            E_alg = (self.E * self.H) / (self.E + self.H)

        if update_history:
            self.ep_n = ep_n1
            self.alpha_n = alpha_n1

        # 4. Internal Force Vector
        T = (self.V * tau_n1) / l
        Tb = T * n
        f_int_1 = -Tb
        f_int_2 = Tb

        # 5. Tangent Stiffness Matrix (match FLagSHyP truss formulation)
        k_coeff = (self.V / (l**2)) * (E_alg - 2.0 * tau_n1)
        k_sub = k_coeff * np.outer(n, n) + (T / l) * np.eye(len(n))

        return f_int_1, f_int_2, k_sub, tau_n1, ep_n1, eps_n1


class TrussAnalysis:
    def __init__(self, nodes):
        self.nodes = np.array(nodes, dtype=float)
        self.current_nodes = self.nodes.copy()
        self.elements = []
        self.constrained_dofs = {}
        self.loads = {}
        self.lam = 0.0
        self._last_du = None
        self._last_dlam = None
        self._last_arc_failure = None
        self._last_arc_details = None
        self._last_arc_iters = None
        self._arc_xincr = None
        self._arc_ds = None
        self._arc_step = 0
        self._arc_dlamb = None

    def add_element(self, element):
        element.initialize(self.nodes)
        self.elements.append(element)

    def add_constraint(self, node_idx, axis_idx, value=0.0):
        self.constrained_dofs[(node_idx, axis_idx)] = value

    def add_load(self, node_idx, axis_idx, value):
        self.loads[(node_idx, axis_idx)] = value

    def set_displacement(self, node_idx, axis_idx, value):
        self.constrained_dofs[(node_idx, axis_idx)] = value

    def solve_step(self, tol=1e-6, max_iter=20):
        num_nodes = len(self.nodes)
        dof_per_node = len(self.nodes[0])
        total_dofs = num_nodes * dof_per_node

        free_dofs = []
        for i in range(num_nodes):
            for j in range(dof_per_node):
                if (i, j) not in self.constrained_dofs:
                    free_dofs.append(i * dof_per_node + j)

        for (n_idx, a_idx), val in self.constrained_dofs.items():
            self.current_nodes[n_idx, a_idx] = self.nodes[n_idx, a_idx] + val

        for k in range(max_iter):
            F_int = np.zeros(total_dofs)
            K_tan = np.zeros((total_dofs, total_dofs))

            for elem in self.elements:
                try:
                    f1, f2, k_sub, _, _, _ = elem.get_force_and_stiffness(
                        self.current_nodes
                    )
                except ValueError:
                    return False, F_int

                idx1 = elem.node1_idx * dof_per_node
                idx2 = elem.node2_idx * dof_per_node

                F_int[idx1 : idx1 + dof_per_node] += f1
                F_int[idx2 : idx2 + dof_per_node] += f2

                K_tan[idx1 : idx1 + dof_per_node, idx1 : idx1 + dof_per_node] += k_sub
                K_tan[idx2 : idx2 + dof_per_node, idx2 : idx2 + dof_per_node] += k_sub
                K_tan[idx1 : idx1 + dof_per_node, idx2 : idx2 + dof_per_node] -= k_sub
                K_tan[idx2 : idx2 + dof_per_node, idx1 : idx1 + dof_per_node] -= k_sub

            R = -F_int
            R_free = R[free_dofs]

            if np.linalg.norm(R_free) < tol and k > 0:
                for elem in self.elements:
                    elem.get_force_and_stiffness(
                        self.current_nodes, update_history=True
                    )
                return True, F_int

            K_free = K_tan[np.ix_(free_dofs, free_dofs)]
            try:
                du_free = np.linalg.solve(K_free, R_free)
            except np.linalg.LinAlgError:
                return False, F_int

            for idx, dof_idx in enumerate(free_dofs):
                node_idx = dof_idx // dof_per_node
                axis_idx = dof_idx % dof_per_node
                self.current_nodes[node_idx, axis_idx] += du_free[idx]

        return False, F_int

    def solve_arc_length_step(
        self,
        ds,
        prev_du=None,
        psi=0.0,
        tol=1e-6,
        max_iter=20,
        use_extrapolated_predictor=True,
    ):
        self._last_arc_failure = None
        self._last_arc_details = None
        num_nodes = len(self.nodes)
        dof_per_node = len(self.nodes[0])
        total_dofs = num_nodes * dof_per_node

        # Identify free DOFs
        free_dofs = []
        for i in range(num_nodes):
            for j in range(dof_per_node):
                if (i, j) not in self.constrained_dofs:
                    free_dofs.append(i * dof_per_node + j)

        if prev_du is None and self._last_du is not None:
            prev_du = self._last_du
        prev_dlam = self._last_dlam

        for (n_idx, a_idx), val in self.constrained_dofs.items():
            self.current_nodes[n_idx, a_idx] = self.nodes[n_idx, a_idx] + val

        # Preserve the current state so failed steps can be retried safely.
        saved_nodes = self.current_nodes.copy()
        saved_lam = self.lam

        # Construct Reference Load Vector
        F_ext_ref = np.zeros(total_dofs)
        for (n_idx, a_idx), val in self.loads.items():
            idx = n_idx * dof_per_node + a_idx
            F_ext_ref[idx] = val

        F_ext_ref_free = F_ext_ref[free_dofs]
        fext_norm_sq = float(np.dot(F_ext_ref_free, F_ext_ref_free))
        if fext_norm_sq == 0.0:
            self._last_arc_failure = "zero_reference_load"
            return False, None, self.lam, None

        def solve_k(mat, rhs):
            try:
                return np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError:
                try:
                    return np.linalg.lstsq(mat, rhs, rcond=None)[0]
                except np.linalg.LinAlgError:
                    return None

        def select_root(dl1, dl2, u_R, u_F, du_total_free, dlam_total):
            if not np.isfinite(dl2):
                return dl1

            s_ref = None
            if prev_du is not None:
                prev_du_free = prev_du[free_dofs]
                prev_dlam_use = 0.0 if prev_dlam is None else prev_dlam
                s_ref = np.concatenate(
                    [prev_du_free, psi * prev_dlam_use * F_ext_ref_free]
                )
                if np.linalg.norm(s_ref) == 0.0:
                    s_ref = None

            if s_ref is None:
                s_ref = np.concatenate(
                    [du_total_free, psi * dlam_total * F_ext_ref_free]
                )
                if np.linalg.norm(s_ref) == 0.0:
                    return dl1

            def score(dl):
                du_trial = du_total_free + u_R + dl * u_F
                dlam_trial = dlam_total + dl
                s_trial = np.concatenate([du_trial, psi * dlam_trial * F_ext_ref_free])
                return float(np.dot(s_ref, s_trial))

            return dl2 if score(dl2) > score(dl1) else dl1

        use_predictor = (
            use_extrapolated_predictor
            and self._last_du is not None
            and self._arc_ds is not None
            and self._arc_dlamb is not None
            and self._arc_ds > 0.0
        )

        if use_predictor:
            alpha = ds / self._arc_ds
            du_total_free = alpha * self._last_du[free_dofs]
            d_lam_total = alpha * self._arc_dlamb
            self.lam = saved_lam + d_lam_total
            for idx, dof_idx in enumerate(free_dofs):
                node_idx = dof_idx // dof_per_node
                axis_idx = dof_idx % dof_per_node
                self.current_nodes[node_idx, axis_idx] = (
                    saved_nodes[node_idx, axis_idx] + du_total_free[idx]
                )
        else:
            # --- Predictor Step ---
            K_tan = np.zeros((total_dofs, total_dofs))
            for elem in self.elements:
                try:
                    _, _, k_sub, _, _, _ = elem.get_force_and_stiffness(
                        self.current_nodes
                    )
                except ValueError:
                    self.current_nodes = saved_nodes
                    self.lam = saved_lam
                    self._last_arc_failure = "invalid_length_predictor"
                    return False, None, self.lam, None
                idx1 = elem.node1_idx * dof_per_node
                idx2 = elem.node2_idx * dof_per_node
                K_tan[idx1 : idx1 + dof_per_node, idx1 : idx1 + dof_per_node] += k_sub
                K_tan[idx2 : idx2 + dof_per_node, idx2 : idx2 + dof_per_node] += k_sub
                K_tan[idx1 : idx1 + dof_per_node, idx2 : idx2 + dof_per_node] -= k_sub
                K_tan[idx2 : idx2 + dof_per_node, idx1 : idx1 + dof_per_node] -= k_sub

            K_free = K_tan[np.ix_(free_dofs, free_dofs)]
            u_F = solve_k(K_free, F_ext_ref_free)
            if u_F is None:
                self.current_nodes = saved_nodes
                self.lam = saved_lam
                self._last_arc_failure = "singular_tangent_predictor"
                return False, None, self.lam, None

            denom = float(np.dot(u_F, u_F) + (psi**2) * fext_norm_sq)
            if denom <= 0.0:
                self.current_nodes = saved_nodes
                self.lam = saved_lam
                self._last_arc_failure = "nonpositive_denom"
                self._last_arc_details = {"denom": denom}
                return False, None, self.lam, None

            d_lam = ds / np.sqrt(denom)
            if prev_du is not None:
                prev_du_free = prev_du[free_dofs]
                prev_dlam_use = 0.0 if prev_dlam is None else prev_dlam
                s_prev = np.concatenate(
                    [prev_du_free, psi * prev_dlam_use * F_ext_ref_free]
                )
                s_trial = np.concatenate([d_lam * u_F, psi * d_lam * F_ext_ref_free])
                if np.dot(s_prev, s_trial) < 0.0:
                    d_lam = -d_lam

            du_free = d_lam * u_F

            # Update State
            self.lam += d_lam
            for idx, dof_idx in enumerate(free_dofs):
                node_idx = dof_idx // dof_per_node
                axis_idx = dof_idx % dof_per_node
                self.current_nodes[node_idx, axis_idx] += du_free[idx]

            # Track total increments for this step
            du_total_free = du_free.copy()
            d_lam_total = d_lam

        # --- Corrector Loop ---
        for k in range(max_iter):
            F_int = np.zeros(total_dofs)
            K_tan = np.zeros((total_dofs, total_dofs))

            for elem in self.elements:
                try:
                    f1, f2, k_sub, _, _, _ = elem.get_force_and_stiffness(
                        self.current_nodes
                    )
                except ValueError:
                    self.current_nodes = saved_nodes
                    self.lam = saved_lam
                    self._last_arc_failure = "invalid_length_corrector"
                    return False, F_int, self.lam, None
                idx1 = elem.node1_idx * dof_per_node
                idx2 = elem.node2_idx * dof_per_node
                F_int[idx1 : idx1 + dof_per_node] += f1
                F_int[idx2 : idx2 + dof_per_node] += f2
                K_tan[idx1 : idx1 + dof_per_node, idx1 : idx1 + dof_per_node] += k_sub
                K_tan[idx2 : idx2 + dof_per_node, idx2 : idx2 + dof_per_node] += k_sub
                K_tan[idx1 : idx1 + dof_per_node, idx2 : idx2 + dof_per_node] -= k_sub
                K_tan[idx2 : idx2 + dof_per_node, idx1 : idx1 + dof_per_node] -= k_sub

            R = F_int - self.lam * F_ext_ref
            R_free = R[free_dofs]

            constraint = (
                np.dot(du_total_free, du_total_free)
                + (psi**2) * (d_lam_total**2) * fext_norm_sq
                - ds**2
            )

            if np.linalg.norm(R_free) < tol and abs(constraint) < tol:
                for elem in self.elements:
                    elem.get_force_and_stiffness(
                        self.current_nodes, update_history=True
                    )

                du_full = np.zeros(total_dofs)
                du_full[free_dofs] = du_total_free
                self._last_du = du_full.copy()
                self._last_dlam = d_lam_total
                self._last_arc_iters = k + 1
                self._arc_ds = ds
                self._arc_dlamb = d_lam_total
                self._arc_xincr = du_total_free.copy()
                self._arc_step += 1
                return True, F_int, self.lam, du_full

            K_free = K_tan[np.ix_(free_dofs, free_dofs)]
            u_R = solve_k(K_free, -R_free)
            u_F = solve_k(K_free, F_ext_ref_free)
            if u_R is None or u_F is None:
                self.current_nodes = saved_nodes
                self.lam = saved_lam
                self._last_arc_failure = "singular_tangent_corrector"
                return False, F_int, self.lam, None

            a1 = float(np.dot(u_F, u_F) + (psi**2) * fext_norm_sq)
            a2 = float(
                2.0 * np.dot(u_F, du_total_free + u_R)
                + 2.0 * (psi**2) * fext_norm_sq * d_lam_total
            )
            a3 = float(
                np.dot(du_total_free + u_R, du_total_free + u_R)
                + (psi**2) * (d_lam_total**2) * fext_norm_sq
                - ds**2
            )

            if abs(a1) < 1.0e-14:
                if abs(a2) < 1.0e-14:
                    self.current_nodes = saved_nodes
                    self.lam = saved_lam
                    self._last_arc_failure = "quadratic_degenerate"
                    return False, F_int, self.lam, None
                delta_lam = -a3 / a2
            else:
                disc = a2 * a2 - 4.0 * a1 * a3
                if disc < 0.0:
                    disc = 0.0
                sqrt_disc = np.sqrt(disc)
                dl1 = (-a2 + sqrt_disc) / (2.0 * a1)
                dl2 = (-a2 - sqrt_disc) / (2.0 * a1)
                delta_lam = select_root(dl1, dl2, u_R, u_F, du_total_free, d_lam_total)

            delta_u_free = u_R + delta_lam * u_F

            # Update
            self.lam += delta_lam
            d_lam_total += delta_lam
            du_total_free += delta_u_free

            for idx, dof_idx in enumerate(free_dofs):
                node_idx = dof_idx // dof_per_node
                axis_idx = dof_idx % dof_per_node
                self.current_nodes[node_idx, axis_idx] += delta_u_free[idx]

        self.current_nodes = saved_nodes
        self.lam = saved_lam
        self._last_arc_failure = "max_iterations"
        self._last_arc_iters = max_iter
        return False, F_int, self.lam, None

    def solve_arc_length_step_crisfield(
        self, ds, tol=1e-6, max_iter=20, use_extrapolated_predictor=True
    ):
        self._last_arc_failure = None
        self._last_arc_details = None
        num_nodes = len(self.nodes)
        dof_per_node = len(self.nodes[0])
        total_dofs = num_nodes * dof_per_node

        free_dofs = []
        for i in range(num_nodes):
            for j in range(dof_per_node):
                if (i, j) not in self.constrained_dofs:
                    free_dofs.append(i * dof_per_node + j)

        for (n_idx, a_idx), val in self.constrained_dofs.items():
            self.current_nodes[n_idx, a_idx] = self.nodes[n_idx, a_idx] + val

        saved_nodes = self.current_nodes.copy()
        saved_lam = self.lam

        F_ext_ref = np.zeros(total_dofs)
        for (n_idx, a_idx), val in self.loads.items():
            idx = n_idx * dof_per_node + a_idx
            F_ext_ref[idx] = val

        F_ext_ref_free = F_ext_ref[free_dofs]
        if np.linalg.norm(F_ext_ref_free) == 0.0:
            self._last_arc_failure = "zero_reference_load"
            return False, None, self.lam, None

        use_predictor = (
            use_extrapolated_predictor
            and self._last_du is not None
            and self._arc_ds is not None
            and self._arc_dlamb is not None
            and self._arc_ds > 0.0
        )
        if use_predictor:
            alpha = ds / self._arc_ds
            xincr = alpha * self._last_du[free_dofs]
            self.lam = saved_lam + alpha * self._arc_dlamb
            for idx, dof_idx in enumerate(free_dofs):
                node_idx = dof_idx // dof_per_node
                axis_idx = dof_idx % dof_per_node
                self.current_nodes[node_idx, axis_idx] = (
                    saved_nodes[node_idx, axis_idx] + xincr[idx]
                )
        else:
            xincr = (
                self._arc_xincr.copy()
                if self._arc_xincr is not None
                else np.zeros(len(free_dofs))
            )
        arc_ds = ds

        for k in range(max_iter):
            F_int = np.zeros(total_dofs)
            K_tan = np.zeros((total_dofs, total_dofs))

            for elem in self.elements:
                try:
                    f1, f2, k_sub, _, _, _ = elem.get_force_and_stiffness(
                        self.current_nodes
                    )
                except ValueError:
                    self.current_nodes = saved_nodes
                    self.lam = saved_lam
                    self._last_arc_failure = "invalid_length_corrector"
                    return False, F_int, self.lam, None
                idx1 = elem.node1_idx * dof_per_node
                idx2 = elem.node2_idx * dof_per_node
                F_int[idx1 : idx1 + dof_per_node] += f1
                F_int[idx2 : idx2 + dof_per_node] += f2
                K_tan[idx1 : idx1 + dof_per_node, idx1 : idx1 + dof_per_node] += k_sub
                K_tan[idx2 : idx2 + dof_per_node, idx2 : idx2 + dof_per_node] += k_sub
                K_tan[idx1 : idx1 + dof_per_node, idx2 : idx2 + dof_per_node] -= k_sub
                K_tan[idx2 : idx2 + dof_per_node, idx1 : idx1 + dof_per_node] -= k_sub

            R = F_int - self.lam * F_ext_ref
            R_free = R[free_dofs]

            if np.linalg.norm(R_free) < tol and k > 0:
                for elem in self.elements:
                    elem.get_force_and_stiffness(
                        self.current_nodes, update_history=True
                    )
                du_full = np.zeros(total_dofs)
                du_full[free_dofs] = xincr
                self._last_du = du_full.copy()
                self._last_dlam = 0.0
                self._last_arc_iters = k + 1
                self._arc_xincr = xincr.copy()
                self._arc_ds = arc_ds
                self._arc_dlamb = self.lam - saved_lam
                self._arc_step += 1
                return True, F_int, self.lam, du_full

            K_free = K_tan[np.ix_(free_dofs, free_dofs)]
            try:
                displ = np.linalg.solve(K_free, -R_free)
                dispf = np.linalg.solve(K_free, F_ext_ref_free)
            except np.linalg.LinAlgError:
                try:
                    displ = np.linalg.lstsq(K_free, -R_free, rcond=None)[0]
                    dispf = np.linalg.lstsq(K_free, F_ext_ref_free, rcond=None)[0]
                except np.linalg.LinAlgError:
                    self.current_nodes = saved_nodes
                    self.lam = saved_lam
                    self._last_arc_failure = "singular_tangent_corrector"
                    return False, F_int, self.lam, None

            ufuf = float(np.dot(dispf, dispf))
            urur = float(np.dot(displ, displ))
            ufur = float(np.dot(dispf, displ))
            ufdx = float(np.dot(dispf, xincr))

            if k == 0 and not use_predictor:
                if self._arc_step == 0 and urur > 0.0:
                    arc_ds = np.sqrt(urur)
                xincr = np.zeros_like(xincr)
                displ = np.zeros_like(displ)
                if ufuf <= 0.0:
                    self.current_nodes = saved_nodes
                    self.lam = saved_lam
                    self._last_arc_failure = "nonpositive_ufuf"
                    return False, F_int, self.lam, None
                gamma = abs(arc_ds) / np.sqrt(ufuf)
                if ufdx != 0.0:
                    gamma *= np.sign(ufdx)
            else:
                urdx = float(np.dot(displ, xincr))
                dxdx = float(np.dot(xincr, xincr))
                a1 = ufuf
                a2 = 2.0 * (ufdx + ufur)
                a3 = dxdx + 2.0 * urdx + urur - arc_ds * arc_ds
                discr = a2 * a2 - 4.0 * a1 * a3
                if discr < 0.0:
                    self.current_nodes = saved_nodes
                    self.lam = saved_lam
                    self._last_arc_failure = "negative_discriminant"
                    return False, F_int, self.lam, None
                discr = np.sqrt(discr)
                if a2 < 0.0:
                    discr = -discr
                discr = -(a2 + discr) / 2.0
                if discr == 0.0:
                    self.current_nodes = saved_nodes
                    self.lam = saved_lam
                    self._last_arc_failure = "quadratic_degenerate"
                    return False, F_int, self.lam, None
                gamm1 = discr / a1
                gamm2 = a3 / discr
                cos1 = urdx + gamm1 * ufdx
                cos2 = urdx + gamm2 * ufdx
                gamma = gamm2 if cos2 > cos1 else gamm1

            displ = displ + gamma * dispf
            xincr = xincr + displ
            self.lam += gamma

            for idx, dof_idx in enumerate(free_dofs):
                node_idx = dof_idx // dof_per_node
                axis_idx = dof_idx % dof_per_node
                self.current_nodes[node_idx, axis_idx] += displ[idx]

        self.current_nodes = saved_nodes
        self.lam = saved_lam
        self._last_arc_failure = "max_iterations"
        self._last_arc_iters = max_iter
        return False, F_int, self.lam, None
