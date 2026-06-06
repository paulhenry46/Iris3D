import numpy as np
import pyvista as pv
from typing import Optional
from iris3d.models import CollisionEvent
from iris3d.core import CoordinateTransformer

class EventVisualizer:
    """
    Renders high-energy physics collision events in a highly fluid,
    GPU-accelerated 3D canvas using PyVista (VTK), featuring interactive
    picking tooltips, realistic magnetic field bending, and style-coded tracks.
    """
    def __init__(self, theme: str = "dark"):
        self.transformer = CoordinateTransformer()
        self.current_selected_id = None  # Tracker pour l'objet actuellement sélectionné
        self.calorimeter_outer_radius = 2.8
        self.detector_ecal_r = 1.75      # Bord externe exact du calorimètre
        self.detector_muon_r = 4.0
        self.tracker_radius = 1.5
        self.tracker_length = 5.0
        
        if theme == "dark":
            pv.set_plot_theme("dark")
            
    def _get_particle_color(self, pid: int) -> str:
        """Assigns professional color-coding based on Particle ID (PDG code)."""
        abs_pid = abs(pid)
        if abs_pid == 11:    # Electrons / Positrons
            return "cyan"
        elif abs_pid == 13:  # Muons
            return "lime"
        elif abs_pid == 22:  # Photons (Neutral)
            return "yellow"
        elif abs_pid in (211, 321, 2212): # Charged Hadrons (Pions, Kaons, Protons)
            return "salmon"
        return "white"

    def _add_detector_geometry(self, plotter: pv.Plotter):
        """
        Draws passive reference structures centered at (0,0,0) and aligned 
        along the Z-axis (beam pipe line) to provide physical scale.
        """
        tracker = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            radius=self.tracker_radius, height=self.tracker_length, resolution=50
        )
        plotter.add_mesh(
            tracker, color="deepskyblue", opacity=0.08, style="surface",
            show_edges=True, edge_color="dodgerblue", line_width=1,
            name="detector_tracker", pickable=False
        )

        calorimeter_length = 6.0
        calorimeter = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            radius=self.calorimeter_outer_radius, height=calorimeter_length, resolution=50
        )
        plotter.add_mesh(
            calorimeter, color="crimson", opacity=0.04, style="surface",
            show_edges=True, edge_color="firebrick", line_width=1,
            name="detector_calorimeter", pickable=False
        )

    def plot_event(self, event: CollisionEvent, mode: str = "both", p_scale: float = 1.0, j_scale: float = 0.01, B_field: float = 3.8):
        """
        Orchestrates the event display.
        Modes available: "both" (Split Screen), "detector" (3D Detector view only), "lego" (Lego Plot only).
        """
        import numpy as np
        import pyvista as pv

        # 1. Détermination du layout de la fenêtre
        if mode == "both":
            plotter = pv.Plotter(window_size=[1500, 750], shape=(1, 2), title=f"Iris3D - Dual Dynamic Display")
        elif mode in ["detector", "lego"]:
            plotter = pv.Plotter(window_size=[1024, 768], shape=(1, 1), title=f"Iris3D - Single View [{mode.upper()}]")
        else:
            raise ValueError("Invalid mode. Choose from 'both', 'detector', or 'lego'.")

        # 2. Extraction sécurisée des métadonnées pour l'affichage
        try:
            run_id = getattr(event.metadata, "run_id", "N/A")
            event_id = getattr(event.metadata, "event_id", "N/A")
        except Exception:
            try:
                run_id = event["metadata"]["run_id"]
                event_id = event["metadata"]["event_id"]
            except Exception:
                run_id, event_id = "N/A", "N/A"

        # 3. Initialisation des états partagés (Attributs d'instance pour le picking)
        self.tooltip_dict = {}
        self.current_selected_id = None
        self._actor_registry = {
            "particles": {},
            "jet_cones": {},
            "jet_towers": {},
            "met": {}
        }

        # 4. Pipeline de géométrie
        spatial_data = self.transformer.extract_event_arrays(
            event, p_scale=p_scale, j_scale=j_scale, B_field=B_field,
            detector_ecal_r=self.detector_ecal_r,
            detector_hcal_r=self.calorimeter_outer_radius,
            detector_muon_r=self.detector_muon_r
        )

        # 5. Remplissage des Subplots selon la configuration
        if mode == "both":
            self._plot_3d_detector(plotter, spatial_data, event, run_id, event_id, subplot_idx=(0, 0))
            self._plot_lego_calorimeter(plotter, spatial_data, event, subplot_idx=(0, 1))
        elif mode == "detector":
            self._plot_3d_detector(plotter, spatial_data, event, run_id, event_id, subplot_idx=(0, 0))
        elif mode == "lego":
            self._plot_lego_calorimeter(plotter, spatial_data, event, subplot_idx=(0, 0))

        # 6. Injection de la logique de picking unifiée
        def picking_callback(mesh):
            if not mesh or "mesh_id" not in mesh.field_data:
                return
            
            mesh_id = mesh.field_data["mesh_id"][0]
            if mesh_id not in self.tooltip_dict:
                return

            hud_text, orig_color_name = self.tooltip_dict[mesh_id]
            rgb_orig = pv.Color(orig_color_name).float_rgb
            rgb_white = (1.0, 1.0, 1.0)
            
            # Nettoyage de la sélection précédente via le registre partagé
            if self.current_selected_id:
                old_id = self.current_selected_id
                _, old_color_name = self.tooltip_dict[old_id]
                rgb_old = pv.Color(old_color_name).float_rgb
                
                if old_id.startswith("particle_"):
                    idx = int(old_id.split('_')[1])
                    if idx in self._actor_registry["particles"]:
                        self._actor_registry["particles"][idx].GetProperty().SetColor(rgb_old)
                elif old_id.startswith("jet_"):
                    idx = int(old_id.split('_')[1])
                    if idx in self._actor_registry["jet_cones"]:
                        self._actor_registry["jet_cones"][idx].GetProperty().SetColor(rgb_old)
                    if idx in self._actor_registry["jet_towers"]:
                        self._actor_registry["jet_towers"][idx].GetProperty().SetColor(rgb_old)
                elif old_id == "missing_energy_vector" and self._actor_registry["met"]:
                    self._actor_registry["met"]["line"].GetProperty().SetColor(rgb_old)
                    self._actor_registry["met"]["tip"].GetProperty().SetColor(rgb_old)

            # Application de la surbrillance
            if mesh_id.startswith("particle_"):
                idx = int(mesh_id.split('_')[1])
                if idx in self._actor_registry["particles"]:
                    self._actor_registry["particles"][idx].GetProperty().SetColor(rgb_white)
            elif mesh_id.startswith("jet_"):
                idx = int(mesh_id.split('_')[1])
                if idx in self._actor_registry["jet_cones"]:
                    self._actor_registry["jet_cones"][idx].GetProperty().SetColor(rgb_white)
                if idx in self._actor_registry["jet_towers"]:
                    self._actor_registry["jet_towers"][idx].GetProperty().SetColor(rgb_white)
            elif mesh_id == "missing_energy_vector" and self._actor_registry["met"]:
                self._actor_registry["met"]["line"].GetProperty().SetColor(rgb_white)
                self._actor_registry["met"]["tip"].GetProperty().SetColor(rgb_white)

            self.current_selected_id = mesh_id
            
            # Toujours cibler l'emplacement d'origine du HUD (à gauche ou affichage seul)
            if mode in ["both", "detector"]:
                plotter.subplot(0, 0)
                plotter.add_text(hud_text, position="upper_left", font_size=11, font="courier", color="#38bdf8", name="metadata_banner")
            plotter.render()

        # Activation globale unique
        plotter.enable_mesh_picking(callback=picking_callback, show=False, left_clicking=True, show_message=False)
        plotter.show()
    
    def _plot_3d_detector(self, plotter, spatial_data, event, run_id, event_id, subplot_idx=(0, 0)):
        """Draws the central tracking system and geometric outputs."""
        import numpy as np
        import pyvista as pv

        plotter.subplot(*subplot_idx)
        plotter.set_background(color="#0f172a")               
        plotter.add_axes()
        plotter.show_grid(color="#273549")                     
        plotter.enable_anti_aliasing("msaa", multi_samples=4)  
        
        self._add_detector_geometry(plotter)
        
        vertex = pv.Sphere(radius=0.05, center=(0.0, 0.0, 0.0))
        plotter.add_mesh(vertex, color="magenta", render_points_as_spheres=True, label="Interaction Vertex")
        
        p_meta = spatial_data["particle_metadata"]
        p_paths = spatial_data["particle_paths"]

        # Traces
        for i in range(len(p_paths)):
            points = p_paths[i]
            metadata = p_meta[i]
            p_name = metadata.get("name", f"Track {i}")
            p_pid = metadata.get("pid", 0)
            p_charge = metadata.get("charge", 0)
            
            try:
                source_particle = event.particles[i]
                pt_val, eta_val, phi_val = source_particle.pt, source_particle.eta, source_particle.phi
            except Exception:
                try:
                    pt_val = float(event["particles"]["pt"][i])
                    eta_val = float(event["particles"]["eta"][i])
                    phi_val = float(event["particles"]["phi"][i])
                except Exception:
                    pt_val, eta_val, phi_val = 0.0, 0.0, 0.0
            
            if p_charge != 0:
                track_mesh = pv.Spline(points, len(points))
            else:
                track_mesh = pv.PolyData(points)
                if p_pid == 22:  
                    lines_connectivity = []
                    for idx in range(0, len(points) - 1, 2):
                        lines_connectivity.extend([2, idx, idx + 1])
                    track_mesh.lines = np.array(lines_connectivity, dtype=np.int32)
                else:  
                    cells = np.hstack([[len(points)], np.arange(len(points))])
                    track_mesh.lines = cells
            
            color = self._get_particle_color(p_pid)
            mesh_id = f"particle_{i}"
            
            self.tooltip_dict[mesh_id] = ((
                f">> INSPECTING TARGET: PARTICLE TRACK #{i}\n----------------------------------------\n"
                f" Identity    : {p_name} (PDG: {p_pid})\n Momentum pT : {pt_val:.2f} GeV\n"
                f" Pseudo-Rap  : {eta_val:.2f}\n Azimuth phi : {phi_val:.2f} rad\n Charge      : {p_charge:+.0f}"
            ), color)
            
            track_mesh.field_data["mesh_id"] = [mesh_id]
            if p_charge == 0:
                act = plotter.add_mesh(track_mesh, color=color, line_width=2.5 if p_pid == 22 else 1, opacity=0.9 if p_pid == 22 else 0.5, name=mesh_id)
            else:
                act = plotter.add_mesh(track_mesh, color=color, line_width=4, opacity=1.0, name=mesh_id)
            self._actor_registry["particles"][i] = act

        # Cônes de jets
        for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
            direction = np.array(jet_geo["unit_direction"])
            length, radius = jet_geo["length"], jet_geo["radius"]
            
            cone_center = direction * (length / 2.0)
            jet_cone = pv.Cone(center=cone_center, direction=-direction, height=length, radius=radius, resolution=30)
            
            try:
                source_jet = event.jets[i]
                j_energy, j_eta, j_phi, j_dr = source_jet.energy, source_jet.eta, source_jet.phi, source_jet.delta_r
            except Exception:
                try:
                    j_energy = float(event["jets"]["energy"][i])
                    j_eta = float(event["jets"]["eta"][i])
                    j_phi = float(event["jets"]["phi"][i])
                    j_dr = float(event["jets"]["delta_r"][i])
                except Exception:
                    j_energy, j_eta, j_phi, j_dr = 0.0, 0.0, 0.0, 0.4
            
            mesh_id = f"jet_{i}"
            self.tooltip_dict[mesh_id] = ((
                f">> INSPECTING TARGET: RECONSTRUCTED JET #{i}\n----------------------------------------\n"
                f" Transverse E : {j_energy:.2f} GeV\n Pseudo-Rap   : {j_eta:.2f}\n"
                f" Azimuth phi  : {j_phi:.2f} rad\n Cone Radius  : {j_dr:.2f} (delta_R)"
            ), "orange")
            
            jet_cone.field_data["mesh_id"] = [mesh_id]
            act = plotter.add_mesh(jet_cone, color="orange", opacity=0.35, show_edges=True, edge_color="darkorange", name=mesh_id)
            self._actor_registry["jet_cones"][i] = act

        # Missing Energy (MET)
        met_data = spatial_data.get("missing_energy", {"pt": 0.0, "phi": 0.0, "vector": (0.0, 0.0, 0.0)})
        if met_data["pt"] > 0.5:
            met_vector = np.array(met_data["vector"])
            met_points = np.linspace(np.array([0.0, 0.0, 0.0]), met_vector, 30)
            met_mesh = pv.PolyData(met_points)
            
            met_lines = []
            for idx in range(0, len(met_points) - 1, 2):
                met_lines.extend([2, idx, idx + 1])
            met_mesh.lines = np.array(met_lines, dtype=np.int32)
            
            met_direction = met_vector / np.linalg.norm(met_vector)
            met_cone_tip = pv.Cone(center=met_vector, direction=met_direction, height=0.5, radius=0.25, resolution=20)
            
            mesh_id = "missing_energy_vector"
            self.tooltip_dict[mesh_id] = ((
                f">> WARNING: MISSING TRANSVERSE ENERGY DETECTED\n----------------------------------------\n"
                f" Unseen pT    : {met_data['pt']:.2f} GeV\n Escape Angle : {met_data['phi']:.2f} rad\n"
                f" Source       : Neutrino / Dark Matter Candidate"
            ), "red")
            
            met_mesh.field_data["mesh_id"] = [mesh_id]
            met_cone_tip.field_data["mesh_id"] = [mesh_id]
            
            act_line = plotter.add_mesh(met_mesh, color="red", line_width=5, opacity=1.0, name=f"{mesh_id}_line")
            act_tip = plotter.add_mesh(met_cone_tip, color="red", opacity=1.0, name=f"{mesh_id}_tip")
            self._actor_registry["met"] = {"line": act_line, "tip": act_tip}

        # Injection HUD texte
        plotter.add_text(
            f"IRIS3D // EVENT DETECTOR HUD ACTIVE\n-----------------------------------\nRun ID : {run_id} | Event ID : {event_id}\n\nSelect sub-atomic signature to decode...", 
            position="upper_left", font_size=11, font="courier", color="#38bdf8", name="metadata_banner"
        )
        plotter.add_legend(bcolor=None, face="circle")
        plotter.camera_position = [(5.0, 5.0, 4.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        plotter.camera.zoom(0.8)


    def _plot_lego_calorimeter(self, plotter, spatial_data, event, subplot_idx=(0, 1)):
        """Draws the static 2D plane unfold calorimeter towers."""
        import numpy as np
        import pyvista as pv

        plotter.subplot(*subplot_idx)
        plotter.set_background(color="#090d16")
        plotter.add_text("CALORIMETER METRIC (STATIC LEGO PLOT)", position=(0.05, 0.92), font_size=12, font="courier", color="#fb923c")
        
        # Sol cylindrique déroulé
        lego_floor = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=6.0, j_size=2 * np.pi)
        plotter.add_mesh(lego_floor, color="#1e293b", style="surface", show_edges=True, edge_color="#334155", pickable=False)
        
        # Marquages d'axes
        plotter.add_point_labels(np.array([[-3.0, -3.3, 0.01], [0.0, -3.3, 0.01], [3.0, -3.3, 0.01]]), ["eta = -3.0", "eta = 0.0", "eta = +3.0"], font_family="courier", font_size=12, show_points=False)
        plotter.add_mesh(pv.Line([-3.0, -3.3, 0.01], [3.0, -3.3, 0.01]), color="#fb923c", line_width=2, pickable=False)
        plotter.add_point_labels(np.array([[-3.3, -np.pi, 0.01], [-3.3, 0.0, 0.01], [-3.3, np.pi, 0.01]]), ["phi = -pi", "phi = 0", "phi = +pi"], font_family="courier", font_size=12, show_points=False)
        plotter.add_mesh(pv.Line([-3.2, -np.pi, 0.01], [-3.2, np.pi, 0.01]), color="#fb923c", line_width=2, pickable=False)

        # Calcul de normalisation adaptative de la hauteur
        all_energies = []
        for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
            try:
                j_energy = float(event["jets"]["energy"][i])
            except Exception:
                try: j_energy = event.jets[i].energy
                except Exception: j_energy = jet_geo["length"] * 10.0
            all_energies.append(j_energy)
            
        max_allowed_height = 2.5 
        max_e = max(all_energies) if len(all_energies) > 0 else 1.0
        v_scale = max_allowed_height / max_e  

        # Instanciation des boîtes LEGO et liaison d'ID pour le picking synchrone
        for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
            direction = np.array(jet_geo["unit_direction"])
            eta = jet_geo.get("eta", direction[2] * 1.5)
            phi = jet_geo.get("phi", np.arctan2(direction[1], direction[0]))
            final_height = all_energies[i] * v_scale  

            lego_tower = pv.Box(bounds=[eta - 0.18, eta + 0.18, phi - 0.18, phi + 0.18, 0.0, final_height])
            mesh_id = f"jet_{i}"
            lego_tower.field_data["mesh_id"] = [mesh_id]
            
            act = plotter.add_mesh(lego_tower, color="orange", opacity=0.85, show_edges=True, edge_color="white", name=f"lego_tower_{i}")
            self._actor_registry["jet_towers"][i] = act

        # Positionnement de la vue
        plotter.camera_position = [(0.0, -5.0, 7.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        plotter.camera.zoom(0.75)

    def animate_event(self, event: CollisionEvent, mode: str = "both", p_scale: float = 1.0, j_scale: float = 0.01, B_field: float = 3.8, speed: float = 0.05):
        """
        Orchestrates the high-performance cinematic animation display.
        Modes available: "both", "detector", "lego".
        """
        import numpy as np
        import pyvista as pv

        # 1. Configuration des fenêtres selon le mode choisi
        if mode == "both":
            plotter = pv.Plotter(window_size=[1500, 750], shape=(1, 2), title="Iris3D - Dual Cinematic Display & Lego Plot")
        elif mode in ["detector", "lego"]:
            plotter = pv.Plotter(window_size=[1024, 768], shape=(1, 1), title=f"Iris3D - Cinematic [{mode.upper()}]")
        else:
            raise ValueError("Invalid mode. Choose from 'both', 'detector', or 'lego'.")

        # Extraction globale unique des données spatiales
        spatial_data = self.transformer.extract_event_arrays(
            event, p_scale=p_scale, j_scale=j_scale, B_field=B_field,
            detector_ecal_r=self.detector_ecal_r,
            detector_hcal_r=self.calorimeter_outer_radius,
            detector_muon_r=self.detector_muon_r
        )

        # Initialisation des états et registres partagés pour le picking
        self.tooltip_dict = {}
        self.current_selected_id = None
        self._actor_registry = {
            "particles": {},
            "jet_cones": {},
            "jet_towers": {},
            "met": {}
        }

        # Conteneurs d'animation internes (permettent de lier les variables locales aux sous-fonctions)
        ctx = {}

        # 2. Routage et initialisation des subplots
        if mode == "both":
            self._prepare_cinematic_detector(plotter, spatial_data, ctx, subplot_idx=(0, 0))
            self._prepare_cinematic_calorimeter(plotter, spatial_data, ctx, subplot_idx=(0, 1))
        elif mode == "detector":
            self._prepare_cinematic_detector(plotter, spatial_data, ctx, subplot_idx=(0, 0))
        elif mode == "lego":
            self._prepare_cinematic_calorimeter(plotter, spatial_data, ctx, subplot_idx=(0, 0))

        # Gestion de l'état du clavier
        state = {"is_paused": False}
        plotter.add_key_event('space', lambda: state.update({"is_paused": not state["is_paused"]}))

        plotter.show(auto_close=False, interactive_update=True)

        max_r = self.detector_muon_r
        current_r = -3.0  

        # ==========================================================
        # BOUCLE DE RENDU OPTIMISÉE (ZÉRO ALLOCATION EN SURFACE)
        # ==========================================================
        while plotter.render_window is not None:
            if not hasattr(plotter, 'iren') or plotter.iren is None or plotter.render_window.GetInteractor().GetDone():
                break

            if not state["is_paused"]:
                current_r += speed
                if current_r > max_r + 0.5:
                    current_r = -3.0  

            # --- MISE À JOUR : SUBPLOT GAUCHE (DÉTECTEUR) ---
            if mode in ["both", "detector"]:
                plotter.subplot(0, 0)
                
                if current_r < 0: 
                    ctx["beam1_actor"].SetVisibility(True)
                    ctx["beam2_actor"].SetVisibility(True)
                    ctx["vertex_actor"].SetVisibility(True)
                    ctx["vertex_actor"].GetProperty().SetColor(pv.Color("gray").float_rgb)
                    ctx["shockwave_actor"].SetVisibility(False)
                    
                    if ctx["met_actor_line"]: ctx["met_actor_line"].SetVisibility(False)
                    if ctx["met_actor_tip"]: ctx["met_actor_tip"].SetVisibility(False)
                    
                    for act in ctx["particle_actors"]: act.SetVisibility(False)
                    for act in ctx["jet_actors"]: act.SetVisibility(False)
                    for idx, act in self._actor_registry["jet_towers"].items(): act.SetVisibility(False)
                    
                    dist = abs(current_r)
                    beam_len = min(1.0, dist * 0.5) 
                    ctx["beam1_actor"].mapper.dataset.points = np.array([[0.0, 0.0, dist + beam_len], [0.0, 0.0, dist]])
                    ctx["beam2_actor"].mapper.dataset.points = np.array([[0.0, 0.0, -(dist + beam_len)], [0.0, 0.0, -dist]])
                    ctx["beam1_actor"].mapper.dataset.Modified()
                    ctx["beam2_actor"].mapper.dataset.Modified()
                    
                    ctx["calorimeter_actor"].GetProperty().SetOpacity(0.02)
                    ctx["calorimeter_actor"].GetProperty().SetEdgeColor(pv.Color("firebrick").float_rgb)
                    
                    status_txt = "STATUS: PAUSED" if state["is_paused"] else "Status: STEERING PACKETS"
                    ctx["hud"].SetInput(f"IRIS3D // LHC BEAMS APPROACHING\n-------------------------------------------\n{status_txt}")
                    
                else:  # Phase de Collision
                    ctx["beam1_actor"].SetVisibility(False)
                    ctx["beam2_actor"].SetVisibility(False)
                    ctx["vertex_actor"].GetProperty().SetColor(pv.Color("white" if current_r < 0.2 else "magenta").float_rgb)

                    if current_r >= self.detector_ecal_r:
                        ctx["calorimeter_actor"].GetProperty().SetOpacity(0.15 if current_r < self.calorimeter_outer_radius else 0.06)
                        ctx["calorimeter_actor"].GetProperty().SetEdgeColor(pv.Color("red").float_rgb)
                        
                        if current_r < self.calorimeter_outer_radius + 0.3:
                            ctx["shockwave_actor"].SetVisibility(True)
                            ctx["shockwave_actor"].scale = (current_r, current_r, 1.0)
                        else:
                            ctx["shockwave_actor"].SetVisibility(False)
                    else:
                        ctx["calorimeter_actor"].GetProperty().SetOpacity(0.02)
                        ctx["calorimeter_actor"].GetProperty().SetEdgeColor(pv.Color("firebrick").float_rgb)
                        ctx["shockwave_actor"].SetVisibility(False)

                    # Masquage dynamique des trajectoires
                    for i, (poly, full_path) in enumerate(ctx["particle_polydata_lists"]):
                        actor = ctx["particle_actors"][i]
                        distances = np.linalg.norm(full_path, axis=1)
                        visible_mask = distances <= current_r
                        visible_points = full_path[visible_mask]
                        
                        if len(visible_points) > 1:
                            actor.SetVisibility(True)
                            new_spline = pv.Spline(visible_points, n_points=len(visible_points) * 2)
                            actor.mapper.dataset.copy_from(new_spline)
                            actor.mapper.dataset.Modified()
                        else:
                            actor.SetVisibility(False)

                    # Apparition progressive des cônes
                    if current_r >= self.tracker_radius:
                        for i, act in enumerate(ctx["jet_actors"]):
                            act.SetVisibility(True)
                            act.GetProperty().SetOpacity(min(0.35, (current_r - self.tracker_radius) * 0.2))
                    else:
                        for act in ctx["jet_actors"]: act.SetVisibility(False)

                    # Énergie manquante (MET)
                    if spatial_data.get("missing_energy", {}).get("pt", 0.0) > 0.5 and current_r >= self.calorimeter_outer_radius + 1:
                        if ctx["met_actor_line"]: ctx["met_actor_line"].SetVisibility(True)
                        if ctx["met_actor_tip"]: ctx["met_actor_tip"].SetVisibility(True)
                    else:
                        if ctx["met_actor_line"]: ctx["met_actor_line"].SetVisibility(False)
                        if ctx["met_actor_tip"]: ctx["met_actor_tip"].SetVisibility(False)

                    state_label = "|| PAUSED" if state["is_paused"] else ('TRACKING CORE' if current_r < self.detector_ecal_r else 'CALORIMETER SHOWER')
                    ctx["hud"].SetInput(f"IRIS3D // TIME-OF-FLIGHT SIMULATION ACTIVE\n-------------------------------------------\nWavefront Radius : {current_r:.2f} meters\nSub-atomic State : {state_label}")

            # --- MISE À JOUR : SUBPLOT DROIT (LEGO PLOT) ---
            if mode in ["both", "lego"]:
                # Si on est en mode "lego" pur (pas de détecteur), l'animation se cale sur un rayon virtuel positif
                v_current_r = current_r if mode == "both" else (current_r + 3.0)
                
                if mode == "lego":
                    plotter.subplot(0, 0)
                else:
                    plotter.subplot(0, 1)

                if v_current_r >= self.detector_ecal_r:
                    growth_range = self.calorimeter_outer_radius - self.detector_ecal_r
                    progress = (v_current_r - self.detector_ecal_r) / growth_range
                    
                    for i, (box_mesh, init_pts) in enumerate(ctx["lego_mesh_references"]):
                        target_height = ctx["max_heights"][i] * min(1.0, max(0.0, progress))
                        
                        if target_height > 0.001:
                            self._actor_registry["jet_towers"][i].SetVisibility(True)
                            new_pts = init_pts.copy()
                            new_pts[init_pts[:, 2] > 0.001, 2] = target_height
                            box_mesh.points = new_pts
                        else:
                            self._actor_registry["jet_towers"][i].SetVisibility(False)
                else:
                    for idx, act in self._actor_registry["jet_towers"].items(): act.SetVisibility(False)

            plotter.update(16, force_redraw=True)

        # 3. INTERACTION APRÈS FIN DE LA SIMULATION (CROSS-PICKING ACTIF)
        def picking_callback(mesh):
            if not mesh or "mesh_id" not in mesh.field_data:
                return
            mesh_id = mesh.field_data["mesh_id"][0]
            if mesh_id not in self.tooltip_dict:
                return

            hud_text, orig_color_name = self.tooltip_dict[mesh_id]
            rgb_orig = pv.Color(orig_color_name).float_rgb
            rgb_white = (1.0, 1.0, 1.0)
            
            # Nettoyage
            if self.current_selected_id:
                old_id = self.current_selected_id
                _, old_color_name = self.tooltip_dict[old_id]
                rgb_old = pv.Color(old_color_name).float_rgb
                
                if old_id.startswith("particle_"):
                    idx = int(old_id.split('_')[1])
                    if idx in self._actor_registry["particles"]: self._actor_registry["particles"][idx].GetProperty().SetColor(rgb_old)
                elif old_id.startswith("jet_"):
                    idx = int(old_id.split('_')[1])
                    if idx in self._actor_registry["jet_cones"]: self._actor_registry["jet_cones"][idx].GetProperty().SetColor(rgb_old)
                    if idx in self._actor_registry["jet_towers"]: self._actor_registry["jet_towers"][idx].GetProperty().SetColor(rgb_old)
                elif old_id == "missing_energy_vector" and self._actor_registry["met"]:
                    self._actor_registry["met"]["line"].GetProperty().SetColor(rgb_old)
                    self._actor_registry["met"]["tip"].GetProperty().SetColor(rgb_old)

            # Highlight
            if mesh_id.startswith("particle_"):
                idx = int(mesh_id.split('_')[1])
                if idx in self._actor_registry["particles"]: self._actor_registry["particles"][idx].GetProperty().SetColor(rgb_white)
            elif mesh_id.startswith("jet_"):
                idx = int(mesh_id.split('_')[1])
                if idx in self._actor_registry["jet_cones"]: self._actor_registry["jet_cones"][idx].GetProperty().SetColor(rgb_white)
                if idx in self._actor_registry["jet_towers"]: self._actor_registry["jet_towers"][idx].GetProperty().SetColor(rgb_white)
            elif mesh_id == "missing_energy_vector" and self._actor_registry["met"]:
                self._actor_registry["met"]["line"].GetProperty().SetColor(rgb_white)
                self._actor_registry["met"]["tip"].GetProperty().SetColor(rgb_white)

            self.current_selected_id = mesh_id
            if mode in ["both", "detector"]:
                plotter.subplot(0, 0)
                ctx["hud"].SetInput(hud_text)
            plotter.render()

        # Activation finale de l'interactivité
        plotter.enable_mesh_picking(callback=picking_callback, show=False, left_clicking=True, show_message=False)
        plotter.show()

    def _prepare_cinematic_detector(self, plotter, spatial_data, ctx, subplot_idx=(0, 0)):
        """Pre-allocates resources and registers metadata for the 3D Tracking detector view."""
        import numpy as np
        import pyvista as pv

        plotter.subplot(*subplot_idx)
        plotter.set_background(color="#0f172a")
        plotter.add_axes()
        plotter.show_grid(color="#273549")
        
        ctx["calorimeter_mesh"] = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=self.calorimeter_outer_radius, height=6.0, resolution=50)
        ctx["calorimeter_actor"] = plotter.add_mesh(ctx["calorimeter_mesh"], color="crimson", opacity=0.02, style="surface", show_edges=True, edge_color="firebrick")

        tracker_mesh = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=self.tracker_radius, height=self.tracker_length, resolution=50)
        plotter.add_mesh(tracker_mesh, color="deepskyblue", opacity=0.08, style="surface", show_edges=True, edge_color="dodgerblue", pickable=False)

        ctx["hud"] = plotter.add_text("IRIS3D // INITIALIZING...", position=(0.02, 0.85), font_size=11, font="courier", color="#38bdf8")

        beam1 = pv.Line([0, 0, 5.0], [0, 0, 0.0])
        beam2 = pv.Line([0, 0, -5.0], [0, 0, 0.0])
        ctx["beam1_actor"] = plotter.add_mesh(beam1, color="#38bdf8", line_width=6, pickable=False)
        ctx["beam2_actor"] = plotter.add_mesh(beam2, color="#38bdf8", line_width=6, pickable=False)
        
        vertex_mesh = pv.Sphere(radius=0.05, center=(0.0, 0.0, 0.0))
        ctx["vertex_actor"] = plotter.add_mesh(vertex_mesh, color="gray", pickable=False)

        shockwave_base = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=1.0, height=5.8, resolution=40)
        ctx["shockwave_actor"] = plotter.add_mesh(shockwave_base, color="orange", opacity=0.0, style="wireframe", line_width=2, pickable=False)

        ctx["particle_actors"] = []
        ctx["particle_polydata_lists"] = []
        
        p_paths = spatial_data["particle_paths"]
        p_meta = spatial_data["particle_metadata"]

        for i, path in enumerate(p_paths):
            color = self._get_particle_color(p_meta[i]["pid"])
            full_path = np.array(path)
            poly = pv.PolyData(full_path)
            
            cells = np.full((len(full_path)-1, 3), 2, dtype=np.int_)
            cells[:, 1] = np.arange(0, len(full_path)-1)
            cells[:, 2] = np.arange(1, len(full_path))
            poly.lines = cells
            
            # Injection ID picking
            mesh_id = f"particle_{i}"
            poly.field_data["mesh_id"] = [mesh_id]
            self.tooltip_dict[mesh_id] = ((
                f">> INSPECTING TARGET: PARTICLE TRACK #{i}\n----------------------------------------\n"
                f" Identity    : {p_meta[i].get('name', f'Track {i}')} (PDG: {p_meta[i].get('pid', 0)})\n Charge      : {p_meta[i].get('charge', 0):+.0f}"
            ), color)

            lw = 4 if p_meta[i]["charge"] != 0 else 1.5
            act = plotter.add_mesh(poly, color=color, line_width=lw, name=mesh_id)
            act.SetVisibility(False)
            
            ctx["particle_actors"].append(act)
            ctx["particle_polydata_lists"].append((poly, full_path))
            self._actor_registry["particles"][i] = act

        ctx["jet_actors"] = []
        for i, jet_geo in enumerate(spatial_data.get("jet_geometries", [])):
            direction = np.array(jet_geo["unit_direction"])
            length, radius = jet_geo["length"], jet_geo["radius"]
            jet_cone = pv.Cone(center=direction * (length / 2.0), direction=-direction, height=length, radius=radius, resolution=30)
            
            mesh_id = f"jet_{i}"
            jet_cone.field_data["mesh_id"] = [mesh_id]
            # Les données textuelles complètes seront récupérées ou injectées de manière croisée
            if mesh_id not in self.tooltip_dict:
                self.tooltip_dict[mesh_id] = (f">> INSPECTING RECONSTRUCTED JET #{i}", "orange")

            act = plotter.add_mesh(jet_cone, color="orange", opacity=0.0, show_edges=True, edge_color="darkorange", name=mesh_id)
            ctx["jet_actors"].append(act)
            self._actor_registry["jet_cones"][i] = act

        ctx["met_actor_line"], ctx["met_actor_tip"] = None, None
        met_data = spatial_data.get("missing_energy", {"pt": 0.0, "phi": 0.0, "vector": (0.0, 0.0, 0.0)})
        if met_data["pt"] > 0.5:
            met_vector = np.array(met_data["vector"])
            met_mesh_line = pv.Line([0, 0, 0], met_vector)
            
            mesh_id = "missing_energy_vector"
            met_mesh_line.field_data["mesh_id"] = [mesh_id]
            self.tooltip_dict[mesh_id] = (f">> WARNING: MISSING TRANSVERSE ENERGY DETECTED\n Unseen pT : {met_data['pt']:.2f} GeV", "red")

            ctx["met_actor_line"] = plotter.add_mesh(met_mesh_line, color="red", line_width=5, name=f"{mesh_id}_line")
            ctx["met_actor_line"].SetVisibility(False)
            
            met_cone = pv.Cone(center=met_vector, direction=met_vector/np.linalg.norm(met_vector), height=0.5, radius=0.25, resolution=60)
            met_cone.field_data["mesh_id"] = [mesh_id]
            ctx["met_actor_tip"] = plotter.add_mesh(met_cone, color="red", name=f"{mesh_id}_tip")
            ctx["met_actor_tip"].SetVisibility(False)
            self._actor_registry["met"] = {"line": ctx["met_actor_line"], "tip": ctx["met_actor_tip"]}

        plotter.camera_position = [(5.0, 5.0, 4.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        plotter.camera.zoom(0.8)

    def _prepare_cinematic_calorimeter(self, plotter, spatial_data, ctx, subplot_idx=(0, 1)):
        """Pre-allocates resource geometries for the static plane unfold Lego view."""
        import numpy as np
        import pyvista as pv

        plotter.subplot(*subplot_idx)
        plotter.set_background(color="#090d16")
        plotter.add_text("CALORIMETER METRIC (LEGO PLOT)", position=(0.05, 0.92), font_size=12, font="courier", color="#fb923c")
        
        lego_floor = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=6.0, j_size=2 * np.pi)
        plotter.add_mesh(lego_floor, color="#1e293b", style="surface", show_edges=True, edge_color="#334155", pickable=False)
        
        plotter.add_point_labels(np.array([[-3.0, -3.3, 0.01], [0.0, -3.3, 0.01], [3.0, -3.3, 0.01]]), ["eta = -3.0", "eta = 0.0", "eta = +3.0"], font_family="courier", font_size=12, show_points=False)
        plotter.add_mesh(pv.Line([-3.0, -3.3, 0.01], [3.0, -3.3, 0.01]), color="#fb923c", line_width=2, pickable=False)
        plotter.add_point_labels(np.array([[-3.3, -np.pi, 0.01], [-3.3, 0.0, 0.01], [-3.3, np.pi, 0.01]]), ["phi = -pi", "phi = 0", "phi = +pi"], font_family="courier", font_size=12, show_points=False)
        plotter.add_mesh(pv.Line([-3.2, -np.pi, 0.01], [-3.2, np.pi, 0.01]), color="#fb923c", line_width=2, pickable=False)

        jet_geometries = spatial_data.get("jet_geometries", [])
        jet_energies = []
        for jet_geo in jet_geometries:
            pt_energy = jet_geo.get("pt", jet_geo.get("energy", jet_geo["length"] * 10.0))
            jet_energies.append(pt_energy)
            
        max_allowed_height = 2.5
        max_e = max(jet_energies) if len(jet_energies) > 0 else 1.0
        v_scale = max_allowed_height / max_e
        
        ctx["lego_mesh_references"] = []
        ctx["max_heights"] = []
        
        for i, jet_geo in enumerate(jet_geometries):
            direction = np.array(jet_geo["unit_direction"])
            eta = jet_geo.get("eta", direction[2] * 1.5)
            phi = jet_geo.get("phi", np.arctan2(direction[1], direction[0]))
            pt_energy = jet_energies[i]
            
            base_box = pv.Box(bounds=[eta - 0.18, eta + 0.18, phi - 0.18, phi + 0.18, 0.0, 1.0])
            
            mesh_id = f"jet_{i}"
            base_box.field_data["mesh_id"] = [mesh_id]
            if mesh_id not in self.tooltip_dict:
                self.tooltip_dict[mesh_id] = (f">> INSPECTING RECONSTRUCTED JET #{i}\n Transverse E : {pt_energy:.2f} GeV", "orange")

            act = plotter.add_mesh(base_box, color="orange", opacity=0.85, show_edges=True, edge_color="white", name=f"lego_tower_{i}")
            act.SetVisibility(False)
            
            ctx["lego_mesh_references"].append((base_box, base_box.points.copy()))
            ctx["max_heights"].append(pt_energy * v_scale)
            self._actor_registry["jet_towers"][i] = act

        plotter.camera_position = [(0.0, -0.01, 9.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        plotter.camera.zoom(0.72)