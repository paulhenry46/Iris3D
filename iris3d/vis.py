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

    def plot_event(self, event: CollisionEvent, p_scale: float = 1.0, j_scale: float = 0.01, B_field: float = 3.8):
        """
        Generates and displays the fluid 3D interactive scene with helical track bending,
        Missing Transverse Energy (MET) balancing and an advanced HUD Overlay.
        """
        plotter = pv.Plotter(window_size=[1024, 768], title=f"Iris3D - Run {event.metadata.run_id} Event {event.metadata.event_id}")
        plotter.set_background(color="#0f172a")               
        plotter.add_axes()
        plotter.show_grid(color="#273549")                     
        plotter.enable_anti_aliasing("msaa", multi_samples=4)  
        
        self._add_detector_geometry(plotter)
        
        vertex = pv.Sphere(radius=0.05, center=(0.0, 0.0, 0.0))
        plotter.add_mesh(vertex, color="magenta", render_points_as_spheres=True, label="Interaction Vertex")
        
        self.tooltip_dict = {}
        self.current_selected_id = None
        
        spatial_data = self.transformer.extract_event_arrays(
            event, p_scale=p_scale, j_scale=j_scale, B_field=B_field,
            detector_ecal_r=self.detector_ecal_r,
            detector_hcal_r=self.calorimeter_outer_radius,
            detector_muon_r=self.detector_muon_r
        )
        
        p_meta = spatial_data["particle_metadata"]
        p_paths = spatial_data["particle_paths"]
        
        for i in range(len(p_paths)):
            points = p_paths[i]
            metadata = p_meta[i]
            p_name = metadata.get("name", f"Track {i}")
            p_pid = metadata.get("pid", 0)
            p_charge = metadata.get("charge", 0)
            
            try:
                source_particle = event.particles[i]
                pt_val = source_particle.pt
                eta_val = source_particle.eta
                phi_val = source_particle.phi
            except (TypeError, KeyError, IndexError, AttributeError):
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
            
            hover_text = (
                f">> INSPECTING TARGET: PARTICLE TRACK #{i}\n"
                f"----------------------------------------\n"
                f" Identity    : {p_name} (PDG: {p_pid})\n"
                f" Momentum pT : {pt_val:.2f} GeV\n"
                f" Pseudo-Rap  : {eta_val:.2f}\n"
                f" Azimuth phi : {phi_val:.2f} rad\n"
                f" Charge      : {p_charge:+.0f}"
            )
            self.tooltip_dict[mesh_id] = (hover_text, color)
            track_mesh.field_data["mesh_id"] = [mesh_id]
            
            if p_charge == 0:
                if p_pid == 22:
                    plotter.add_mesh(track_mesh, color=color, line_width=2.5, opacity=0.9, name=mesh_id)
                else:
                    plotter.add_mesh(track_mesh, color=color, line_width=1, opacity=0.5, name=mesh_id)
            else:
                plotter.add_mesh(track_mesh, color=color, line_width=4, opacity=1.0, name=mesh_id)

        for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
            direction = np.array(jet_geo["unit_direction"])
            length = jet_geo["length"]
            radius = jet_geo["radius"]
            
            cone_center = direction * (length / 2.0)
            jet_cone = pv.Cone(center=cone_center, direction=-direction, height=length, radius=radius, resolution=30)
            
            try:
                source_jet = event.jets[i]
                j_energy = source_jet.energy
                j_eta = source_jet.eta
                j_phi = source_jet.phi
                j_dr = source_jet.delta_r
            except (TypeError, KeyError, IndexError, AttributeError):
                try:
                    j_energy = float(event["jets"]["energy"][i])
                    j_eta = float(event["jets"]["eta"][i])
                    j_phi = float(event["jets"]["phi"][i])
                    j_dr = float(event["jets"]["delta_r"][i])
                except Exception:
                    j_energy, j_eta, j_phi, j_dr = 0.0, 0.0, 0.0, 0.4
            
            mesh_id = f"jet_{i}"
            color = "orange"
            
            jet_hover_text = (
                f">> INSPECTING TARGET: RECONSTRUCTED JET #{i}\n"
                f"----------------------------------------\n"
                f" Transverse E : {j_energy:.2f} GeV\n"
                f" Pseudo-Rap   : {j_eta:.2f}\n"
                f" Azimuth phi  : {j_phi:.2f} rad\n"
                f" Cone Radius  : {j_dr:.2f} (delta_R)"
            )
            self.tooltip_dict[mesh_id] = (jet_hover_text, color)
            jet_cone.field_data["mesh_id"] = [mesh_id]
            
            plotter.add_mesh(jet_cone, color=color, opacity=0.35, show_edges=True, edge_color="darkorange", name=mesh_id)

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
            color = "red"
            
            met_hover_text = (
                f">> WARNING: MISSING TRANSVERSE ENERGY DETECTED\n"
                f"----------------------------------------\n"
                f" Unseen pT    : {met_data['pt']:.2f} GeV\n"
                f" Escape Angle : {met_data['phi']:.2f} rad\n"
                f" Source       : Neutrino / Dark Matter Candidate"
            )
            self.tooltip_dict[mesh_id] = (met_hover_text, color)
            
            met_mesh.field_data["mesh_id"] = [mesh_id]
            met_cone_tip.field_data["mesh_id"] = [mesh_id]
            
            plotter.add_mesh(met_mesh, color=color, line_width=5, opacity=1.0, name=f"{mesh_id}_line")
            plotter.add_mesh(met_cone_tip, color=color, opacity=1.0, name=f"{mesh_id}_tip")

        plotter.add_text(
            "IRIS3D // EVENT DETECTOR HUD ACTIVE\n-----------------------------------\nSelect sub-atomic signature to decode...", 
            position="upper_left", font_size=11, font="courier", color="#38bdf8", name="metadata_banner"
        )

        def picking_callback(mesh):
            if mesh and "mesh_id" in mesh.field_data:
                mesh_id = mesh.field_data["mesh_id"][0]
                if mesh_id in self.tooltip_dict:
                    hud_text, orig_color_name = self.tooltip_dict[mesh_id]
                    rgb_orig = pv.Color(orig_color_name).float_rgb
                    rgb_white = (1.0, 1.0, 1.0)
                    
                    if self.current_selected_id and self.current_selected_id in self.tooltip_dict:
                        old_id = self.current_selected_id
                        _, old_color_name = self.tooltip_dict[old_id]
                        rgb_old = pv.Color(old_color_name).float_rgb
                        if old_id.startswith("missing_energy"):
                            if f"{old_id}_line" in plotter.actors: plotter.actors[f"{old_id}_line"].GetProperty().SetColor(rgb_old)
                            if f"{old_id}_tip" in plotter.actors: plotter.actors[f"{old_id}_tip"].GetProperty().SetColor(rgb_old)
                        else:
                            if old_id in plotter.actors: plotter.actors[old_id].GetProperty().SetColor(rgb_old)
                    
                    if mesh_id.startswith("missing_energy"):
                        if f"{mesh_id}_line" in plotter.actors: plotter.actors[f"{mesh_id}_line"].GetProperty().SetColor(rgb_white)
                        if f"{mesh_id}_tip" in plotter.actors: plotter.actors[f"{mesh_id}_tip"].GetProperty().SetColor(rgb_white)
                    else:
                        if mesh_id in plotter.actors: plotter.actors[mesh_id].GetProperty().SetColor(rgb_white)
                            
                    self.current_selected_id = mesh_id
                    plotter.add_text(hud_text, position="upper_left", font_size=11, font="courier", color="#38bdf8", name="metadata_banner")
                    
        plotter.add_legend(bcolor=None, face="circle")
        plotter.camera_position = [(5.0, 5.0, 4.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        plotter.camera.zoom(0.8)
        plotter.enable_mesh_picking(callback=picking_callback, show=False, left_clicking=True, show_message=False)
        plotter.show()

    def animate_event(self, event: CollisionEvent, p_scale: float = 1.0, j_scale: float = 0.01, B_field: float = 3.8, speed: float = 0.05):
        """
        Advanced stable cinematic version with split screen layout.
        - Left (0, 0): 3D Cylindrical detector view
        - Right (0, 1): Flat 2D/3D Lego Plot (eta vs phi vs pT)
        """
        import numpy as np
        import pyvista as pv

        plotter = pv.Plotter(window_size=[1500, 750], shape=(1, 2), title="Iris3D - Dual Cinematic Display & Lego Plot")
        
        # 1. Extraction des données
        spatial_data = self.transformer.extract_event_arrays(
            event, p_scale=p_scale, j_scale=j_scale, B_field=B_field,
            detector_ecal_r=self.detector_ecal_r,
            detector_hcal_r=self.calorimeter_outer_radius,
            detector_muon_r=self.detector_muon_r
        )
        p_paths = spatial_data["particle_paths"]
        p_meta = spatial_data["particle_metadata"]
        met_data = spatial_data.get("missing_energy", {"pt": 0.0, "phi": 0.0, "vector": (0.0, 0.0, 0.0)})
        
        # ==========================================================
        # INITIALISATION STRICTE DU SUBPLOT GAUCHE : DÉTECTEUR CYLINDRIQUE 3D
        # ==========================================================
        plotter.subplot(0, 0)
        plotter.set_background(color="#0f172a")
        plotter.add_axes()
        plotter.show_grid(color="#273549")
        
        calorimeter_mesh = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), 
            radius=self.calorimeter_outer_radius, height=6.0, resolution=50
        )
        calorimeter_actor = plotter.add_mesh(
            calorimeter_mesh, color="crimson", opacity=0.02, style="surface", 
            show_edges=True, edge_color="firebrick"
        )

        tracker_mesh = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            radius=self.tracker_radius, height=self.tracker_length, resolution=50
        )
        plotter.add_mesh(
            tracker_mesh, color="deepskyblue", opacity=0.08, style="surface",
            show_edges=True, edge_color="dodgerblue", line_width=1,
            name="detector_tracker", pickable=False
        )

        hud = plotter.add_text("IRIS3D // INJECTING BEAMS...", position=(0.02, 0.85), font_size=11, font="courier", color="#38bdf8")

        met_actor_line = None
        met_actor_tip = None
        if met_data["pt"] > 0.5:
            met_vector = np.array(met_data["vector"])
            met_mesh_line = pv.Line([0, 0, 0], met_vector)
            met_actor_line = plotter.add_mesh(met_mesh_line, color="red", line_width=5, pickable=False)
            met_actor_line.SetVisibility(False)
            
            met_cone = pv.Cone(
                center=met_vector, direction=met_vector/np.linalg.norm(met_vector), 
                height=0.5, radius=0.25, resolution=60
            )
            met_actor_tip = plotter.add_mesh(met_cone, color="red", pickable=False)
            met_actor_tip.SetVisibility(False)

        plotter.camera_position = [(5.0, 5.0, 4.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        plotter.camera.zoom(0.8)

        # ==========================================================
        # INITIALISATION STRICTE DU SUBPLOT DROIT : VUE LEGO EN PLAN DES SÉRIES (eta, phi)
        # ==========================================================
        # ==========================================================
        # INITIALISATION STRICTE DU SUBPLOT DROIT : VUE LEGO EN PLAN DES SÉRIES (eta, phi)
        # ==========================================================
        plotter.subplot(0, 1)
        plotter.set_background(color="#090d16")
        
        # Titres du HUD (toujours utiles en haut)
        plotter.add_text("CALORIMETER METRIC (LEGO PLOT)", position=(0.05, 0.92), font_size=12, font="courier", color="#fb923c")
        
        # Le sol déplié de référence (X: -3 à 3, Y: -pi à pi)
        lego_floor = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=6.0, j_size=2 * np.pi)
        plotter.add_mesh(lego_floor, color="#1e293b", style="surface", show_edges=True, edge_color="#334155", pickable=False)
        
        # --- ANCRAGE DES AXES ET LÉGENDES DIRECTEMENT SUR LE PLAN (3D) ---
        
        # 1. Labels pour l'axe Horizontal (X) : Pseudo-rapidité eta de -3.0 à +3.0
        eta_coords = np.array([[-3.0, -3.3, 0.01], [0.0, -3.3, 0.01], [3.0, -3.3, 0.01]])
        eta_labels = ["eta = -3.0", "eta = 0.0", "eta = +3.0"]
        plotter.add_point_labels(
            eta_coords, eta_labels, font_family="courier", font_size=12, 
            point_color=None, text_color="#ffffff", show_points=False, name="eta_3d_labels"
        )
        
        # Ligne de rappel physique sous l'axe X
        eta_axis_line = pv.Line([-3.0, -3.3, 0.01], [3.0, -3.3, 0.01])
        plotter.add_mesh(eta_axis_line, color="#fb923c", line_width=2, pickable=False)

        # 2. Labels pour l'axe Vertical (Y) : Angle azimutal phi de -pi à +pi
        phi_coords = np.array([[-3.3, -np.pi, 0.01], [-3.3, 0.0, 0.01], [-3.3, np.pi, 0.01]])
        phi_labels = ["phi = -pi", "phi = 0", "phi = +pi"]
        plotter.add_point_labels(
            phi_coords, phi_labels, font_family="courier", font_size=12, 
            point_color=None, text_color="#ffffff", show_points=False, name="phi_3d_labels"
        )
        
        # Ligne de rappel physique le long de l'axe Y
        phi_axis_line = pv.Line([-3.2, -np.pi, 0.01], [-3.2, np.pi, 0.01])
        plotter.add_mesh(phi_axis_line, color="#fb923c", line_width=2, pickable=False)

        # --- BARRE D'ÉCHELLE DES COULEURS (SCALAR BAR INVISIBLE À L'ÉCRAN) ---
        dummy_box = pv.Box(bounds=[-0.01, 0.01, -0.01, 0.01, 0.0, 1.0])
        dummy_box["Jet pT (GeV)"] = np.linspace(0.0, 100.0, 8) 
        plotter.add_mesh(
            dummy_box, cmap="Oranges", opacity=0.0, 
            scalar_bar_args={
                "title": "Jet pT Scale", "position_x": 0.88, "position_y": 0.25,
                "height": 0.5, "width": 0.05, "font_family": "courier",
                "title_font_size": 10, "label_font_size": 9, "n_labels": 3, "fmt": "%.0f GeV"
            }
        )

        # Fixation de la caméra par-dessus
        plotter.camera_position = [(0.0, -0.01, 9.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        plotter.camera.zoom(0.72)

        # --- GESTION INTERACTION CLAVIER ---
        state = {"is_paused": False}
        def toggle_pause():
            state["is_paused"] = not state["is_paused"]
        plotter.add_key_event('space', toggle_pause)

        # Lancement de la fenêtre
        plotter.show(auto_close=False, interactive_update=True)

        max_r = self.detector_muon_r
        current_r = -3.0  

        # 2. MAIN LOOP
        while not plotter.render_window.GetInteractor().GetDone():
            
            if not state["is_paused"]:
                current_r += speed
                if current_r > max_r + 0.5:
                    current_r = -3.0  

            # ==========================================================
            # BLOC GAUCHE : ANIMATION DES FAISCEAUX ET TRACES 3D
            # ==========================================================
            plotter.subplot(0, 0)
            
            if current_r < 0:
                if met_actor_line: met_actor_line.SetVisibility(False)
                if met_actor_tip: met_actor_tip.SetVisibility(False)
                
                for i in range(len(p_paths)):
                    if f"particle_{i}" in plotter.actors: plotter.remove_actor(f"particle_{i}")
                for i in range(len(spatial_data.get("jet_geometries", []))):
                    if f"jet_{i}" in plotter.actors: plotter.remove_actor(f"jet_{i}")
                if "shockwave" in plotter.actors: plotter.remove_actor("shockwave")
                
                # Nettoyage synchrone des tours Lego à Droite
                plotter.subplot(0, 1)
                for i in range(len(spatial_data.get("jet_geometries", []))):
                    if f"lego_tower_{i}" in plotter.actors: plotter.remove_actor(f"lego_tower_{i}")
                plotter.subplot(0, 0) # Retour à gauche
                
                calorimeter_actor.GetProperty().SetOpacity(0.02)
                calorimeter_actor.GetProperty().SetEdgeColor(pv.Color("firebrick").float_rgb)
                
                z_pos = -current_r
                beam1 = pv.Line([0, 0, z_pos], [0, 0, max(0, z_pos - 1.0)])
                beam2 = pv.Line([0, 0, -z_pos], [0, 0, min(0, -z_pos + 1.0)])
                
                plotter.add_mesh(beam1, color="#38bdf8", line_width=6, name="beam1", pickable=False)
                plotter.add_mesh(beam2, color="#38bdf8", line_width=6, name="beam2", pickable=False)
                
                vertex = pv.Sphere(radius=0.03, center=(0.0, 0.0, 0.0))
                plotter.add_mesh(vertex, color="gray", name="vertex", pickable=False)
                
                status_txt = "STATUS: PAUSED" if state["is_paused"] else "Status: STEERING PACKETS"
                hud.SetInput(f"IRIS3D // LHC BEAMS APPROACHING\n-------------------------------------------\n{status_txt}")
                
            else:
                if "beam1" in plotter.actors: plotter.remove_actor("beam1")
                if "beam2" in plotter.actors: plotter.remove_actor("beam2")
                
                if current_r < 0.2:
                    vertex = pv.Sphere(radius=0.12, center=(0.0, 0.0, 0.0))
                    plotter.add_mesh(vertex, color="white", name="vertex", pickable=False)
                else:
                    vertex = pv.Sphere(radius=0.05, center=(0.0, 0.0, 0.0))
                    plotter.add_mesh(vertex, color="magenta", name="vertex", pickable=False)

                if current_r >= self.detector_ecal_r:
                    opacity_pulse = 0.15 if current_r < self.calorimeter_outer_radius else 0.06
                    calorimeter_actor.GetProperty().SetOpacity(opacity_pulse)
                    calorimeter_actor.GetProperty().SetEdgeColor(pv.Color("red").float_rgb)
                    
                    if current_r < self.calorimeter_outer_radius + 0.3:
                        wave_radius = current_r
                        shockwave = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=wave_radius, height=5.8, resolution=40)
                        plotter.add_mesh(shockwave, color="orange", opacity=0.25, style="wireframe", line_width=2, name="shockwave", pickable=False)
                    else:
                        if "shockwave" in plotter.actors: plotter.remove_actor("shockwave")
                else:
                    calorimeter_actor.GetProperty().SetOpacity(0.02)
                    calorimeter_actor.GetProperty().SetEdgeColor(pv.Color("firebrick").float_rgb)
                    if "shockwave" in plotter.actors: plotter.remove_actor("shockwave")

                # --- DESSIN DES PARTICULES (SUBPLOT GAUCHE SEUL) ---
                for i, path in enumerate(p_paths):
                    mesh_id = f"particle_{i}"
                    p_charge = p_meta[i]["charge"]
                    color = self._get_particle_color(p_meta[i]["pid"])
                    
                    visible_points = self.transformer.get_path_at_time(path, current_r)
                    
                    if visible_points is not None and len(visible_points) > 0:
                        visible_points = np.array(visible_points)
                        
                        if p_charge != 0 and len(visible_points) > 1:
                            t_brut = np.linspace(0, 1, len(visible_points))
                            t_interp = np.linspace(0, 1, 50)
                            smoothed_points = np.zeros((50, 3))
                            for coord in range(3):
                                smoothed_points[:, coord] = np.interp(t_interp, t_brut, visible_points[:, coord])
                                
                            sub_mesh = pv.Spline(smoothed_points, n_points=100)
                            plotter.add_mesh(sub_mesh, color=color, line_width=4, name=mesh_id, pickable=False)
                        else:
                            p_start = visible_points[0]
                            p_end = visible_points[-1]
                            if np.allclose(p_start, p_end):
                                p_end = p_start + np.array([1e-5, 0.0, 0.0])
                                
                            sub_mesh = pv.Line(p_start, p_end)
                            plotter.add_mesh(sub_mesh, color=color, line_width=1.5, name=mesh_id, pickable=False)
                    else:
                        if mesh_id in plotter.actors: plotter.remove_actor(mesh_id)

                # --- TRAITEMENT DES JETS (AVEC ROUTAGE SUBPLOT APPRÉCIÉ) ---
                # --- TRAITEMENT DES JETS (MORPHING DE POINTS ULTRA-FLUIDE) ---
                for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
                    mesh_id = f"jet_{i}"
                    lego_id = f"lego_tower_{i}"
                    
                    if current_r >= self.tracker_radius:
                        # 1. Dessin des Cônes 3D -> STRICTEMENT À GAUCHE
                        plotter.subplot(0, 0)
                        direction = np.array(jet_geo["unit_direction"])
                        length, radius = jet_geo["length"], jet_geo["radius"]
                        jet_cone = pv.Cone(center=direction * (length / 2.0), direction=-direction, height=length, radius=radius, resolution=30)
                        plotter.add_mesh(jet_cone, color="orange", opacity=min(0.35, (current_r - self.tracker_radius) * 0.2), show_edges=True, edge_color="darkorange", name=mesh_id, pickable=False)
                        
                        # 2. Dessin des Lego Towers (MORPHING SANS RE-CRÉATION) -> STRICTEMENT À DROITE
                        plotter.subplot(0, 1)
                        
                        # CORRECTION ICI : On déballe explicitement le tuple renvoyé par la méthode
                        base_mesh, target_height = self._create_lego_tower_mesh2(jet_geo, current_r=current_r)
                        
                        if target_height > 0.001:
                            # Si l'acteur n'existe pas encore dans ce subplot, on l'initialise avec son maillage de base
                            if lego_id not in plotter.actors:
                                plotter.add_mesh(base_mesh, color="orange", opacity=0.85, show_edges=True, edge_color="white", name=lego_id, pickable=False)
                            
                            # Récupération du maillage VTK actif pour modification directe
                            current_mesh = plotter.actors[lego_id].mapper.dataset
                            
                            # On duplique les points d'origine pour appliquer l'étirement sur l'axe Z
                            animated_points = base_mesh.points.copy()
                            
                            # Les points du "plafond" de notre pv.Box initiale (où Z == 1.0) reçoivent la hauteur dynamique
                            animated_points[base_mesh.points[:, 2] > 0.5, 2] = target_height
                            
                            # Injection des nouveaux points et notification à VTK
                            current_mesh.points = animated_points
                            current_mesh.Modified()
                        else:
                            # Si le front d'onde n'a pas encore atteint l'ECAL, on s'assure que la tour est masquée/détruite
                            if lego_id in plotter.actors:
                                plotter.remove_actor(lego_id)
                    else:
                        plotter.subplot(0, 0)
                        if mesh_id in plotter.actors: plotter.remove_actor(mesh_id)
                        plotter.subplot(0, 1)
                        if lego_id in plotter.actors: plotter.remove_actor(lego_id)

                # --- CONTROL MET (GAUCHE) ---
                plotter.subplot(0, 0)
                if met_data["pt"] > 0.5 and current_r >= self.calorimeter_outer_radius:
                    if met_actor_line: met_actor_line.SetVisibility(True)
                    if met_actor_tip: met_actor_tip.SetVisibility(True)
                else:
                    if met_actor_line: met_actor_line.SetVisibility(False)
                    if met_actor_tip: met_actor_tip.SetVisibility(False)

                state_label = "|| PAUSED" if state["is_paused"] else ('TRACKING CORE' if current_r < self.detector_ecal_r else 'CALORIMETER SHOWER')
                hud.SetInput(
                    f"IRIS3D // TIME-OF-FLIGHT SIMULATION ACTIVE\n"
                    f"-------------------------------------------\n"
                    f"Wavefront Radius : {current_r:.2f} meters\n"
                    f"Sub-atomic State : {state_label}"
                )

            # Cadencement global
            plotter.update(16, force_redraw=True)
            
        plotter.close()
    
    def _create_lego_tower_mesh2(self, jet_geo, current_r: float = None):
        """
        Calcule la hauteur dynamique et renvoie la géométrie de base ainsi que la hauteur cible.
        """
        import numpy as np
        import pyvista as pv

        direction = np.array(jet_geo["unit_direction"])
        length = jet_geo["length"]
        
        eta = jet_geo.get("eta", direction[2] * 1.5)
        phi = jet_geo.get("phi", np.arctan2(direction[1], direction[0]))
        pt_energy = jet_geo.get("pt", length * 10.0)
        
        max_height = pt_energy * 0.08
        
        if current_r is not None:
            if current_r < self.detector_ecal_r:
                current_height = 0.0
            else:
                growth_range = self.calorimeter_outer_radius - self.detector_ecal_r
                progress = (current_r - self.detector_ecal_r) / growth_range
                current_height = max_height * min(1.0, max(0.0, progress))
        else:
            current_height = max_height

        # On crée une boîte de base (gabarit) centrée au bon endroit
        base_box = pv.Box(bounds=[
            eta - 0.18, eta + 0.18,
            phi - 0.18, phi + 0.18,
            0.0, 1.0 # Hauteur normalisée à 1.0 pour faciliter l'étirement
        ])
        
        return base_box, current_height
    
    def plot_lego_standalone(self, event: CollisionEvent, p_scale: float = 1.0, j_scale: float = 0.01, B_field: float = 3.8):
        """
        Renders a static, standalone 3D Lego Plot (eta vs phi vs pT) for event analysis.
        """
        import numpy as np
        import pyvista as pv

        spatial_data = self.transformer.extract_event_arrays(
            event, p_scale=p_scale, j_scale=j_scale, B_field=B_field,
            detector_ecal_r=self.detector_ecal_r,
            detector_hcal_r=self.calorimeter_outer_radius,
            detector_muon_r=self.detector_muon_r
        )

        plotter = pv.Plotter(window_size=[800, 600], title="Iris3D - Standalone Lego Plot")
        plotter.set_background(color="#090d16")
        plotter.add_text("STANDALONE LEGO PLOT (eta vs phi)", position=(0.05, 0.92), font_size=12, font="courier", color="#fb923c")

        lego_floor = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=6.0, j_size=2 * np.pi)
        plotter.add_mesh(lego_floor, color="#1e293b", style="surface", show_edges=True, edge_color="#334155", pickable=False)
        
        for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
            tower_mesh = self._create_lego_tower_mesh(jet_geo, current_r=None)
            if tower_mesh:
                plotter.add_mesh(tower_mesh, color="orange", opacity=0.8, show_edges=True, edge_color="white", pickable=False)

        plotter.add_axes(labels_off=False, box=True)
        plotter.camera_position = [(0.0, -0.01, 9.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        plotter.show()