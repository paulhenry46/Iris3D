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
        self.detector_ecal_r=1.75      # Bord externe exact du calorimètre
        self.detector_muon_r=4.0
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
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=self.tracker_radius,
            height=self.tracker_length,
            resolution=50
        )
        
        plotter.add_mesh(
            tracker,
            color="deepskyblue",
            opacity=0.08,
            style="surface",
            show_edges=True,
            edge_color="dodgerblue",
            line_width=1,
            name="detector_tracker",
            pickable=False
        )

        calorimeter_length = 6.0
        
        calorimeter = pv.Cylinder(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=self.calorimeter_outer_radius,
            height=calorimeter_length,
            resolution=50
        )
        
        plotter.add_mesh(
            calorimeter,
            color="crimson",
            opacity=0.04,
            style="surface",
            show_edges=True,
            edge_color="firebrick",
            line_width=1,
            name="detector_calorimeter",
            pickable=False
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
        
        # Le dictionnaire stocke : { mesh_id: (text_hud, original_color_name) }
        self.tooltip_dict = {}
        self.current_selected_id = None
        
        spatial_data = self.transformer.extract_event_arrays(
            event, 
            p_scale=p_scale, 
            j_scale=j_scale, 
            B_field=B_field,
            detector_ecal_r=self.detector_ecal_r,         # Moitié du calorimètre (Ex: entre le tracker 1.5 et la fin 2.8)
            detector_hcal_r=self.calorimeter_outer_radius,         # Bord externe exact de ton calorimètre de vis.py
            detector_muon_r=self.detector_muon_r          # Zone libre externe dédiée aux muons
        )
        
        p_meta = spatial_data["particle_metadata"]
        p_paths = spatial_data["particle_paths"]
        
        # 4. Traitement et Style des Traces
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
                    pt_val = 0.0
                    eta_val = 0.0
                    phi_val = 0.0
            
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

        # 5. Process and Render Jet Cones
        for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
            direction = np.array(jet_geo["unit_direction"])
            length = jet_geo["length"]
            radius = jet_geo["radius"]
            
            cone_center = direction * (length / 2.0)
            jet_cone = pv.Cone(
                center=cone_center, direction=-direction, height=length, radius=radius, resolution=30
            )
            
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
                    j_energy = 0.0
                    j_eta = 0.0
                    j_phi = 0.0
                    j_dr = 0.4
            
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
            
            plotter.add_mesh(
                jet_cone, color=color, opacity=0.35, show_edges=True, edge_color="darkorange", name=mesh_id
            )

        # --- 5.5. Process and Render Missing Transverse Energy (MET) ---
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
            met_cone_tip = pv.Cone(
                center=met_vector, direction=met_direction, height=0.5, radius=0.25, resolution=20
            )
            
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

        # Initialisation de la bannière HUD technologique
        plotter.add_text(
            "IRIS3D // EVENT DETECTOR HUD ACTIVE\n"
            "-----------------------------------\n"
            "Select sub-atomic signature to decode...", 
            position="upper_left", 
            font_size=11, 
            font="courier",
            color="#38bdf8",  
            name="metadata_banner"
        )

        # Define picker callback function
        def picking_callback(mesh):
            if mesh and "mesh_id" in mesh.field_data:
                mesh_id = mesh.field_data["mesh_id"][0]
                
                if mesh_id in self.tooltip_dict:
                    hud_text, orig_color_name = self.tooltip_dict[mesh_id]
                    
                    # Conversion propre du nom de la couleur vers un tuple RGB compatible VTK via PyVista
                    rgb_orig = pv.Color(orig_color_name).float_rgb
                    rgb_white = (1.0, 1.0, 1.0)
                    
                    # 1. Restauration de l'ancien acteur sélectionné
                    if self.current_selected_id and self.current_selected_id in self.tooltip_dict:
                        old_id = self.current_selected_id
                        _, old_color_name = self.tooltip_dict[old_id]
                        rgb_old = pv.Color(old_color_name).float_rgb
                        
                        if old_id.startswith("missing_energy"):
                            if f"{old_id}_line" in plotter.actors:
                                plotter.actors[f"{old_id}_line"].GetProperty().SetColor(rgb_old)
                            if f"{old_id}_tip" in plotter.actors:
                                plotter.actors[f"{old_id}_tip"].GetProperty().SetColor(rgb_old)
                        else:
                            if old_id in plotter.actors:
                                plotter.actors[old_id].GetProperty().SetColor(rgb_old)
                    
                    # 2. Application de la surbrillance blanche sur le nouvel acteur sélectionné
                    if mesh_id.startswith("missing_energy"):
                        if f"{mesh_id}_line" in plotter.actors:
                            plotter.actors[f"{mesh_id}_line"].GetProperty().SetColor(rgb_white)
                        if f"{mesh_id}_tip" in plotter.actors:
                            plotter.actors[f"{mesh_id}_tip"].GetProperty().SetColor(rgb_white)
                    else:
                        if mesh_id in plotter.actors:
                            plotter.actors[mesh_id].GetProperty().SetColor(rgb_white)
                            
                    self.current_selected_id = mesh_id
                    
                    # 3. Mise à jour du texte du HUD
                    plotter.add_text(
                        hud_text, 
                        position="upper_left", 
                        font_size=11, 
                        font="courier",
                        color="#38bdf8",
                        name="metadata_banner"
                    )
                    
        # 6. Open the engine window with clean default settings
        plotter.add_legend(bcolor=None, face="circle")
        plotter.camera_position = [
            (5.0, 5.0, 4.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0)
        ]
        plotter.camera.zoom(0.8)
        
        plotter.enable_mesh_picking(callback=picking_callback, show=False, left_clicking=True, show_message=False)
        
        plotter.show()

    def animate_event(self, event: CollisionEvent, p_scale: float = 1.0, j_scale: float = 0.01, B_field: float = 3.8, speed: float = 0.05):
        """
        Version cinématique avancée stable. Includes a pre-collision beam animation,
        a localized wavefront shockwave flash upon calorimeter impact, stabilized MET, 
        and clean photon lines without artifacts.
        """
        import numpy as np
        import pyvista as pv

        plotter = pv.Plotter(window_size=[1024, 768], title="Iris3D - Cinematic Event Display")
        plotter.set_background(color="#0f172a")
        plotter.add_axes()
        plotter.show_grid(color="#273549")
        
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
        
        # Structure géométrique : Le Calorimètre
        calorimeter_mesh = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), 
            radius=self.calorimeter_outer_radius, height=6.0, resolution=50
        )
        calorimeter_actor = plotter.add_mesh(
            calorimeter_mesh, color="crimson", opacity=0.02, style="surface", 
            show_edges=True, edge_color="firebrick"
        )

        # Structure géométrique : Le Tracker intégré
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

        # --- PRÉ-INSTANCIATION DE LA MET UNIQUE (Évite le gigotement et le clignotement) ---
        met_actor_line = None
        met_actor_tip = None
        if met_data["pt"] > 0.5:
            met_vector = np.array(met_data["vector"])
            # Ligne de corps du vecteur MET
            met_mesh_line = pv.Line([0, 0, 0], met_vector)
            met_actor_line = plotter.add_mesh(met_mesh_line, color="red", line_width=5, pickable=False)
            met_actor_line.SetVisibility(False)  # Masqué au début (temps négatif)
            
            # Cône de pointe haute résolution stable (60 facettes parfaites)
            met_cone = pv.Cone(
                center=met_vector, direction=met_vector/np.linalg.norm(met_vector), 
                height=0.5, radius=0.25, resolution=60
            )
            met_actor_tip = plotter.add_mesh(met_cone, color="red", pickable=False)
            met_actor_tip.SetVisibility(False)  # Masqué au début

        # Position initiale fixe de la caméra
        plotter.camera_position = [(5.0, 5.0, 4.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        plotter.camera.zoom(0.8)
        
        plotter.show(auto_close=False, interactive_update=True)

        max_r = self.detector_muon_r
        current_r = -3.0  # On commence à t = -3.0 pour l'injection

        # 2. BOUCLE PRINCIPALE
        while not plotter.render_window.GetInteractor().GetDone():
            
            current_r += speed
            if current_r > max_r + 0.5:
                current_r = -3.0  # On reboucle au tout début

            # ==========================================================
            # PHASE 1 : ANIMATION PRÉ-COLLISION (Faisceaux entrants)
            # ==========================================================
            if current_r < 0:
                # Masquage immédiat de la MET fixe
                if met_actor_line: met_actor_line.SetVisibility(False)
                if met_actor_tip: met_actor_tip.SetVisibility(False)
                
                # Nettoyage complet des acteurs de l'explosion précédente
                for i in range(len(p_paths)):
                    if f"particle_{i}" in plotter.actors: plotter.remove_actor(f"particle_{i}")
                for i in range(len(spatial_data.get("jet_geometries", []))):
                    if f"jet_{i}" in plotter.actors: plotter.remove_actor(f"jet_{i}")
                if "shockwave" in plotter.actors: plotter.remove_actor("shockwave")
                
                # Remise à zéro visuelle du calorimètre
                calorimeter_actor.GetProperty().SetOpacity(0.02)
                calorimeter_actor.GetProperty().SetEdgeColor(pv.Color("firebrick").float_rgb)
                
                # Position et déplacement des faisceaux (axe Z)
                z_pos = -current_r
                beam1 = pv.Line([0, 0, z_pos], [0, 0, max(0, z_pos - 1.0)])
                beam2 = pv.Line([0, 0, -z_pos], [0, 0, min(0, -z_pos + 1.0)])
                
                plotter.add_mesh(beam1, color="#38bdf8", line_width=6, name="beam1", pickable=False)
                plotter.add_mesh(beam2, color="#38bdf8", line_width=6, name="beam2", pickable=False)
                
                # Vertex en attente
                vertex = pv.Sphere(radius=0.03, center=(0.0, 0.0, 0.0))
                plotter.add_mesh(vertex, color="gray", name="vertex", pickable=False)
                
                hud.SetInput("IRIS3D // LHC BEAMS APPROACHING\n-------------------------------------------\nStatus: STEERING PACKETS")
                
            # ==========================================================
            # PHASE 2 : IMPACT & EXPULSION (Collision active)
            # ==========================================================
            else:
                # Suppression des paquets de faisceaux initiaux
                if "beam1" in plotter.actors: plotter.remove_actor("beam1")
                if "beam2" in plotter.actors: plotter.remove_actor("beam2")
                
                # Flash transitoire du Vertex à l'impact (t=0)
                if current_r < 0.2:
                    vertex = pv.Sphere(radius=0.12, center=(0.0, 0.0, 0.0))
                    plotter.add_mesh(vertex, color="white", name="vertex", pickable=False)
                else:
                    vertex = pv.Sphere(radius=0.05, center=(0.0, 0.0, 0.0))
                    plotter.add_mesh(vertex, color="magenta", name="vertex", pickable=False)

                # --- CHRONOLOGIE DÉTECTEUR & ONDE DE CHOC ---
                if current_r >= self.detector_ecal_r:
                    opacity_pulse = 0.15 if current_r < self.calorimeter_outer_radius else 0.06
                    calorimeter_actor.GetProperty().SetOpacity(opacity_pulse)
                    calorimeter_actor.GetProperty().SetEdgeColor(pv.Color("red").float_rgb)
                    
                    # Génération de la shockwave filaire orange sur le front d'onde
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

                # --- DESSIN DES PARTICULES (Zéro points carrés) ---
                for i, path in enumerate(p_paths):
                    mesh_id = f"particle_{i}"
                    p_charge = p_meta[i]["charge"]
                    color = self._get_particle_color(p_meta[i]["pid"])
                    
                    visible_points = self.transformer.get_path_at_time(path, current_r)
                    
                    if visible_points is not None and len(visible_points) > 0:
                        visible_points = np.array(visible_points)
                        
                        if p_charge != 0 and len(visible_points) > 1:
                            # --- PARTICULES CHARGÉES (Spline continue) ---
                            t_brut = np.linspace(0, 1, len(visible_points))
                            t_interp = np.linspace(0, 1, 50)
                            smoothed_points = np.zeros((50, 3))
                            for coord in range(3):
                                smoothed_points[:, coord] = np.interp(t_interp, t_brut, visible_points[:, coord])
                                
                            sub_mesh = pv.Spline(smoothed_points, n_points=100)
                            plotter.add_mesh(sub_mesh, color=color, line_width=4, name=mesh_id, pickable=False)
                        else:
                            # --- PARTICULES NEUTRES / PHOTONS (Ligne pure sans aucun point) ---
                            p_start = visible_points[0]
                            p_end = visible_points[-1]
                            
                            # Sécurité VTK si le segment est encore confondu à l'origine
                            if np.allclose(p_start, p_end):
                                p_end = p_start + np.array([1e-5, 0.0, 0.0])
                                
                            sub_mesh = pv.Line(p_start, p_end)
                            plotter.add_mesh(sub_mesh, color=color, line_width=1.5, name=mesh_id, pickable=False)
                    else:
                        if mesh_id in plotter.actors: plotter.remove_actor(mesh_id)

                # --- ANIMATION DES JETS ---
                for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
                    mesh_id = f"jet_{i}"
                    if current_r >= self.tracker_radius:
                        direction = np.array(jet_geo["unit_direction"])
                        length, radius = jet_geo["length"], jet_geo["radius"]
                        jet_cone = pv.Cone(center=direction * (length / 2.0), direction=-direction, height=length, radius=radius, resolution=30)
                        plotter.add_mesh(jet_cone, color="orange", opacity=min(0.35, (current_r - self.tracker_radius) * 0.2), show_edges=True, edge_color="darkorange", name=mesh_id, pickable=False)
                    else:
                        if mesh_id in plotter.actors: plotter.remove_actor(mesh_id)

                # --- CONTROL DE LA VISIBILITÉ DE LA MET STABLE ---
                if met_data["pt"] > 0.5 and current_r >= self.calorimeter_outer_radius:
                    if met_actor_line: met_actor_line.SetVisibility(True)
                    if met_actor_tip: met_actor_tip.SetVisibility(True)
                else:
                    if met_actor_line: met_actor_line.SetVisibility(False)
                    if met_actor_tip: met_actor_tip.SetVisibility(False)

                # Mise à jour des données textuelles du HUD
                hud.SetInput(
                    f"IRIS3D // TIME-OF-FLIGHT SIMULATION ACTIVE\n"
                    f"-------------------------------------------\n"
                    f"Wavefront Radius : {current_r:.2f} meters\n"
                    f"Sub-atomic State : {'TRACKING CORE' if current_r < self.detector_ecal_r else 'CALORIMETER SHOWER'}"
                )

            # Cadencement à 16ms (~60 FPS) avec rafraîchissement d'interactivité souris
            plotter.update(16, force_redraw=True)
            
        plotter.close()