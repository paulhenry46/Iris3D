import numpy as np
import pyvista as pv
from typing import Optional
from iris3d.models import CollisionEvent
from iris3d.core import CoordinateTransformer
from iris3d.themes import THEMES

class EventVisualizer:
    """
    Renders high-energy physics collision events in a highly fluid,
    GPU-accelerated 3D canvas using PyVista (VTK), featuring interactive
    picking tooltips, realistic magnetic field bending, and style-coded tracks.
    """
    def __init__(self, theme_name: str = "cyberpunk"):
        self.transformer = CoordinateTransformer()
        self.current_selected_id = None  # Tracker pour l'objet actuellement sélectionné
        self.calorimeter_outer_radius = 2.8
        self.detector_ecal_r = 1.75      # Bord externe exact du calorimètre
        self.detector_muon_r = 4.0
        self.tracker_radius = 1.5
        self.tracker_length = 5.0

        if theme_name not in THEMES:
            print(f"[Iris3D CORE] Warning: Theme '{theme_name}' not found. Falling back to 'cyberpunk'.")
            theme_name = "cyberpunk"
        
        # 2. Copie locale du blueprint pour accès direct et ultra-rapide
        self.theme = THEMES[theme_name]

        if self.theme['dark']:
            pv.set_plot_theme("dark")
            
    def _get_particle_color(self, pid: int) -> str:
        """Assigns professional color-coding based on Particle ID (PDG code)."""
        particle_theme = self.theme.get("particles", {})
        
        return particle_theme.get(pid, particle_theme.get("default", "#94a3b8"))

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
            tracker, color=self.theme["detector_tracker"], opacity=0.08, style="surface",
            show_edges=True, edge_color=self.theme["detector_tracker_edge"], line_width=1,
            name="detector_tracker", pickable=False
        )

        calorimeter_length = 6.0
        calorimeter = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            radius=self.calorimeter_outer_radius, height=calorimeter_length, resolution=50
        )
        plotter.add_mesh(
            calorimeter, color=self.theme["detector_ecal"], opacity=0.04, style="surface",
            show_edges=True, edge_color=self.theme["detector_ecal_edge"], line_width=1,
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
        self._multi_select_ids = []
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
            self._plot_3d_detector(plotter, spatial_data, event=event, run_id=run_id, event_id=event_id, is_cinematic=False, subplot_idx=(0, 0))
            
            self._plot_lego_calorimeter(plotter, spatial_data, event=event, is_cinematic=False, subplot_idx=(0, 1))
        elif mode == "detector":
            self._plot_3d_detector(plotter, spatial_data, event=event, run_id=run_id, event_id=event_id, is_cinematic=False, subplot_idx=(0, 0))
        elif mode == "lego":
            self._plot_lego_calorimeter(plotter, spatial_data, event=event, is_cinematic=False, subplot_idx=(0, 1))

        # 6. Injection de la logique de picking unifiée
        # 6. Logiciel de Picking Unifié + Calculateur de Masse Invariante (Shift + Clic)
        def picking_callback(mesh):
            if not mesh or "mesh_id" not in mesh.field_data:
                return
            
            mesh_id = mesh.field_data["mesh_id"][0]
            if mesh_id not in self.tooltip_dict:
                return

            hud_text, orig_color_name = self.tooltip_dict[mesh_id]
            rgb_white = (1.0, 1.0, 1.0)
            
            # --- ÉTAPE A : DÉTECTION DU MODE (SHIFT ENFONCÉ OU NON) ---
            # PyVista permet de vérifier l'état du clavier via le plotter
            shift_pressed = plotter.iren.interactor.GetShiftKey()

            if shift_pressed and mesh_id.startswith("particle_"):
                # --- MODE ANALYSE MULTI-PARTICULES ---
                p_idx = int(mesh_id.split('_')[1])
                
                # Éviter les doublons dans la sélection
                if mesh_id not in self._multi_select_ids:
                    self._multi_select_ids.append(mesh_id)
                    # Coloration flash de la particule sélectionnée pour l'analyse
                    if p_idx in self._actor_registry["particles"]:
                        self._actor_registry["particles"][p_idx]["actor"].GetProperty().SetColor((0.0, 1.0, 1.0)) # Cyan pour l'analyse
                        self._actor_registry["particles"][p_idx]["actor"].GetProperty().SetLineWidth(6)
                
                # Si on a collecté 2 particules, on lance le calcul de physique
                if len(self._multi_select_ids) == 2:
                    id1, id2 = self._multi_select_ids[0], self._multi_select_ids[1]
                    idx1, idx2 = int(id1.split('_')[1]), int(id2.split('_')[1])
                    
                    p1_data = self._actor_registry["particles"][idx1]
                    p2_data = self._actor_registry["particles"][idx2]
                    
                    # Dictionnaire des masses au repos standards (en GeV)
                    mass_map = {11: 0.000511, 13: 0.10566, 211: 0.13957, 22: 0.0} # e, mu, pi+/-, photon
                    
                    # Extraction des cinématiques
                    pt1, eta1, phi1 = p1_data["pt"], p1_data["eta"], p1_data["phi"]
                    pt2, eta2, phi2 = p2_data["pt"], p2_data["eta"], p2_data["phi"]
                    
                    pid1 = p1_data.get("pid", 13) # Muon par défaut si non spécifié
                    pid2 = p2_data.get("pid", 13)
                    m1 = mass_map.get(abs(int(pid1)), 0.139) # par défaut masse du pion si inconnu
                    m2 = mass_map.get(abs(int(pid2)), 0.139)
                    
                    # Reconstruction des quadri-vecteurs
                    px1, py1, pz1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1), pt1 * np.sinh(eta1)
                    p1_mag = np.sqrt(px1**2 + py1**2 + pz1**2)
                    E1 = np.sqrt(p1_mag**2 + m1**2)
                    
                    px2, py2, pz2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2), pt2 * np.sinh(eta2)
                    p2_mag = np.sqrt(px2**2 + py2**2 + pz2**2)
                    E2 = np.sqrt(p2_mag**2 + m2**2)
                    
                    # Calculs finaux de la résonance
                    sum_E = E1 + E2
                    sum_px, sum_py, sum_pz = px1 + px2, py1 + py2, pz1 + pz2
                    m_inv2 = sum_E**2 - (sum_px**2 + sum_py**2 + sum_pz**2)
                    m_inv = np.sqrt(max(0.0, m_inv2))
                    
                    # Calcul du Delta R
                    d_eta = eta1 - eta2
                    d_phi = phi1 - phi2
                    while d_phi > np.pi:  d_phi -= 2.0 * np.pi
                    while d_phi < -np.pi: d_phi += 2.0 * np.pi
                    delta_R = np.sqrt(d_eta**2 + d_phi**2)
                    
                    # Formatage du HUD d'Analyse Relativiste
                    hud_text = (
                        f"========================================\n"
                        f" 🔬 KINEMATIC RESONANCE ANALYSIS       \n"
                        f"========================================\n"
                        f" Track A : ID #{idx1} (pT={pt1:.2f} GeV, η={eta1:.2f})\n"
                        f" Track B : ID #{idx2} (pT={pt2:.2f} GeV, η={eta2:.2f})\n"
                        f" ----------------------------------------\n"
                        f" Spatial Separation ΔR : {delta_R:.4f}\n"
                        f" >> INVARIANT MASS M(ab) : {m_inv:.3f} GeV <<\n"
                        f"========================================"
                    )
                    
                    # Reset de la liste pour la prochaine analyse
                    self._multi_select_ids = []
                    
                else:
                    hud_text = f" -> Particule #{p_idx} sélectionnée pour analyse.\n Maintenez [SHIFT] et cliquez sur une 2ème particule..."

            else:
                # --- MODE CLIC CLASSIQUE SINGLE-OBJECT ---
                # Si l'utilisateur clique normalement, on reset le mode analyse et ses couleurs cyan
                for old_id in self._multi_select_ids:
                    idx = int(old_id.split('_')[1])
                    if idx in self._actor_registry["particles"]:
                        _, col = self.tooltip_dict[old_id]
                        self._actor_registry["particles"][idx]["actor"].GetProperty().SetColor(pv.Color(col).float_rgb)
                self._multi_select_ids = []

                # Application du comportement standard de nettoyage / reset précédent
                if self.current_selected_id:
                    old_id = self.current_selected_id
                    _, old_color_name = self.tooltip_dict[old_id]
                    rgb_old = pv.Color(old_color_name).float_rgb
                    
                    if old_id.startswith("particle_"):
                        idx = int(old_id.split('_')[1])
                        if idx in self._actor_registry["particles"]:
                            self._actor_registry["particles"][idx]["actor"].GetProperty().SetColor(rgb_old)
                    elif old_id.startswith("jet_"):
                        idx = int(old_id.split('_')[1])
                        if idx in self._actor_registry["jet_cones"]:
                            self._actor_registry["jet_cones"][idx]["actor"].GetProperty().SetColor(rgb_old)
                        if idx in self._actor_registry["jet_towers"]:
                            lego_tower = self._actor_registry["jet_towers"][idx]
                            if isinstance(lego_tower, dict) and "actor" in lego_tower:
                                lego_tower["actor"].GetProperty().SetColor(rgb_old)
                            elif hasattr(lego_tower, "GetProperty"):
                                lego_tower.GetProperty().SetColor(rgb_old)

                # Application de la surbrillance blanche sur l'élément unique cliqué
                if mesh_id.startswith("particle_"):
                    idx = int(mesh_id.split('_')[1])
                    if idx in self._actor_registry["particles"]:
                        self._actor_registry["particles"][idx]["actor"].GetProperty().SetColor(rgb_white)
                elif mesh_id.startswith("jet_"):
                    idx = int(mesh_id.split('_')[1])
                    if idx in self._actor_registry["jet_cones"]:
                        self._actor_registry["jet_cones"][idx]["actor"].GetProperty().SetColor(rgb_white)
                    if idx in self._actor_registry["jet_towers"]:
                        lego_tower = self._actor_registry["jet_towers"][idx]
                        if isinstance(lego_tower, dict) and "actor" in lego_tower:
                            lego_tower["actor"].GetProperty().SetColor(rgb_white)
                        elif hasattr(lego_tower, "GetProperty"):
                            lego_tower.GetProperty().SetColor(rgb_white)

                self.current_selected_id = mesh_id

            # --- MISE À JOUR COMMUNE DU CANVAS ---
            if mode in ["both", "detector"]:
                plotter.subplot(0, 0)
                plotter.add_text(hud_text, position="upper_left", font_size=11, font="courier", color=self.theme["hud_text"], name="metadata_banner")
            plotter.render()
        # Activation globale unique
        self._setup_interactive_shortcuts(plotter)
        plotter.enable_mesh_picking(callback=picking_callback, show=False, left_clicking=True, show_message=False)
        plotter.show()
    
    def _plot_3d_detector(self, plotter, spatial_data, event=None, run_id="N/A", event_id="N/A", ctx=None, is_cinematic: bool = False, subplot_idx=(0, 0)):
        """
        Unified method to draw the central 3D tracking system and geometric signatures,
        supporting both static exploration and high-performance cinematic simulation contexts.
        """
        import numpy as np
        import pyvista as pv

        plotter.subplot(*subplot_idx)
        plotter.set_background(color=self.theme["bg_detector"])
        plotter.add_axes(labels_off=True)
        plotter.show_grid(color=self.theme["grid_color"])                     
        plotter.enable_anti_aliasing("msaa", multi_samples=4)  
        
        # 1. GÉOMÉTRIE DU DÉTECTEUR ET INFRASTRUCTURE DE FOND
        if is_cinematic and ctx is not None:
            # Éléments passifs spécifiques à l'animation
            ctx["calorimeter_mesh"] = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=self.calorimeter_outer_radius, height=6.0, resolution=50)
            ctx["calorimeter_actor"] = plotter.add_mesh(ctx["calorimeter_mesh"], color=self.theme["detector_ecal"], opacity=0.02, style="surface", show_edges=True, edge_color=self.theme["detector_ecal_edge"])

            tracker_mesh = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=self.tracker_radius, height=self.tracker_length, resolution=50)
            plotter.add_mesh(tracker_mesh, color=self.theme["detector_tracker"], opacity=0.08, style="surface", show_edges=True, edge_color=self.theme["detector_tracker_edge"], pickable=False)

            ctx["hud"] = plotter.add_text("IRIS3D // INITIALIZING...", position=(0.02, 0.85), font_size=11, font="courier", color=self.theme["hud_text"])

            beam1 = pv.Line([0, 0, 5.0], [0, 0, 0.0])
            beam2 = pv.Line([0, 0, -5.0], [0, 0, 0.0])
            ctx["beam1_actor"] = plotter.add_mesh(beam1, color=self.theme["beam"], line_width=6, pickable=False)
            ctx["beam2_actor"] = plotter.add_mesh(beam2, color=self.theme["beam"], line_width=6, pickable=False)
            
            vertex_mesh = pv.Sphere(radius=0.05, center=(0.0, 0.0, 0.0))
            ctx["vertex_actor"] = plotter.add_mesh(vertex_mesh, color=self.theme["vertex_anim"], pickable=False)

            shockwave_base = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=1.0, height=5.8, resolution=40)
            ctx["shockwave_actor"] = plotter.add_mesh(shockwave_base, color=self.theme["shockwave"], opacity=0.0, style="wireframe", line_width=2, pickable=False)

            ctx["particle_actors"] = []
            ctx["particle_polydata_lists"] = []
            ctx["jet_actors"] = []
        else:
            # Mode statique : Géométrie simplifiée par défaut
            self._add_detector_geometry(plotter)
            vertex = pv.Sphere(radius=0.05, center=(0.0, 0.0, 0.0))
            plotter.add_mesh(vertex, color=self.theme["vertex_static"], render_points_as_spheres=True, label="Interaction Vertex")

        p_meta = spatial_data["particle_metadata"]
        p_paths = spatial_data["particle_paths"]

        # 2. RENDU ET ALLOCATION DES TRACES DE PARTICULES
        for i in range(len(p_paths)):
            points = p_paths[i]
            metadata = p_meta[i]
            p_name = metadata.get("name", f"Track {i}")
            p_pid = metadata.get("pid", 0)
            p_charge = metadata.get("charge", 0)
            
            # Parsing sécurisé des propriétés cinématiques
            try:
                source_particle = event.particles[i] if event else None
                pt_val, eta_val, phi_val = source_particle.pt, source_particle.eta, source_particle.phi
            except Exception:
                try:
                    pt_val = float(event["particles"]["pt"][i]) if event else 0.0
                    eta_val = float(event["particles"]["eta"][i]) if event else 0.0
                    phi_val = float(event["particles"]["phi"][i]) if event else 0.0
                except Exception:
                    pt_val, eta_val, phi_val = 0.0, 0.0, 0.0
            
            # Génération du Mesh topologique de la trace
            if p_charge != 0:
                track_mesh = pv.Spline(points, len(points))
            else:
                track_mesh = pv.PolyData(points)
                if p_pid == 22:  # Photons (Ligne pointillée via connectivité segmentée)
                    lines_connectivity = []
                    for idx in range(0, len(points) - 1, 2):
                        lines_connectivity.extend([2, idx, idx + 1])
                    track_mesh.lines = np.array(lines_connectivity, dtype=np.int32)
                else:  
                    cells = np.hstack([[len(points)], np.arange(len(points))])
                    track_mesh.lines = cells
            
            color = self._get_particle_color(p_pid)
            mesh_id = f"particle_{i}"
            
            # Enregistrement des chaînes HUD dans le registre d'infobulles
            self.tooltip_dict[mesh_id] = ((
                f">> INSPECTING TARGET: PARTICLE TRACK #{i}\n----------------------------------------\n"
                f" Identity    : {p_name} (PDG: {p_pid})\n Momentum pT : {pt_val:.2f} GeV\n"
                f" Pseudo-Rap  : {eta_val:.2f}\n Azimuth phi : {phi_val:.2f} rad\n Charge      : {p_charge:+.0f}"
            ), color)
            track_mesh.field_data["mesh_id"] = [mesh_id]
            
            # Paramètres d'affichage sélectifs
            lw = 4 if p_charge != 0 else (2.5 if p_pid == 22 else 1.5)
            op = 1.0 if p_charge != 0 else (0.9 if p_pid == 22 else 0.5)
            
            act = plotter.add_mesh(track_mesh, color=color, line_width=lw, opacity=op, name=mesh_id)
            
            if is_cinematic and ctx is not None:
                act.SetVisibility(False)
                ctx["particle_actors"].append(act)
                ctx["particle_polydata_lists"].append((track_mesh, np.array(points)))
                
            #self._actor_registry["particles"][i] = act

            self._actor_registry["particles"][i] = {
                    "actor": act,
                    "pt": pt_val,
                    "eta": eta_val,
                    "phi": phi_val
                }

        # 3. RENDU DES CÔNES DE JETS
        for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
            direction = np.array(jet_geo["unit_direction"])
            length, radius = jet_geo["length"], jet_geo["radius"]
            
            cone_center = direction * (length / 2.0)
            jet_cone = pv.Cone(center=cone_center, direction=-direction, height=length, radius=radius, resolution=30)
            
            try:
                source_jet = event.jets[i] if event else None
                j_energy, j_eta, j_phi, j_dr = source_jet.energy, source_jet.eta, source_jet.phi, source_jet.delta_r
            except Exception:
                try:
                    j_energy = float(event["jets"]["energy"][i]) if event else 0.0
                    j_eta = float(event["jets"]["eta"][i]) if event else 0.0
                    j_phi = float(event["jets"]["phi"][i]) if event else 0.0
                    j_dr = float(event["jets"]["delta_r"][i]) if event else 0.4
                except Exception:
                    j_energy, j_eta, j_phi, j_dr = 0.0, 0.0, 0.0, 0.4
            
            mesh_id = f"jet_{i}"
            self.tooltip_dict[mesh_id] = ((
                f">> INSPECTING TARGET: RECONSTRUCTED JET #{i}\n----------------------------------------\n"
                f" Transverse E : {j_energy:.2f} GeV\n Pseudo-Rap   : {j_eta:.2f}\n"
                f" Azimuth phi  : {j_phi:.2f} rad\n Cone Radius  : {j_dr:.2f} (delta_R)"
            ), "orange")
            jet_cone.field_data["mesh_id"] = [mesh_id]
            
            initial_opacity = 0.0 if is_cinematic else 0.35
            act = plotter.add_mesh(jet_cone, color=self.theme["jet_cone"], opacity=initial_opacity, show_edges=True, edge_color=self.theme["jet_cone_edge"], name=mesh_id)
            
            if is_cinematic and ctx is not None:
                ctx["jet_actors"].append(act)
            #self._actor_registry["jet_cones"][i] = act
            self._actor_registry["jet_cones"][i] = {
                "actor": act,
                "pt": j_energy,       # ou jet_energy selon tes données
                "eta": j_eta,
                "phi": j_phi
            }

        # 4. RENDU DE L'ÉNERGIE MANQUANTE (MET)
        met_data = spatial_data.get("missing_energy", {"pt": 0.0, "phi": 0.0, "vector": (0.0, 0.0, 0.0)})
        
        # Initialisation sécurisée des variables locales pour l'animation
        if is_cinematic and ctx is not None:
            ctx["met_actor_line"] = None
            ctx["met_actor_tip"] = None
        
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
            
            act_line = plotter.add_mesh(met_mesh, color=self.theme["met"], line_width=5, opacity=1.0, name=f"{mesh_id}_line")
            act_tip = plotter.add_mesh(met_cone_tip, color=self.theme["met"], opacity=1.0, name=f"{mesh_id}_tip")
            
            if is_cinematic and ctx is not None:
                act_line.SetVisibility(False)
                act_tip.SetVisibility(False)
                ctx["met_actor_line"] = act_line
                ctx["met_actor_tip"] = act_tip
                
            self._actor_registry["met"] = {"line": act_line, "tip": act_tip}

        # 5. RAFFRAÎCHISSEMENT INTERFACE ET TEXTES HUD
        if not is_cinematic:
            plotter.add_text(
                f"IRIS3D // EVENT DETECTOR HUD ACTIVE\n-----------------------------------\nRun ID : {run_id} | Event ID : {event_id}\n\nSelect sub-atomic signature to decode...", 
                position="upper_left", font_size=11, font="courier", color=self.theme["hud_text"], name="metadata_banner"
            )
            plotter.add_legend(bcolor=None, face="circle")
            
        # À la fin de _plot_3d_detector, juste avant le réglage caméra :
        if not is_cinematic:
            self._add_filter_widgets(plotter)
            
        plotter.camera_position = [(5.0, 5.0, 4.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        plotter.camera.zoom(0.8)
    
    def animate_event(self, event: CollisionEvent, mode: str = "both", p_scale: float = 1.0, j_scale: float = 0.01, B_field: float = 3.8, speed: float = 0.05):
        """
        Orchestrates the high-performance cinematic animation display.
        Modes available: "both", "detector", "lego".
        Enhanced: Real-time track sub-stepping interpolation for fluid rendering.
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
        self._multi_select_ids = []
        self._actor_registry = {
            "particles": {},
            "jet_cones": {},
            "jet_towers": {},
            "met": {}
        }

        # Conteneurs d'animation internes
        ctx = {}

        # Sauvegarde des pointeurs pour le module d'exportation vidéo
        self._current_ctx = ctx
        self._current_mode = mode
        self._current_spatial_data = spatial_data
        self._active_plotter = plotter 

        # 2. Routage et initialisation des subplots
        if mode == "both":
            self._plot_3d_detector(plotter, spatial_data, event=event, ctx=ctx, is_cinematic=True, subplot_idx=(0, 0))
            self._plot_lego_calorimeter(plotter, spatial_data, event=event, ctx=ctx, is_cinematic=True, subplot_idx=(0, 1))
        elif mode == "detector":
            self._plot_3d_detector(plotter, spatial_data, event=event, ctx=ctx, is_cinematic=True, subplot_idx=(0, 0))
        elif mode == "lego":
            self._plot_lego_calorimeter(plotter, spatial_data, event=event, ctx=ctx, is_cinematic=True, subplot_idx=(0, 1))

        # Gestion de l'état du clavier
        self._setup_interactive_shortcuts(plotter)
        state = {"is_paused": False}
        plotter.add_key_event('space', lambda: state.update({"is_paused": not state["is_paused"]}))

        plotter.show(auto_close=False, interactive_update=True)

        max_r = self.detector_muon_r
        current_r = -3.0  

        # Pre-calcul des distances pour s'affranchir de la surcharge CPU dans la boucle standard
        track_distances = [np.linalg.norm(full_path, axis=1) for _, full_path in ctx["particle_polydata_lists"]]

        # ==========================================================
        # BOUCLE DE RENDU OPTIMISÉE (INTERPOLATION EN DIRECT)
        # ==========================================================
        while plotter.render_window is not None:
            if not hasattr(plotter, 'iren') or plotter.iren is None or plotter.render_window.GetInteractor().GetDone():
                break

            if not state["is_paused"]:
                current_r += speed
                if current_r > max_r + 0.5:
                    current_r = -3.0  

            # --- MISE À UPDATE : SUBPLOT GAUCHE (DÉTECTEUR) ---
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
                    for idx, act in self._actor_registry["jet_towers"].items(): act["actor"].SetVisibility(False)
                    
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
                    
                else:  # Phase de Collision (Avec Interpolation Continue Intégrée)
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

                    # INTERPOLATION VECTORIELLE DES TRACES EN DIRECT
                    for i, (poly, full_path) in enumerate(ctx["particle_polydata_lists"]):
                        actor = ctx["particle_actors"][i]
                        distances = track_distances[i]
                        
                        inside_mask = distances <= current_r
                        visible_points = list(full_path[inside_mask])
                        
                        outside_indices = np.where(~inside_mask)[0]
                        
                        if len(outside_indices) > 0 and len(visible_points) > 0:
                            next_idx = outside_indices[0]
                            prev_idx = next_idx - 1
                            
                            p_prev, p_next = full_path[prev_idx], full_path[next_idx]
                            d_prev, d_next = distances[prev_idx], distances[next_idx]
                            
                            if d_next != d_prev:
                                frac = (current_r - d_prev) / (d_next - d_prev)
                                exact_front_point = p_prev + frac * (p_next - p_prev)
                                visible_points.append(exact_front_point)
                        
                        if len(visible_points) > 1:
                            actor.SetVisibility(True)
                            # Calcul de spline à haute définition pour amortir les virages géométriques
                            new_spline = pv.Spline(np.array(visible_points), n_points=max(15, len(visible_points) * 2))
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

            # --- MISE À UPDATE : SUBPLOT DROIT (LEGO PLOT) ---
            if mode in ["both", "lego"]:
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
                            self._actor_registry["jet_towers"][i]["actor"].SetVisibility(True)
                            new_pts = init_pts.copy()
                            new_pts[init_pts[:, 2] > 0.001, 2] = target_height
                            box_mesh.points = new_pts
                        else:
                            self._actor_registry["jet_towers"][i]["actor"].SetVisibility(False)
                else:
                    for idx, act in self._actor_registry["jet_towers"].items(): act["actor"].SetVisibility(False)

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

        plotter.enable_mesh_picking(callback=picking_callback, show=False, left_clicking=True, show_message=False)
        plotter.show()
    
    def _plot_lego_calorimeter(self, plotter, spatial_data, event=None, ctx=None, is_cinematic: bool = False, subplot_idx=(0, 1)):
        """
        Unified method to draw the calorimeter Lego view, supporting both static 
        and high-performance dynamic morphing displays.
        """
        import numpy as np
        import pyvista as pv

        plotter.subplot(*subplot_idx)
        plotter.set_background(color=self.theme["bg_detector"])
        
        title_type = "LEGO PLOT" if is_cinematic else "STATIC LEGO PLOT"
        plotter.add_text(f"CALORIMETER METRIC ({title_type})", position=(0.05, 0.92), font_size=12, font="courier", color=self.theme["lego_title"], name="lego_title")
        
        # 1. RENDU DE L'ENVIRONNEMENT ET DES AXES (Commun)
        lego_floor = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=6.0, j_size=2 * np.pi)
        plotter.add_mesh(lego_floor, color=self.theme["lego_floor"], style="surface", show_edges=True, edge_color=self.theme["lego_floor_edge"], pickable=False)
        
        plotter.add_point_labels(np.array([[-3.0, -3.3, 0.01], [0.0, -3.3, 0.01], [3.0, -3.3, 0.01]]), ["eta = -3.0", "eta = 0.0", "eta = +3.0"], font_family="courier", font_size=12, show_points=False)
        plotter.add_mesh(pv.Line([-3.0, -3.3, 0.01], [3.0, -3.3, 0.01]), color=self.theme["lego_title"], line_width=2, pickable=False)
        plotter.add_point_labels(np.array([[-3.3, -np.pi, 0.01], [-3.3, 0.0, 0.01], [-3.3, np.pi, 0.01]]), ["phi = -pi", "phi = 0", "phi = +pi"], font_family="courier", font_size=12, show_points=False)
        plotter.add_mesh(pv.Line([-3.2, -np.pi, 0.01], [-3.2, np.pi, 0.01]), color=self.theme["lego_title"], line_width=2, pickable=False)

        # 2. COLLECTE DES ÉNERGIES ET NORMALISATION DE L'ÉCHELLE (Commun)
        jet_geometries = spatial_data.get("jet_geometries", [])
        jet_energies = []
        for i, jet_geo in enumerate(jet_geometries):
            try:
                # Priorité au format Event dict/object, sinon fallback géométrique
                j_energy = float(event["jets"]["energy"][i]) if event else jet_geo.get("pt", jet_geo.get("energy"))
            except Exception:
                try: j_energy = event.jets[i].energy if event else jet_geo["length"] * 10.0
                except Exception: j_energy = jet_geo["length"] * 10.0
            jet_energies.append(j_energy)
            
        max_allowed_height = 2.5 
        max_e = max(jet_energies) if len(jet_energies) > 0 else 1.0
        v_scale = max_allowed_height / max_e  

        # Initialisation des listes du contexte si mode cinématique actif
        if is_cinematic and ctx is not None:
            ctx["lego_mesh_references"] = []
            ctx["max_heights"] = []

        # 3. CONSTRUCTION DES TOURS LEGO
        for i, jet_geo in enumerate(jet_geometries):
            direction = np.array(jet_geo["unit_direction"])
            eta = jet_geo.get("eta", direction[2] * 1.5)
            phi = jet_geo.get("phi", np.arctan2(direction[1], direction[0]))
            
            pt_energy = jet_energies[i]
            final_height = pt_energy * v_scale  

            # Détermination de la géométrie de départ selon le mode
            current_height = 1.0 if is_cinematic else final_height
            lego_box = pv.Box(bounds=[eta - 0.18, eta + 0.18, phi - 0.18, phi + 0.18, 0.0, current_height])
            
            mesh_id = f"jet_{i}"
            lego_box.field_data["mesh_id"] = [mesh_id]
            
            # Injection sécurisée dans le dictionnaire de tooltips globaux
            if mesh_id not in self.tooltip_dict:
                self.tooltip_dict[mesh_id] = (f">> INSPECTING RECONSTRUCTED JET #{i}\n Transverse E : {pt_energy:.2f} GeV", "orange")

            # Ajout au plotter
            act = plotter.add_mesh(lego_box, color=self.theme["jet_tower"], opacity=1, show_edges=True, edge_color=self.theme["jet_tower_edge"], name=f"lego_tower_{i}")
            
            # Application de la logique spécifique au mode
            if is_cinematic and ctx is not None:
                act.SetVisibility(False)
                ctx["lego_mesh_references"].append((lego_box, lego_box.points.copy()))
                ctx["max_heights"].append(final_height)
            
            # Enregistrement dans le registre central (Crucial pour la synchro)
            #self._actor_registry["jet_towers"][i] = act

            self._actor_registry["jet_towers"][i] = {
                "actor": act,
                "pt": pt_energy,
                "eta": eta,
                "phi": phi
            }

        # 4. CADRAGE CAMÉRA ADJUSTÉ SELON LE MODE
        if is_cinematic:
            plotter.camera_position = [(0.0, -0.01, 9.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
            plotter.camera.zoom(0.72)
        else:
            plotter.camera_position = [(0.0, -5.0, 7.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
            plotter.camera.zoom(0.75)
    
    def _apply_cinematic_filters(self):
        """
        Scans all actor registries (particles, jet cones, lego towers) and toggles 
        visibility synchronously based on current cinematic slider states.
        """
        # 1. FILTRAGE DES PARTICULES
        for i, track_data in self._actor_registry["particles"].items():
            actor = track_data["actor"]
            pt_ok = track_data["pt"] >= self._current_filter_pt
            eta_ok = self._current_filter_eta_min <= track_data["eta"] <= self._current_filter_eta_max
            phi_ok = np.abs(track_data["phi"]) <= self._current_filter_phi_max
            
            actor.SetVisibility(True if (pt_ok and eta_ok and phi_ok) else False)

        # 2. FILTRAGE DES CÔNES DE JETS (VUE 3D)
        for i, jet_data in self._actor_registry["jet_cones"].items():
            actor = jet_data["actor"]
            pt_ok = jet_data["pt"] >= self._current_filter_pt
            eta_ok = self._current_filter_eta_min <= jet_data["eta"] <= self._current_filter_eta_max
            phi_ok = np.abs(jet_data["phi"]) <= self._current_filter_phi_max
            
            actor.SetVisibility(True if (pt_ok and eta_ok and phi_ok) else False)

        # 3. FILTRAGE DES TOURS LEGO (VUE 2D DÉROULÉE)
        for i, tower_data in self._actor_registry["jet_towers"].items():
            actor = tower_data["actor"]
            pt_ok = tower_data["pt"] >= self._current_filter_pt
            eta_ok = self._current_filter_eta_min <= tower_data["eta"] <= self._current_filter_eta_max
            phi_ok = np.abs(tower_data["phi"]) <= self._current_filter_phi_max
            
            actor.SetVisibility(True if (pt_ok and eta_ok and phi_ok) else False)
                
        # Rafraîchissement matériel des deux subplots en même temps
        if hasattr(self, "_active_plotter") and self._active_plotter:
            self._active_plotter.render()
    
    def _add_filter_widgets(self, plotter):
        """Adds interactive sliders with toggle capabilities using the 'f' key."""
        import numpy as np

        # Initialisation des états internes de filtrage
        self._current_filter_pt = 0.0
        self._current_filter_eta_min = -5.0
        self._current_filter_eta_max = 5.0
        self._current_filter_phi_max = np.pi  
        self._active_plotter = plotter  
        
        # Structure de contrôle pour le masquage
        self._filter_widgets = []
        self._filters_visible = True

        # Callbacks des sliders
        def callback_pt(value):
            self._current_filter_pt = value
            self._apply_cinematic_filters()

        def callback_eta_max(value):
            self._current_filter_eta_max = value
            self._apply_cinematic_filters()

        def callback_phi_max(value):
            self._current_filter_phi_max = value
            self._apply_cinematic_filters()

        # 1. Configuration et stockage du Slider pT
        w_pt = plotter.add_slider_widget(
            callback=callback_pt, rng=[0.0, 20.0], value=0.0, title="Min pT (GeV)",
            pointa=(0.02, 0.06), pointb=(0.25, 0.06), color=self.theme["hud_text"],
            title_height=0.015, style="modern", interaction_event="always"
        )
        self._filter_widgets.append(w_pt)

        # 2. Configuration et stockage du Slider Eta
        w_eta = plotter.add_slider_widget(
            callback=callback_eta_max, rng=[0.0, 5.0], value=5.0, title="Max eta",
            pointa=(0.02, 0.18), pointb=(0.25, 0.18), color=self.theme["hud_text"],
            title_height=0.015, style="modern", interaction_event="always"
        )
        self._filter_widgets.append(w_eta)

        # 3. Configuration et stockage du Slider Phi
        w_phi = plotter.add_slider_widget(
            callback=callback_phi_max, rng=[0.0, np.pi], value=np.pi, title="Max phi (rad)",
            pointa=(0.02, 0.30), pointb=(0.25, 0.30), color=self.theme["hud_text"],
            title_height=0.015, style="modern", interaction_event="always"
        )
        self._filter_widgets.append(w_phi)

        # 4. Liaison de la touche "f" à notre fonction de bascule
        plotter.add_key_event("f", self._toggle_filter_visibility)

    def _toggle_filter_visibility(self):
        """Toggles the on-screen visibility of all cinematic slider widgets cleanly."""
        if not hasattr(self, "_filter_widgets") or not self._filter_widgets:
            return
        
        if not hasattr(self, "_active_plotter") or not self._active_plotter:
            return

        # SÉCURITÉ VTK : On force le plotter à se re-concentrer sur le subplot 3D (0, 0)
        # pour éviter le warning "no current renderer" lors de la désactivation des widgets
        try:
            self._active_plotter.subplot(0, 0)
        except Exception:
            pass # Sécurité si l'architecture des subplots changeait à l'avenir

        # Inversion du drapeau d'état
        self._filters_visible = not self._filters_visible

        # Application du changement d'état
        for widget in self._filter_widgets:
            if self._filters_visible:
                widget.EnabledOn()  
            else:
                widget.EnabledOff() 

        # Rafraîchissement matériel immédiat
        self._active_plotter.render()
    
    def _setup_interactive_shortcuts(self, plotter):
        """Binds 'e' for screenshots and 'r' for video recording using the export module."""
        from . import export  # Import local pour éviter les boucles d'import
        
        # Touche [E] pour l'image
        plotter.add_key_event("e", lambda: export.export_screenshot(self))
        
        # Touche [R] (comme Record) pour la vidéo
        plotter.add_key_event("r", lambda: export.export_interactive_video(self, fps=30, duration_seconds=3.0))