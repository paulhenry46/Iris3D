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
        tracker_radius = 1.5
        tracker_length = 5.0
        
        tracker = pv.Cylinder(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=tracker_radius,
            height=tracker_length,
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

        calorimeter_outer_radius = 2.8
        calorimeter_length = 6.0
        
        calorimeter = pv.Cylinder(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=calorimeter_outer_radius,
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
        Generates and displays the fluid 3D interactive scene with helical track bending
        and Missing Transverse Energy (MET) balancing.
        """
        # 1. Initialize the high-performance PyVista Plotter
        plotter = pv.Plotter(window_size=[1024, 768], title=f"Iris3D - Run {event.metadata.run_id} Event {event.metadata.event_id}")
        plotter.add_axes()
        plotter.show_grid(color="gray")
        
        # 2. Render Passive Detector Reference Subsystems
        self._add_detector_geometry(plotter)
        
        # 3. Add the Interaction Vertex Marker
        vertex = pv.Sphere(radius=0.05, center=(0.0, 0.0, 0.0))
        plotter.add_mesh(vertex, color="magenta", render_points_as_spheres=True, label="Interaction Vertex")
        
        # Dictionary to store metadata strings associated with custom geometric IDs
        self.tooltip_dict = {}
        
        # 4. Process Particle Tracks (Helical vs Straight)
        spatial_data = self.transformer.extract_event_arrays(event, p_scale=p_scale, j_scale=j_scale, B_field=B_field)
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
                
                if p_pid == 22:  # Photons -> Pointillés physiques (Dashed)
                    lines_connectivity = []
                    for idx in range(0, len(points) - 1, 2):
                        lines_connectivity.extend([2, idx, idx + 1])
                    track_mesh.lines = np.array(lines_connectivity, dtype=np.int32)
                else:  # Hadrons Neutres -> Ligne droite continue
                    cells = np.hstack([[len(points)], np.arange(len(points))])
                    track_mesh.lines = cells
            
            color = self._get_particle_color(p_pid)
            mesh_id = f"particle_{i}"
            
            hover_text = (
                f"Particle Track #{i}\n"
                f"Identity : {p_name}\n"
                f"pT       : {pt_val:.2f} GeV\n"
                f"eta      : {eta_val:.2f}\n"
                f"phi      : {phi_val:.2f} rad\n"
                f"Charge   : {p_charge}"
            )
            self.tooltip_dict[mesh_id] = hover_text
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
            jet_hover_text = (
                f"Reconstructed Jet #{i}\n"
                f"Energy : {j_energy:.2f} GeV\n"
                f"eta    : {j_eta:.2f}\n"
                f"phi    : {j_phi:.2f} rad\n"
                f"delta_R: {j_dr:.2f}"
            )
            self.tooltip_dict[mesh_id] = jet_hover_text
            
            jet_cone.field_data["mesh_id"] = [mesh_id]
            
            plotter.add_mesh(
                jet_cone, color="orange", opacity=0.35, show_edges=True, edge_color="darkorange", name=mesh_id
            )

        # --- 5.5. Process and Render Missing Transverse Energy (MET) ---
        met_data = spatial_data.get("missing_energy", {"pt": 0.0, "phi": 0.0, "vector": (0.0, 0.0, 0.0)})
        
        if met_data["pt"] > 0.5:  # Seuil physique d'activation
            met_vector = np.array(met_data["vector"])
            
            # Échantillonnage discret pour générer un maillage pointillé propre
            met_points = np.linspace(np.array([0.0, 0.0, 0.0]), met_vector, 30)
            met_mesh = pv.PolyData(met_points)
            
            met_lines = []
            for idx in range(0, len(met_points) - 1, 2):
                met_lines.extend([2, idx, idx + 1])
            met_mesh.lines = np.array(met_lines, dtype=np.int32)
            
            # Cône directionnel pour la tête de la flèche
            met_direction = met_vector / np.linalg.norm(met_vector)
            met_cone_tip = pv.Cone(
                center=met_vector,
                direction=met_direction,
                height=3,
                radius=1,
                resolution=20
            )
            
            mesh_id = "missing_energy_vector"
            met_hover_text = (
                f"Missing Transverse Energy (MET)\n"
                f"Unseen Momentum (pT) : {met_data['pt']:.2f} GeV\n"
                f"phi direction        : {met_data['phi']:.2f} rad\n"
            )
            self.tooltip_dict[mesh_id] = met_hover_text
            
            met_mesh.field_data["mesh_id"] = [mesh_id]
            met_cone_tip.field_data["mesh_id"] = [mesh_id]
            
            # Affichage de l'indicateur rouge fluo standardisé
            plotter.add_mesh(met_mesh, color="red", line_width=5, opacity=1.0, name=f"{mesh_id}_line")
            plotter.add_mesh(met_cone_tip, color="red", opacity=1.0, name=f"{mesh_id}_tip")

        # Affichage de la bannière en "upper_left" pour éviter la superposition de texte
        plotter.add_text(
            "Click an object to inspect...", 
            position="upper_left", 
            font_size=11, 
            color="white",
            name="metadata_banner"
        )

        # Define picker callback function
        def picking_callback(mesh):
            if mesh and "mesh_id" in mesh.field_data:
                mesh_id = mesh.field_data["mesh_id"][0]
                if mesh_id in self.tooltip_dict:
                    plotter.add_text(
                        self.tooltip_dict[mesh_id], 
                        position="upper_left", 
                        font_size=11, 
                        color="white",
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
        
        # Initialisation propre du mesh picking après la configuration de la caméra
        plotter.enable_mesh_picking(callback=picking_callback, show=False, left_clicking=True, show_message=False)
        
        plotter.show()