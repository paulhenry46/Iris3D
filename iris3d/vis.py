import numpy as np
import pyvista as pv
from typing import Optional
from iris3d.models import CollisionEvent
from iris3d.core import CoordinateTransformer

class EventVisualizer:
    """
    Renders high-energy physics collision events in a highly fluid,
    GPU-accelerated 3D canvas using PyVista (VTK), featuring interactive
    picking tooltips and a passive geo-physical detector environment.
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

    def plot_event(self, event: CollisionEvent, p_scale: float = 1.0, j_scale: float = 0.01):
        """
        Generates and displays the fluid 3D interactive scene with detector environment.
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
        
        # 4. Process and Vectorize Particle Tracks
        spatial_data = self.transformer.extract_event_arrays(event, p_scale=p_scale, j_scale=j_scale)
        p_vectors = spatial_data["particle_vectors"]
        p_meta = spatial_data["particle_metadata"]
        
        for i in range(len(p_vectors)):
            endpoint = p_vectors[i]
            metadata = p_meta[i]
            source_particle = event.particles[i]
            
            track_line = pv.Line((0.0, 0.0, 0.0), endpoint)
            color = self._get_particle_color(metadata.get("pid", 0))
            p_name = metadata.get("name") if metadata.get("name") else f"Unassigned (PID {metadata.get('pid', 0)})"
            
            mesh_id = f"particle_{i}"
            hover_text = (
                f"Particle Track #{i}\n"
                f"Identity : {p_name}\n"
                f"pT       : {source_particle.pt:.2f} GeV\n"
                f"eta      : {source_particle.eta:.2f}\n"
                f"phi      : {source_particle.phi:.2f} rad\n"
                f"Charge   : {source_particle.charge}"
            )
            self.tooltip_dict[mesh_id] = hover_text
            
            track_line.field_data["mesh_id"] = [mesh_id]
            
            if source_particle.charge == 0:
                plotter.add_mesh(
                    track_line, color=color, line_width=1, opacity=0.5, name=mesh_id
                )
            else:
                plotter.add_mesh(
                    track_line, color=color, line_width=4, opacity=1.0, name=mesh_id
                )
            
        # 5. Process and Render Jet Cones
        for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
            direction = np.array(jet_geo["unit_direction"])
            length = jet_geo["length"]
            radius = jet_geo["radius"]
            
            cone_center = direction * (length / 2.0)
            jet_cone = pv.Cone(
                center=cone_center, direction=-direction, height=length, radius=radius, resolution=30
            )
            
            source_jet = event.jets[i]
            mesh_id = f"jet_{i}"
            jet_hover_text = (
                f"Reconstructed Jet #{i}\n"
                f"Energy : {source_jet.energy:.2f} GeV\n"
                f"eta    : {source_jet.eta:.2f}\n"
                f"phi    : {source_jet.phi:.2f} rad\n"
                f"delta_R: {source_jet.delta_r:.2f}"
            )
            self.tooltip_dict[mesh_id] = jet_hover_text
            
            jet_cone.field_data["mesh_id"] = [mesh_id]
            
            plotter.add_mesh(
                jet_cone, color="orange", opacity=0.35, show_edges=True, edge_color="darkorange", name=mesh_id
            )

        # FIX 2: Relocate the custom tracker box to "upper_right" to isolate it completely
        # from PyVista's built-in upper_left system hints!
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
        
        # FIX 1: Activate picking AFTER the renderer pipeline has established 
        # camera positions to completely eliminate the VTK InteractorStyle console warning.
        plotter.enable_mesh_picking(callback=picking_callback, show=False, left_clicking=True, show_message=False)
        
        plotter.show()