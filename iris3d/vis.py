import numpy as np
import pyvista as pv
from typing import Optional
from iris3d.models import CollisionEvent
from iris3d.core import CoordinateTransformer

class EventVisualizer:
    """
    Renders high-energy physics collision events in a highly fluid,
    GPU-accelerated 3D canvas using PyVista (VTK).
    """
    def __init__(self, theme: str = "dark"):
        self.transformer = CoordinateTransformer()
        
        # Set up a clean, professional dark mode background for event displays
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
        return "white" # Fallback/Unknown tracks

    def plot_event(self, event: CollisionEvent, p_scale: float = 1.0, j_scale: float = 0.05):
        """
        Generates and displays the fluid 3D interactive scene.
        """
        # 1. Initialize the high-performance PyVista Plotter
        plotter = pv.Plotter(window_size=[1024, 768], title=f"Iris3D - Run {event.metadata.run_id} Event {event.metadata.event_id}")
        plotter.add_axes()
        plotter.show_grid(color="gray")
        
        # 2. Add the Interaction Vertex Marker
        vertex = pv.Sphere(radius=0.05, center=(0.0, 0.0, 0.0))
        plotter.add_mesh(vertex, color="magenta", render_points_as_spheres=True, label="Interaction Vertex")
        
        # 3. Process and Vectorize Particle Tracks
        spatial_data = self.transformer.extract_event_arrays(event, p_scale=p_scale, j_scale=j_scale)
        p_vectors = spatial_data["particle_vectors"]
        p_meta = spatial_data["particle_metadata"]
        
        for i in range(len(p_vectors)):
            endpoint = p_vectors[i]
            metadata = p_meta[i]
            
            # Define a flat line segment path from origin (0,0,0) to the particle endpoint
            track_line = pv.Line((0.0, 0.0, 0.0), endpoint)
            color = self._get_particle_color(metadata["pid"])
            
            # Neutral track distinction handling:
            # Neutral tracks are rendered with distinct width and high transparency 
            # to prevent blocking primary charged particle tracks.
            if metadata["charge"] == 0:
                plotter.add_mesh(
                    track_line,
                    color=color,
                    line_width=1,
                    opacity=0.5,
                    name=f"particle_{i}"
                )
            else:
                plotter.add_mesh(
                    track_line, 
                    color=color, 
                    line_width=4, 
                    opacity=1.0,
                    name=f"particle_{i}"
                )
            
        # 4. Process and Render Jet Cones
        for i, jet_geo in enumerate(spatial_data["jet_geometries"]):
            direction = np.array(jet_geo["unit_direction"])
            length = jet_geo["length"]
            radius = jet_geo["radius"]
            
            # By pointing the PyVista cone in the opposite direction (-direction),
            # the apex (summit) flips to face the origin (0,0,0), and the cone
            # expands outward along the true particle track trajectory
            cone_center = direction * (length / 2.0)
            
            jet_cone = pv.Cone(
                center=cone_center,
                direction=-direction,  # Inverted direction of vector
                height=length,
                radius=radius,
                resolution=30
            )
            
            # Render jets with a semi-transparent, luminous orange overlay
            plotter.add_mesh(
                jet_cone, 
                color="orange", 
                opacity=0.35, 
                show_edges=True, 
                edge_color="darkorange",
                name=f"jet_{i}"
            )
            
        # 5. Open the fluid display engine window
        plotter.add_legend(bcolor=None, face="circle")
        plotter.show()