import numpy as np
from typing import Tuple, List, Dict, Any
from iris3d.models import CollisionEvent, Particle, Jet

class CoordinateTransformer:
    """
    Handles the geometric spatial transformations required to map 
    cylindrical high-energy physics parameters (pt, eta, phi) into 
    3D Cartesian space (X, Y, Z) for visual rendering.
    """

    @staticmethod
    def particle_to_vector(particle: Particle, scale: float = 1.0) -> Tuple[float, float, float]:
        """
        Calculates the 3D momentum trajectory vector of a single particle.
        
        Args:
            particle: The target Particle dataclass instance.
            scale: Multiplier to adjust visual track line length in the canvas.
            
        Returns:
            A tuple of native floats representing (X, Y, Z) directional endpoints.
        """
        x = float(particle.pt * np.cos(particle.phi))
        y = float(particle.pt * np.sin(particle.phi))
        z = float(particle.pt * np.sinh(particle.eta))
        
        return x * scale, y * scale, z * scale

    @staticmethod
    def jet_to_cone(jet: Jet, scale: float = 0.01) -> Dict[str, Any]:
        """
        Calculates the physical orientation coordinates and geometry properties 
        required to render a Jet as a spatial 3D cone.
        
        Args:
            jet: The target Jet dataclass instance.
            scale: Multiplier to scale the massive Energy values (GeV) down 
                   to a manageable visual scale alongside particle tracks.
                   
        Returns:
            A dictionary containing the directional unit vector, scaled endpoint vector,
            calculated visual length, and outer radius profile based on delta_r.
        """
        cos_phi = np.cos(jet.phi)
        sin_phi = np.sin(jet.phi)
        sinh_eta = np.sinh(jet.eta)
        cosh_eta = np.cosh(jet.eta) 
        
        # Calculate a perfect unit direction vector (Length = 1.0)
        dir_x = float(cos_phi / cosh_eta)
        dir_y = float(sin_phi / cosh_eta)
        dir_z = float(sinh_eta / cosh_eta)
        
        # Scale total spatial length of the jet proportional to its measured Energy
        visual_length = float(jet.energy * scale)
        
        # Generate the precise physical endpoint vector in 3D space
        end_x = dir_x * visual_length
        end_y = dir_y * visual_length
        end_z = dir_z * visual_length
        
        # Calculate the base radius of the jet cone based on its isolation radius delta_r
        cone_radius = float(visual_length * np.tan(jet.delta_r))
        
        return {
            "unit_direction": (dir_x, dir_y, dir_z),
            "endpoint": (end_x, end_y, end_z),
            "length": visual_length,
            "radius": cone_radius
        }

    def extract_event_arrays(self, event: CollisionEvent, p_scale: float = 1.0, j_scale: float = 0.01) -> Dict[str, Any]:
        """
        Extracts and translates an entire CollisionEvent into flat, structured coordinates.
        Optimizes calculation overhead using NumPy vectorization whenever applicable.
        
        Returns:
            A dictionary containing clean aligned coordinate matrices 
            for both particles and jets.
        """
        p_count = len(event.particles)
        p_meta = []
        
        if p_count > 0:
            # High-performance Vectorization: Extract array vectors all at once
            pts = np.array([p.pt for p in event.particles])
            etas = np.array([p.eta for p in event.particles])
            phis = np.array([p.phi for p in event.particles])
            
            # Compute full X, Y, Z coordinate matrices with vectorized method.
            p_xyz = np.zeros((p_count, 3))
            p_xyz[:, 0] = pts * np.cos(phis) * p_scale
            p_xyz[:, 1] = pts * np.sin(phis) * p_scale
            p_xyz[:, 2] = pts * np.sinh(etas) * p_scale
            
            for i, p in enumerate(event.particles):
                p_meta.append({
                    "pid": p.pid,
                    "charge": p.charge,
                    "name": p.name or f"Track {i}"
                })
        else:
            p_xyz = np.zeros((0, 3))
            
        # Extract jet target attributes
        j_vectors = [self.jet_to_cone(j, scale=j_scale) for j in event.jets]
            
        return {
            "particle_vectors": p_xyz,
            "particle_metadata": p_meta,
            "jet_geometries": j_vectors
        }