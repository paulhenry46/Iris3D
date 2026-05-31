import numpy as np
from typing import Tuple, List, Dict, Any
from iris3d.models import CollisionEvent, Particle, Jet

class CoordinateTransformer:
    """
    Handles the geometric spatial transformations required to map 
    cylindrical high-energy physics parameters (pt, eta, phi) into 
    3D Cartesian space (X, Y, Z) for visual rendering, supporting
    helical magnetic field trajectories.
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
    def particle_to_helix(particle: Particle, linear_endpoint: np.ndarray, B_field: float = 3.8, num_points: int = 50) -> np.ndarray:
        """
        Calculates a sequence of 3D points forming a helix for charged particles
        under an axial magnetic field B_z (aligned with Z axis). Neutral particles
        return a perfectly straight line sequence.
        
        Physics formula: R = pt / (0.3 * B)
        """
        q = particle.charge
        pt = particle.pt
        
        # if neutral, no fieald or pt, straight curve.
        if q == 0 or B_field == 0 or pt <= 0:
            return np.linspace(np.array([0.0, 0.0, 0.0]), linear_endpoint, num_points)
            
        phi_0 = particle.phi
        eta = particle.eta
        theta = 2 * np.arctan(np.exp(-eta))
        
       
        max_distance = np.linalg.norm(linear_endpoint)
        s_steps = np.linspace(0, max_distance, num_points)
        
        points = np.zeros((num_points, 3))
        
      
        omega = (0.3 * B_field * q) / pt  
        
        
        points[:, 0] = (pt / (0.3 * B_field * q)) * (np.sin(phi_0 + omega * s_steps) - np.sin(phi_0))
        points[:, 1] = -(pt / (0.3 * B_field * q)) * (np.cos(phi_0 + omega * s_steps) - np.cos(phi_0))
        points[:, 2] = s_steps * np.cos(theta)
        
        return points

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

    def extract_event_arrays(self, event: Any, p_scale: float = 1.0, j_scale: float = 0.01, B_field: float = 3.8) -> Dict[str, Any]:
        """
        Extracts and translates a CollisionEvent or an Awkward Record into flat, 
        structured coordinate matrices. Dynamically adapts to object attributes,
        columnar records, or lists of individual particles.
        """
        p_meta = []
        p_paths = []
        
        # --- 1. EXTRACT PARTICLES DATA ---
        if hasattr(event, "particles") and not isinstance(event, dict):
            raw_particles = event.particles
        else:
            try:
                raw_particles = event["particles"]
            except (KeyError, TypeError):
                raw_particles = []

        # Normalisation des données de particules (Columnar vs List-of-Dicts/Objects)
        pts, etas, phis, charges, pids, names = [], [], [], [], [], []

        if isinstance(raw_particles, (list, tuple)) or hasattr(raw_particles, "__iter__") and not hasattr(raw_particles, "fields") and not hasattr(raw_particles, "pt"):
            # CAS A : C'est une liste d'objets individuels ou de dictionnaires (Premier Event)
            for p in raw_particles:
                if isinstance(p, dict) or hasattr(p, "__getitem__"):
                    pts.append(p["pt"])
                    etas.append(p["eta"])
                    phis.append(p["phi"])
                    charges.append(p["charge"])
                    pids.append(p["pid"])
                    names.append(p.get("name", None) if isinstance(p, dict) else p["name"])
                else:
                    pts.append(p.pt)
                    etas.append(p.eta)
                    phis.append(p.phi)
                    charges.append(p.charge)
                    pids.append(p.pid)
                    names.append(p.name if hasattr(p, "name") else None)
        else:
            # CAS B : C'est une structure colonaire (Awkward Record, Dict de listes - Deuxième Event)
            if hasattr(raw_particles, "pt"):
                pts = np.array(raw_particles.pt)
                etas = np.array(raw_particles.eta)
                phis = np.array(raw_particles.phi)
                charges = np.array(raw_particles.charge)
                pids = np.array(raw_particles.pid)
                names = np.array(raw_particles.name) if hasattr(raw_particles, "name") else None
            else:
                pts = np.array(raw_particles["pt"])
                etas = np.array(raw_particles["eta"])
                phis = np.array(raw_particles["phi"])
                charges = np.array(raw_particles["charge"])
                pids = np.array(raw_particles["pid"])
                # Gestion sécurisée du champ optionnel 'name' dans Awkward
                has_name = hasattr(raw_particles, "fields") and "name" in raw_particles.fields or "name" in raw_particles
                names = np.array(raw_particles["name"]) if has_name else None

        # Conversion finale en tableaux NumPy pour le calcul vectorisé
        pts = np.array(pts, dtype=np.float64)
        etas = np.array(etas, dtype=np.float64)
        phis = np.array(phis, dtype=np.float64)
        charges = np.array(charges, dtype=np.int32)
        pids = np.array(pids, dtype=np.int32)

        p_count = len(pts)

        if p_count > 0:
            p_xyz = np.zeros((p_count, 3))
            p_xyz[:, 0] = pts * np.cos(phis) * p_scale
            p_xyz[:, 1] = pts * np.sin(phis) * p_scale
            p_xyz[:, 2] = pts * np.sinh(etas) * p_scale
            
            for i in range(p_count):
                p_name = str(names[i]) if names is not None and names[i] else f"Track {i}"
                
                p_meta.append({
                    "pid": int(pids[i]),
                    "charge": int(charges[i]),
                    "name": p_name
                })
                
                class _TrackParams:
                    def __init__(self, pt, eta, phi, charge):
                        self.pt = pt
                        self.eta = eta
                        self.phi = phi
                        self.charge = charge
                
                p_obj = _TrackParams(pts[i], etas[i], phis[i], charges[i])
                path = self.particle_to_helix(p_obj, p_xyz[i], B_field=B_field)
                p_paths.append(path)
        else:
            p_xyz = np.zeros((0, 3))

        # --- 2. EXTRACT JETS DATA ---
        if hasattr(event, "jets") and not isinstance(event, dict):
            raw_jets = event.jets
        else:
            try:
                raw_jets = event["jets"]
            except (KeyError, TypeError):
                raw_jets = []

        j_energies, j_etas, j_phis, j_drs = [], [], [], []

        if isinstance(raw_jets, (list, tuple)) or hasattr(raw_jets, "__iter__") and not hasattr(raw_jets, "fields") and not hasattr(raw_jets, "energy"):
            for j in raw_jets:
                if isinstance(j, dict) or hasattr(j, "__getitem__"):
                    j_energies.append(j["energy"])
                    j_etas.append(j["eta"])
                    j_phis.append(j["phi"])
                    j_drs.append(j["delta_r"])
                else:
                    j_energies.append(j.energy)
                    j_etas.append(j.eta)
                    j_phis.append(j.phi)
                    j_drs.append(j.delta_r)
        else:
            if hasattr(raw_jets, "energy"):
                j_energies = np.array(raw_jets.energy)
                j_etas = np.array(raw_jets.eta)
                j_phis = np.array(raw_jets.phi)
                j_drs = np.array(raw_jets.delta_r)
            else:
                j_energies = np.array(raw_jets["energy"]) if (hasattr(raw_jets, "fields") and "energy" in raw_jets.fields) or "energy" in raw_jets else np.array([])
                j_etas = np.array(raw_jets["eta"]) if (hasattr(raw_jets, "fields") and "eta" in raw_jets.fields) or "eta" in raw_jets else np.array([])
                j_phis = np.array(raw_jets["phi"]) if (hasattr(raw_jets, "fields") and "phi" in raw_jets.fields) or "phi" in raw_jets else np.array([])
                j_drs = np.array(raw_jets["delta_r"]) if (hasattr(raw_jets, "fields") and "delta_r" in raw_jets.fields) or "delta_r" in raw_jets else np.array([])

        j_count = len(j_energies)
        j_vectors = []
        
        if j_count > 0:
            for i in range(j_count):
                class _JetParams:
                    def __init__(self, energy, eta, phi, delta_r):
                        self.energy = energy
                        self.eta = eta
                        self.phi = phi
                        self.delta_r = delta_r
                        
                j_obj = _JetParams(j_energies[i], j_etas[i], j_phis[i], j_drs[i])
                j_vectors.append(self.jet_to_cone(j_obj, scale=j_scale))
            
        return {
            "particle_vectors": p_xyz,
            "particle_metadata": p_meta,
            "particle_paths": p_paths,
            "jet_geometries": j_vectors
        }