import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from iris3d.models import CollisionEvent, Particle, Jet

class CoordinateTransformer:
    """
    Handles the geometric spatial transformations required to map 
    cylindrical high-energy physics parameters (pt, eta, phi) into 
    3D Cartesian space (X, Y, Z) for visual rendering, supporting
    helical magnetic field trajectories with dynamic physics-based boundary cuts.
    """

    @staticmethod
    def particle_to_vector(particle: Particle, scale: float = 1.0) -> Tuple[float, float, float]:
        """
        Calculates the 3D momentum trajectory vector of a single particle.
        """
        x = float(particle.pt * np.cos(particle.phi))
        y = float(particle.pt * np.sin(particle.phi))
        z = float(particle.pt * np.sinh(particle.eta))
        
        return x * scale, y * scale, z * scale

    @staticmethod
    def particle_to_helix(
        particle: Particle, 
        linear_endpoint: np.ndarray, 
        B_field: float = 3.8, 
        num_points: int = 50,  
        r_max: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculates a sequence of 3D points forming a helix for charged particles.
        Uses a two-pass geometry filtering to ensure millimeter-accurate detector 
        boundary cuts while maintaining perfectly proportioned dashed line gaps.
        """
        q = particle.charge
        pt = particle.pt
        phi_0 = particle.phi
        eta = particle.eta
        
        # PASSE 1 : Haute résolution mathématique pour une coupure nette au bord
        high_res = 200
        
        if q == 0 or B_field == 0 or pt <= 0:
            raw_points = np.linspace(np.array([0.0, 0.0, 0.0]), linear_endpoint, high_res)
        else:
            theta = 2 * np.arctan(np.exp(-eta))
            max_distance = np.linalg.norm(linear_endpoint)
            s_steps = np.linspace(0, max_distance, high_res)
            
            raw_points = np.zeros((high_res, 3))
            omega = (0.3 * B_field * q) / pt  
            
            raw_points[:, 0] = (pt / (0.3 * B_field * q)) * (np.sin(phi_0 + omega * s_steps) - np.sin(phi_0))
            raw_points[:, 1] = -(pt / (0.3 * B_field * q)) * (np.cos(phi_0 + omega * s_steps) - np.cos(phi_0))
            raw_points[:, 2] = s_steps * np.cos(theta)
        
        # Application de la coupure physique (R-cut) sur la haute résolution
        if r_max is not None:
            radii = np.sqrt(raw_points[:, 0]**2 + raw_points[:, 1]**2)
            outside_indices = np.where(radii > r_max)[0]
            if len(outside_indices) > 0:
                raw_points = raw_points[:max(2, outside_indices[0])]

        # PASSE 2 : Sous-échantillonnage adaptatif pour la taille des pointillés
        # Maintenant que la ligne est coupée pile au bon endroit, on mesure sa vraie longueur
        actual_length = np.linalg.norm(raw_points[-1] - raw_points[0]) if len(raw_points) > 0 else 0.0
        
        # On veut environ 3 à 4 segments visibles par unité géométrique
        desired_points = max(4, int(actual_length * 4))
        
        # On échantillonne uniformément la ligne haute résolution pour obtenir le nombre de points cible
        indices = np.linspace(0, len(raw_points) - 1, desired_points, dtype=np.int32)
        return raw_points[indices]

    @staticmethod
    def jet_to_cone(jet: Jet, scale: float = 0.01) -> Dict[str, Any]:
        """
        Calculates the physical orientation coordinates and geometry properties 
        required to render a Jet as a spatial 3D cone.
        """
        cos_phi = np.cos(jet.phi)
        sin_phi = np.sin(jet.phi)
        sinh_eta = np.sinh(jet.eta)
        cosh_eta = np.cosh(jet.eta) 
        
        dir_x = float(cos_phi / cosh_eta)
        dir_y = float(sin_phi / cosh_eta)
        dir_z = float(sinh_eta / cosh_eta)
        
        visual_length = float(jet.energy * scale)
        
        end_x = dir_x * visual_length
        end_y = dir_y * visual_length
        end_z = dir_z * visual_length
        
        cone_radius = float(visual_length * np.tan(jet.delta_r))
        
        return {
            "unit_direction": (dir_x, dir_y, dir_z),
            "endpoint": (end_x, end_y, end_z),
            "length": visual_length,
            "radius": cone_radius
        }

    def extract_event_arrays(
        self, 
        event: Any, 
        p_scale: float = 1.0, 
        j_scale: float = 0.01, 
        B_field: float = 3.8,
        detector_ecal_r: float = 2.0,   # Transmis par vis.py pour éviter le hardcoding
        detector_hcal_r: float = 2.8,   # Transmis par vis.py pour éviter le hardcoding
        detector_muon_r: float = 4.0    # Transmis par vis.py pour éviter le hardcoding
    ) -> Dict[str, Any]:
        """
        Extracts and translates a CollisionEvent or an Awkward Record into flat, 
        structured coordinate matrices. Maps particle trajectories to realistic
        detector boundary stops according to their sub-atomic identity (PID).
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

        pts, etas, phis, charges, pids, names = [], [], [], [], [], []

        if isinstance(raw_particles, (list, tuple)) or hasattr(raw_particles, "__iter__") and not hasattr(raw_particles, "fields") and not hasattr(raw_particles, "pt"):
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
                has_name = hasattr(raw_particles, "fields") and "name" in raw_particles.fields or "name" in raw_particles
                names = np.array(raw_particles["name"]) if has_name else None

        pts = np.array(pts, dtype=np.float64)
        etas = np.array(etas, dtype=np.float64)
        phis = np.array(phis, dtype=np.float64)
        charges = np.array(charges, dtype=np.int32)
        pids = np.array(pids, dtype=np.int32)

        p_count = len(pts)

        # --- CALCUL DE L'ÉNERGIE TRANSVERSE MANQUANTE (MET) CORRIGÉ ---
        met_data = {"pt": 0.0, "phi": 0.0, "vector": (0.0, 0.0, 0.0)}

        if p_count > 0:
            p_xyz = np.zeros((p_count, 3))
            p_xyz[:, 0] = pts * np.cos(phis) * p_scale
            p_xyz[:, 1] = pts * np.sin(phis) * p_scale
            p_xyz[:, 2] = pts * np.sinh(etas) * p_scale
            
            sum_px = np.sum(pts * np.cos(phis))
            sum_py = np.sum(pts * np.sin(phis))
            
            met_px = -sum_px
            met_py = -sum_py
            met_pt = np.sqrt(met_px**2 + met_py**2)
            met_phi = np.arctan2(met_py, met_px)
            
            if met_pt > 0.1:
                # 1. Calcul du vecteur MET brut à l'échelle visuelle
                raw_met_x = float(met_px * p_scale)
                raw_met_y = float(met_py * p_scale)
                raw_met_length = np.sqrt(raw_met_x**2 + raw_met_y**2)
                
                # 2. PLAFOND DYNAMIQUE SANS HARDCODING
                # On s'assure que le vecteur s'arrête au maximum au bord externe (detector_muon_r)
                max_allowed_length = detector_muon_r
                
                if raw_met_length > max_allowed_length and raw_met_length > 0:
                    scale_factor = max_allowed_length / raw_met_length
                    final_met_x = raw_met_x * scale_factor
                    final_met_y = raw_met_y * scale_factor
                else:
                    final_met_x = raw_met_x
                    final_met_y = raw_met_y
                
                met_data = {
                    "pt": float(met_pt),
                    "phi": float(met_phi),
                    # Le vecteur 3D final est désormais bridé géométriquement
                    "vector": (final_met_x, final_met_y, 0.0)
                }
            

            
            for i in range(p_count):
                p_name = str(names[i]) if names is not None and names[i] else f"Track {i}"
                pid_abs = abs(int(pids[i]))
                
                p_meta.append({
                    "pid": int(pids[i]),
                    "charge": int(charges[i]),
                    "name": p_name
                })
                
                # --- FLEXIBILITÉ SANS HARDCODING : Routage physique des rayons max ---
                if pid_abs in (11, 22):    # Électrons & Photons -> ECAL
                    assigned_r_max = detector_ecal_r
                elif pid_abs == 13:        # Muons -> S'échappent jusqu'aux chambres externes
                    assigned_r_max = detector_muon_r
                else:                      # Tous les autres hadrons -> HCAL
                    assigned_r_max = detector_hcal_r
                
                class _TrackParams:
                    def __init__(self, pt, eta, phi, charge):
                        self.pt = pt
                        self.eta = eta
                        self.phi = phi
                        self.charge = charge
                
                p_obj = _TrackParams(pts[i], etas[i], phis[i], charges[i])
                
                # Transmission sécurisée du rayon maximum à la méthode de calcul
                path = self.particle_to_helix(p_obj, p_xyz[i], B_field=B_field, r_max=assigned_r_max)
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
            "jet_geometries": j_vectors,
            "missing_energy": met_data 
        }