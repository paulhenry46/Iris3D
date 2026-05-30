import numpy as np
import dataclasses
from typing import Dict, Any, List, Optional
from iris3d.models import CollisionEvent, EventMetadata, Particle, Jet

def _to_iterable(array_or_list: Any) -> np.ndarray:
    """
    Safely converts an array-like structure (NumPy array, Awkward array, or list)
    into a standardized, indexable NumPy ndarray.
    
    CERN data structures parsed via Awkward Array are often memory-mapped views 
    of complex binary structures. Standard casting can corrupt strings or leak 
    un-materialized memory pointers.
    """
    if array_or_list is None:
        return np.array([])
        
    # 1. Handle Awkward Arrays explicitly without adding a hard dependency
    if type(array_or_list).__module__.startswith("awkward"):
        import awkward as ak
        
        # Specialized check: Awkward stores textual values (like particle names)
        # as character offsets. Direct conversion to NumPy breaks; we must
        # force a clean unrolling into a Python list first.
        if ak.types.is_string_type(ak.type(array_or_list)):
            return np.array(ak.to_list(array_or_list), dtype=object)
            
        # Fallback to standard native binary serialization to flat NumPy
        return ak.to_numpy(array_or_list)
    
    # 2. Standard fallback to conventional NumPy conversions
    return np.asarray(array_or_list)

def _clean_string(value: Any) -> Optional[str]:
    """
    Converts bytes, numpy scalar strings, or object wrappers to a clean native string.
    
    Data stored in binary TTree formats (ROOT/CERN files) frequently reads string data 
    as literal byte streams (e.g., b'mu-'). This function unpacks and decodes them 
    safely into native Python text strings to prevent rendering abstract 'b' prefixes in the UI.
    """
    if value is None or (hasattr(value, "None") and value is type(None)):
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
        
    # Extract underlying scalar text out of high-performance NumPy container wrappers
    actual_val = value.item() if hasattr(value, "item") else value
    if isinstance(actual_val, (bytes, bytearray)):
        return actual_val.decode("utf-8")
    if actual_val is None or str(actual_val) == "None" or actual_val == "":
        return None
        
    return str(actual_val)

def _safe_get_field(obj: Any, key: str, default: Any = None) -> Any:
    """
    Extracts a key/field from an object safely, regardless of whether 
    it is a standard dict, a NumPy structured array, or an Awkward Record.
    
    High-energy physics frameworks employ different lookup syntax conventions:
      - Standard Dict: data.get("particles")
      - NumPy Structured Array: data["particles"]
      - PyROOT/CERN Wrapper: data.particles (Attributes)
    """
    if obj is None:
        return default
        
    # Method A: Try standard dictionary method mapping
    if hasattr(obj, "get"):
        return obj.get(key, default)
        
    # Method B: Try direct subscript lookup (Handles Dicts, NumPy structural records, and Awkward Records)
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
        
    # Method C: Fallback check for abstract fields metadata arrays found in complex C++ framework objects
    if hasattr(obj, "fields") and key in obj.fields:
        try:
            return obj[key]
        except (KeyError, IndexError, TypeError, AttributeError):
            return getattr(obj, key, default)
            
    return default

def load_event(data: Any) -> CollisionEvent:
    """
    Agnostically ingests a single collision event.
    Guarantees strict safety boundaries against structural shape discrepancies 
    across Dicts, NumPy blocks, and CERN Awkward Records.
    """
    # Cache valid dataclass field names to filter unexpected inputs in row mode
    PARTICLE_FIELDS = {f.name for f in dataclasses.fields(Particle)}
    JET_FIELDS = {f.name for f in dataclasses.fields(Jet)}

    # ---------------------------------------------------------
    # 1. METADATA PROCESSING
    # ---------------------------------------------------------
    meta_raw = _safe_get_field(data, "metadata", {})
    
    run_id = int(_safe_get_field(meta_raw, "run_id", 0))
    event_id = int(_safe_get_field(meta_raw, "event_id", 0))
    sqrts_gev = float(_safe_get_field(meta_raw, "sqrts_gev", 13600.0)) # Default: LHC Run 3 Energy
    
    metadata = EventMetadata(run_id=run_id, event_id=event_id, sqrts_gev=sqrts_gev)
    
    # ---------------------------------------------------------
    # 2. PARTICLE TRACK INGESTION
    # ---------------------------------------------------------
    particles_raw = _safe_get_field(data, "particles", [])
    particles: List[Particle] = []
    
    # Format Route A: Row-oriented layout (List of tracking dictionaries)
    if isinstance(particles_raw, list):
        # Filter out unexpected keyword arguments to prevent initialization exceptions
        for p in particles_raw:
            if isinstance(p, dict):
                filtered_p = {k: v for k, v in p.items() if k in PARTICLE_FIELDS}
                particles.append(Particle(**filtered_p))
        
    # Format Route B: Columnar data arrays (Typical layout for experimental data processing)
    elif particles_raw is not None and (hasattr(particles_raw, "keys") or hasattr(particles_raw, "fields") or isinstance(particles_raw, dict)):
        pts = _to_iterable(_safe_get_field(particles_raw, "pt"))
        etas = _to_iterable(_safe_get_field(particles_raw, "eta"))
        phis = _to_iterable(_safe_get_field(particles_raw, "phi"))
        charges = _to_iterable(_safe_get_field(particles_raw, "charge"))
        pids = _to_iterable(_safe_get_field(particles_raw, "pid"))
        
        # Preliminary check: If essential geometry tracks are empty, skip processing
        if len(pts) > 0:
            names_raw = _safe_get_field(particles_raw, "name")
            if names_raw is not None:
                names = _to_iterable(names_raw)
                # If an explicit names collection was provided, include its length inside boundaries
                num_particles = min(len(pts), len(etas), len(phis), len(charges), len(pids), len(names))
            else:
                num_particles = min(len(pts), len(etas), len(phis), len(charges), len(pids))
                names = [None] * num_particles
            
            # Unroll Columnar values into structured data model entities safely
            for i in range(num_particles):
                particles.append(Particle(
                    pt=float(pts[i].item() if hasattr(pts[i], "item") else pts[i]),
                    eta=float(etas[i].item() if hasattr(etas[i], "item") else etas[i]),
                    phi=float(phis[i].item() if hasattr(phis[i], "item") else phis[i]),
                    charge=int(charges[i].item() if hasattr(charges[i], "item") else charges[i]),
                    pid=int(pids[i].item() if hasattr(pids[i], "item") else pids[i]),
                    name=_clean_string(names[i])
                ))

    # ---------------------------------------------------------
    # 3. JET INGESTION
    # ---------------------------------------------------------
    jets_raw = _safe_get_field(data, "jets", [])
    jets: List[Jet] = []
    
    # Format Route A: Row-oriented tracking
    if isinstance(jets_raw, list):
        for j in jets_raw:
            if isinstance(j, dict):
                filtered_j = {k: v for k, v in j.items() if k in JET_FIELDS}
                jets.append(Jet(**filtered_j))
        
    # Format Route B: Columnar jet clustering blocks
    elif jets_raw is not None and (hasattr(jets_raw, "keys") or hasattr(jets_raw, "fields") or isinstance(jets_raw, dict)):
        energies = _to_iterable(_safe_get_field(jets_raw, "energy"))
        etas = _to_iterable(_safe_get_field(jets_raw, "eta"))
        phis = _to_iterable(_safe_get_field(jets_raw, "phi"))
        
        if len(energies) > 0:
            delta_rs_raw = _safe_get_field(jets_raw, "delta_r")
            if delta_rs_raw is not None:
                delta_rs = _to_iterable(delta_rs_raw)
                num_jets = min(len(energies), len(etas), len(phis), len(delta_rs))
            else:
                num_jets = min(len(energies), len(etas), len(phis))
                delta_rs = [0.4] * num_jets # Standard ATLAS / CMS isolation radius configuration
            
            for i in range(num_jets):
                jets.append(Jet(
                    energy=float(energies[i].item() if hasattr(energies[i], "item") else energies[i]),
                    eta=float(etas[i].item() if hasattr(etas[i], "item") else etas[i]),
                    phi=float(phis[i].item() if hasattr(phis[i], "item") else phis[i]),
                    delta_r=float(delta_rs[i].item() if hasattr(delta_rs[i], "item") else delta_rs[i])
                ))
            
    return CollisionEvent(metadata=metadata, particles=particles, jets=jets)