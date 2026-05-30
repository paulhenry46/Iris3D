import numpy as np
import awkward as ak
from iris3d.io import load_event

# =====================================================================
# FORMAT 1: Row-Oriented Layout (Standard Python Lists & Dicts)
# Ideal for simple configurations, JSON streaming, or small test cases.
# =====================================================================
row_based_data = {
    "metadata": {
        "run_id": 402130,
        "event_id": 8943210,
        "sqrts_gev": 13600.0
    },
    "particles": [
        {"pt": 45.2, "eta": 1.2, "phi": -0.5, "charge": -1, "pid": 11, "name": "e-"},
        {"pt": 38.1, "eta": -0.8, "phi": 2.6, "charge": 1, "pid": -11, "name": "e+"},
        {"pt": 2.1, "eta": 2.4, "phi": 1.1, "charge": 0, "pid": 22, "name": "gamma", "ignored_extra_key": True} 
    ],
    "jets": [
        {"energy": 120.5, "eta": 0.4, "phi": -1.2, "delta_r": 0.4},
        {"energy": 85.0, "eta": -1.9, "phi": 1.8} # Falls back to default delta_r=0.4
    ]
}

event_from_rows = load_event(row_based_data)


# =====================================================================
# FORMAT 2: Columnar Data Layout (Flat NumPy / Dictionary Arrays)
# Ideal for tabular data pipelines and flat memory blocks.
# =====================================================================
columnar_data = {
    "metadata": {
        "run_id": 402130,
        "event_id": 8943211
    },
    "particles": {
        "pt": np.array([55.4, 12.3], dtype=np.float32),
        "eta": np.array([0.15, -2.1], dtype=np.float32),
        "phi": np.array([-2.8, 0.9], dtype=np.float32),
        "charge": np.array([1, -1], dtype=np.int32),
        "pid": np.array([13, -13], dtype=np.int32),
        "name": np.array([b"mu+", b"mu-"]) # Testing byte string decoding conversion
    },
    "jets": {
        "energy": np.array([210.0]),
        "eta": np.array([0.05]),
        "phi": np.array([-1.4]),
        "delta_r": np.array([0.4])
    }
}

event_from_columns = load_event(columnar_data)


# =====================================================================
# FORMAT 3: CERN Awkward Records Layout
# Ideal for real high-energy physics analysis workflows.
# =====================================================================
# Creating an Awkward Record mimicking an extracted single-event slice
awkward_data = ak.Record({
    "metadata": {
        "run_id": 402130,
        "event_id": 8943212,
        "sqrts_gev": 13600.0
    },
    "particles": {
        "pt": ak.Array([85.0, 62.1, 4.5]),
        "eta": ak.Array([-0.4, 0.9, 1.1]),
        "phi": ak.Array([1.7, -2.1, 0.3]),
        "charge": ak.Array([-1, 1, 0]),
        "pid": ak.Array([11, 13, 22]),
        "name": ak.Array(["e-", "mu+", "gamma"]) # Text categories mapping
    },
    "jets": {
        "energy": ak.Array([340.5]),
        "eta": ak.Array([0.85]),
        "phi": ak.Array([-1.9]),
        "delta_r": ak.Array([0.4])
    }
})

event_from_awkward = load_event(awkward_data)


# =====================================================================
# VERIFICATION PIPELINE
# Ensure all variations parsed seamlessly into identical strict dataclasses
# =====================================================================
print("--- Row-based parsing summary ---")
print(f"Particles count: {len(event_from_rows.particles)}")
print(f"First particle object type: {type(event_from_rows.particles[0].pt)}")

print("\n--- Columnar-based parsing summary ---")
print(f"Decoded string property check: {event_from_columns.particles[0].name} (Type: {type(event_from_columns.particles[0].name)})")

print("\n--- Awkward-based parsing summary ---")
print(f"Event ID: {event_from_awkward.metadata.event_id}")
print(f"Jet Energy: {event_from_awkward.jets[0].energy} GeV")