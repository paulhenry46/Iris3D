import sys
import os
import awkward as ak

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iris3d.io import load_event
from iris3d.vis import EventVisualizer

# 1. Fabricate a complex simulated collision event slice using Awkward Records
mock_hep_event = ak.Record({
    "metadata": {"run_id": 402130, "event_id": 99824, "sqrts_gev": 13600.0},
    "particles": {
        "pt": ak.Array([45.0, 42.1, 15.0, 2.5]),
        "eta": ak.Array([0.2, -0.4, 1.1, -2.1]),
        "phi": ak.Array([1.5, -1.6, 0.2, 2.8]),
        "charge": ak.Array([-1, 1, 0, 1]),
        "pid": ak.Array([11, 13, 22, 211]), # Electron, Muon, Photon, Pion
        "name": ak.Array(["e-", "mu+", "gamma", "pi+"])
    },
    "jets": {
        "energy": ak.Array([250.0, 180.0]),
        "eta": ak.Array([0.2, -0.4]),
        "phi": ak.Array([1.5, -1.6]),
        "delta_r": ak.Array([0.4, 0.4]) # Jets matching the hard electron/muon directions
    }
})

if __name__ == "__main__":
    print("Ingesting experimental mock event data block...")
    event = load_event(mock_hep_event)
    
    print("Initializing fluid PyVista 3D event viewer...")
    visualizer = EventVisualizer(theme="dark")
    
    # Fire up the graphics layer!
    visualizer.plot_event(event, p_scale=2.2, j_scale=0.012)