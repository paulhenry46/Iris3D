import sys
import os
import awkward as ak

# Configuration des chemins pour importer le package local iris3d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iris3d.io import load_event
from iris3d.vis import EventVisualizer

# --- FABRICATION DES ÉVÉNEMENTS EXPANSIBLES ---
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
        "delta_r": ak.Array([0.4, 0.4])
    }
})

mock_event_extreme_pt = ak.Record({
    "metadata": {"run_id": 402130, "event_id": 99825, "sqrts_gev": 13600.0},
    "particles": {
        "pt": ak.Array([500.0, 0.4, 85.0, 12.0]),
        "eta": ak.Array([0.05, 0.1, -0.8, 1.4]),
        "phi": ak.Array([0.5, -0.5, 2.1, -1.2]),
        "charge": ak.Array([-1, 1, 0, 1]),
        "pid": ak.Array([13, 211, 130, -11]),  # Muon, Pion+, Kaon Long, Positon
        "name": ak.Array(["mu_hard-", "pi_soft+", "K_long", "e+"])
    },
    "jets": {
        "energy": ak.Array([650.0]),
        "eta": ak.Array([0.05]),
        "phi": ak.Array([0.5]),
        "delta_r": ak.Array([0.4])
    }
})

mock_event_higgs_diphoton = ak.Record({
    "metadata": {"run_id": 402130, "event_id": 99826, "sqrts_gev": 13600.0},
    "particles": {
        "pt": ak.Array([65.0, 60.0, 1.2, 0.9]),
        "eta": ak.Array([0.4, -0.4, 2.2, -1.8]),
        "phi": ak.Array([1.0, -2.14, 0.1, 3.0]),
        "charge": ak.Array([0, 0, 1, -1]),
        "pid": ak.Array([22, 22, 211, -211]),  # Photons du Higgs + Bruit
        "name": ak.Array(["gamma_1", "gamma_2", "ch_noise1", "ch_noise2"])
    },
    "jets": {
        "energy": ak.Array([]), "eta": ak.Array([]), "phi": ak.Array([]), "delta_r": ak.Array([])
    }
})

mock_event_empty_edge_case = ak.Record({
    "metadata": {"run_id": 402130, "event_id": 99827, "sqrts_gev": 13600.0},
    "particles": {
        "pt": ak.Array([]), "eta": ak.Array([]), "phi": ak.Array([]), "charge": ak.Array([]), "pid": ak.Array([]), "name": ak.Array([])
    },
    "jets": {
        "energy": ak.Array([]), "eta": ak.Array([]), "phi": ak.Array([]), "delta_r": ak.Array([])
    }
})

mock_event_w_boson = ak.Record({
    "metadata": {"run_id": 402130, "event_id": 99828, "sqrts_gev": 13600.0},
    "particles": {
        "pt": ak.Array([80.0, 1.5, 0.8]),
        "eta": ak.Array([0.1, -1.2, 0.5]),
        "phi": ak.Array([0.0, 0.8, -1.5]),
        "charge": ak.Array([-1, 1, -1]),
        "pid": ak.Array([11, 211, -211]), 
        "name": ak.Array(["e_signal-", "pion1", "pion2"])
    },
    "jets": {
        "energy": ak.Array([]), "eta": ak.Array([]), "phi": ak.Array([]), "delta_r": ak.Array([])
    }
})

if __name__ == "__main__":
    print("==================================================")
    print("       IRIS3D - ADVANCED PHYSICS DISPLAY          ")
    print("==================================================")
    
    # Instance globale du visualiseur avec les propriétés géométriques centralisées
    visualizer = EventVisualizer()
    
    # ------------------------------------------------------------
    # DEMO 1 : NOUVEAU MOTEUR D'ANIMATION (TIME-OF-FLIGHT)
    # ------------------------------------------------------------
    print("\n[STEP 1] Launching Time-of-Flight Animation Loop...")
    print("--> Close the animation window to proceed to interactive static events.")
    
    # Chargement de l'événement de référence via le package I/O
    event_ref = load_event(mock_hep_event)
    
    # Exécution de l'animation (vitesse radiale contrôlée par le paramètre 'speed')
    visualizer.animate_event(event_ref, p_scale=1.5, j_scale=0.012, speed=0.1)
    
    # ------------------------------------------------------------
    # DEMO 2 : BATTERIE DE TESTS INTERACTIFS STATIQUES
    # ------------------------------------------------------------
    print("\n[STEP 2] Entering Interactive Exploration Canvas Mode...")
    
    # Test de l'événement de référence en mode statique avec picking actif
    print("Rendering Reference Event (Static with Tooltips)...")
    visualizer.plot_event(event_ref, p_scale=1.5, j_scale=0.012)

   
    
    print("\nAll event displays successfully processed.")