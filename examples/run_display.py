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

mock_event_extreme_pt = ak.Record({
    "metadata": {"run_id": 402130, "event_id": 99825, "sqrts_gev": 13600.0},
    "particles": {
        # 1. Un muon de type "LHC Run 3" à impulsion transverse massive (doit être quasi-droit)
        # 2. Un pion à ultra-basse énergie (va boucler intensément et s'enrouler)
        # 3. Un hadron neutre à fort pT (ligne droite continue)
        # 4. Un positon à pT intermédiaire (courbe modérée)
        "pt": ak.Array([500.0, 0.4, 85.0, 12.0]),
        "eta": ak.Array([0.05, 0.1, -0.8, 1.4]),
        "phi": ak.Array([0.5, -0.5, 2.1, -1.2]),
        "charge": ak.Array([-1, 1, 0, 1]),
        "pid": ak.Array([13, 211, 130, -11]),  # Muon, Pion+, Kaon Long (neutre), Positon
        "name": ak.Array(["mu_hard-", "pi_soft+", "K_long", "e+"])
    },
    "jets": {
        # Un seul jet sur-énergétique englobant la particule dure
        "energy": ak.Array([650.0]),
        "eta": ak.Array([0.05]),
        "phi": ak.Array([0.5]),
        "delta_r": ak.Array([0.4])
    }
})

mock_event_higgs_diphoton = ak.Record({
    "metadata": {"run_id": 402130, "event_id": 99826, "sqrts_gev": 13600.0},
    "particles": {
        # Deux photons énergétiques principaux opposés dans le détecteur (Back-to-back)
        # Accompagnés de deux traces chargées molles de "bruit"
        "pt": ak.Array([65.0, 60.0, 1.2, 0.9]),
        "eta": ak.Array([0.4, -0.4, 2.2, -1.8]),
        "phi": ak.Array([1.0, -2.14, 0.1, 3.0]),
        "charge": ak.Array([0, 0, 1, -1]),
        "pid": ak.Array([22, 22, 211, -211]),  # Photon, Photon, Pion+, Pion-
        "name": ak.Array(["gamma_1", "gamma_2", "ch_noise1", "ch_noise2"])
    },
    "jets": {
        # Pas de jets reconstruits ici (les photons isolés ne font pas de jets)
        "energy": ak.Array([]),
        "eta": ak.Array([]),
        "phi": ak.Array([]),
        "delta_r": ak.Array([])
    }
})

mock_event_empty_edge_case = ak.Record({
    "metadata": {"run_id": 402130, "event_id": 99827, "sqrts_gev": 13600.0},
    "particles": {
        "pt": ak.Array([]),
        "eta": ak.Array([]),
        "phi": ak.Array([]),
        "charge": ak.Array([]),
        "pid": ak.Array([]),
        "name": ak.Array([])
    },
    "jets": {
        "energy": ak.Array([]),
        "eta": ak.Array([]),
        "phi": ak.Array([]),
        "delta_r": ak.Array([])
    }
})

mock_event_w_boson = ak.Record({
    "metadata": {"run_id": 402130, "event_id": 99828, "sqrts_gev": 13600.0},
    "particles": {
        # Un seul électron très dur propulsé vers la droite (phi = 0.0)
        # Aucune autre particule majeure pour équilibrer. Le neutrino doit partir vers la gauche (phi = pi)
        "pt": ak.Array([80.0, 1.5, 0.8]),
        "eta": ak.Array([0.1, -1.2, 0.5]),
        "phi": ak.Array([0.0, 0.8, -1.5]),
        "charge": ak.Array([-1, 1, -1]),
        "pid": ak.Array([11, 211, -211]), 
        "name": ak.Array(["e_signal-", "pion1", "pion2"])
    },
    "jets": {
        "energy": ak.Array([]),
        "eta": ak.Array([]),
        "phi": ak.Array([]),
        "delta_r": ak.Array([])
    }
})

if __name__ == "__main__":
    print("Ingesting experimental mock event data block...")
    event = load_event(mock_hep_event)
    
    print("Initializing fluid PyVista 3D event viewer...")
    visualizer = EventVisualizer()
    
    # Fire up the graphics layer!
    visualizer.plot_event(event, p_scale=2.2, j_scale=0.012)

    visualizer = EventVisualizer()

    # TEST 1 : Vérifie les hélices extrêmes
    print("Rendering Extreme pT Event...")
    visualizer.plot_event(mock_event_extreme_pt, p_scale=1.5, j_scale=0.01)

    # TEST 2 : Vérifie les pointillés des photons du Higgs
    print("Rendering Higgs di-photon Event...")
    visualizer.plot_event(mock_event_higgs_diphoton, p_scale=1.5)

    # TEST 3 : Vérifie la robustesse (doit afficher un détecteur vide avec le vertex au centre, sans cracher)
    print("Rendering Empty Edge Case Event...")
    visualizer.plot_event(mock_event_empty_edge_case)

    # TEST 3 : Vérifie la robustesse (doit afficher un détecteur vide avec le vertex au centre, sans cracher)
    print("Rendering BOSON RED...")
    visualizer.plot_event(mock_event_w_boson)