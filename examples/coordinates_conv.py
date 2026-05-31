import sys
import os
import numpy as np

# Ensure Python can find the iris3d package relative to this folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iris3d.models import CollisionEvent, EventMetadata, Particle, Jet
from iris3d.core import CoordinateTransformer

def test_single_particle_kinematics():
    print("Testing: Single Particle Coordinate Mapping...")
    transformer = CoordinateTransformer()
    
    # Track pointing entirely along the X-axis (eta=0, phi=0)
    p_x = Particle(pt=100.0, eta=0.0, phi=0.0, charge=1, pid=13)
    x, y, z = transformer.particle_to_vector(p_x)
    assert np.isclose(x, 100.0) and np.isclose(y, 0.0) and np.isclose(z, 0.0), "X-axis alignment failed"
    
    # Track pointing entirely along the Y-axis (eta=0, phi=pi/2)
    p_y = Particle(pt=50.0, eta=0.0, phi=np.pi / 2, charge=-1, pid=-11)
    x, y, z = transformer.particle_to_vector(p_y)
    assert np.isclose(x, 0.0) and np.isclose(y, 50.0) and np.isclose(z, 0.0), "Y-axis alignment failed"
    
    print("Single particle spatial geometry is accurate.")

def test_jet_cone_geometry():
    print("\n Testing: Jet Cone Unit Vector and Scale Proportions...")
    transformer = CoordinateTransformer()
    
    # Create a 200 GeV jet firing off at an angle
    test_jet = Jet(energy=200.0, eta=1.0, phi=-0.75, delta_r=0.4)
    scale_factor = 0.01
    
    jet_data = transformer.jet_to_cone(test_jet, scale=scale_factor)
    
    # 1. Validate the direction vector is a strict mathematical unit vector (Length = 1.0)
    ux, uy, uz = jet_data["unit_direction"]
    unit_vector_length = np.sqrt(ux**2 + uy**2 + uz**2)
    assert np.isclose(unit_vector_length, 1.0), f"Jet direction is not a true unit vector. Length: {unit_vector_length}"
    
    # 2. Validate endpoint calculation scales exactly with Energy
    ex, ey, ez = jet_data["endpoint"]
    expected_length = test_jet.energy * scale_factor  # 2.0 canvas units
    calculated_length = np.sqrt(ex**2 + ey**2 + ez**2)
    assert np.isclose(calculated_length, expected_length), f"Expected length {expected_length}, got {calculated_length}"
    
    # 3. Validate cone base radius calculation: radius = length * tan(delta_r)
    expected_radius = expected_length * np.tan(test_jet.delta_r)
    assert np.isclose(jet_data["radius"], expected_radius), "Cone opening radius logic mismatch"
    
    print("Jet cone projection math checks out perfectly.")

def test_batch_matrix_vectorization():
    print("\n Testing: Bulk Matrix Vectorization...")
    transformer = CoordinateTransformer()
    
    # Build a mock CollisionEvent manually containing multiple tracks
    mock_event = CollisionEvent(
        metadata=EventMetadata(run_id=1, event_id=1),
        particles=[
            Particle(pt=10.0, eta=0.5, phi=-0.2, charge=1, pid=11),
            Particle(pt=20.0, eta=-1.2, phi=1.5, charge=-1, pid=-11),
            Particle(pt=30.0, eta=2.1, phi=3.0, charge=0, pid=22)
        ],
        jets=[
            Jet(energy=100.0, eta=0.5, phi=-0.2)
        ]
    )
    
    arrays = transformer.extract_event_arrays(mock_event, p_scale=1.5, j_scale=0.05)
    
    # Verify the bulk particle vector output shape is matrix [N, 3]
    assert arrays["particle_vectors"].shape == (3, 3), f"Matrix shape error: {arrays['particle_vectors'].shape}"
    
    # Verify the metadata list mirrors the particle length correctly
    assert len(arrays["particle_metadata"]) == 3, "Metadata mapping truncation error"
    assert arrays["particle_metadata"][0]["name"] == "Track 0", "Automatic fallback track name failed"
    
    # Verify that the vector calculation inside the high-performance loop matches individual translation math
    individual_vector = transformer.particle_to_vector(mock_event.particles[1], scale=1.5)
    matrix_vector = arrays["particle_vectors"][1]
    assert np.allclose(matrix_vector, individual_vector), "NumPy matrix optimization diverges from scalar calculation"
    
    print(" Multi-track matrix vectorization is accurate.")

if __name__ == "__main__":
    print("==================================================")
    print("        IRIS3D CORE MATHEMATICS VERIFICATION      ")
    print("==================================================")
    try:
        test_single_particle_kinematics()
        test_jet_cone_geometry()
        test_batch_matrix_vectorization()
        print("\n ALL GEOMETRY TESTS PASSED! Spatial vectors match physical bounds.")
    except AssertionError as e:
        print(f"\n GEOMETRY CORE TEST FAILED: {e}")
        sys.exit(1)