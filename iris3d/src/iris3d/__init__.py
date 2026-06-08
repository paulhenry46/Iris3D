"""
iris3D: Interactive 3D/2D Particle Collision Event Display
"""
__version__ = "1.0.0"

from .models import CollisionEvent, Particle, Jet, EventMetadata
from .vis import EventVisualizer

__all__ = [
    "Event",
    "Particle",
    "EventVisualizer",
    "CollisionEvent",
    "Jet",
    "EventMetadata"
    "__version__",
]