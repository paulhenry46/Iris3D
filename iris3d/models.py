from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class EventMetadata:
    run_id: int
    event_id: int
    sqrts_gev: Optional[float] = 13600.0  #LHC Run 3 LHC

@dataclass(frozen=True)
class Particle:
    pt: float
    eta: float
    phi: float
    charge: int
    pid: int
    name: Optional[str] = None  # Optional

@dataclass(frozen=True)
class Jet:
    energy: float
    eta: float
    phi: float
    delta_r: float = 0.4  #  LHC default value

@dataclass
class CollisionEvent:
    metadata: EventMetadata
    particles: List[Particle] = field(default_factory=list)
    jets: List[Jet] = field(default_factory=list)