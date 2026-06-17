from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Metadata:
    n_periods: int
    n_commodities: int

    n_ports: int
    n_ports_per_region: List[int]

    n_load_regions: int
    n_unload_regions: int
    n_ports_per_load_region: List[int]
    n_ports_per_unload_region: List[int]

    n_vessels: int
    n_vessel_classes: int
    n_vessels_per_class: List[int]

    hours_per_period: float
    spot_market_price: float
    spot_market_discount: float
    reward_finishing_early: float
    single_period_alpha_slack: float


@dataclass
class Port:
    id: int
    name: str
    port_type: str
    region_id: int
    x: float
    y: float
    port_fee: float
    berth_limit: int
    max_transfer_amt: float
    min_transfer_amt: float
    max_inventory: float
    min_inventory: float
    capacity: float
    initial_inventory: float
    revenue: List[float]
    consumption: List[float] = None
    production: List[float] = None


@dataclass
class Vessel:
    id: int
    name: str
    vessel_class: int
    initial_inventory: float
    initial_port: str
    first_available: int


@dataclass
class VesselClass:
    id: int
    name: str
    capacity: float
    speed: float
    travel_cost_per_km: float
    empty_discount: float
    vessels: List[int]
    travel_times: List[int]


@dataclass
class Instance:
    name: str
    type: str
    metadata: Metadata

    ports: List[Port]
    load_ports: List[Port]
    unload_ports: List[Port]

    vessel_classes: List[VesselClass]
    vessels: List[Vessel]

    distances: np.ndarray

    lb: float
    ub: float


@dataclass
class Call:
    port_id: int
    vessel_id: int

    position: int = 0

    prev_vessel_call_id: Optional[int] = None
    next_vessel_call_id: Optional[int] = None
    prev_port_call_id: Optional[int] = None
    next_port_call_id: Optional[int] = None

    port_service_time: float = 0.0
    vessel_service_time: float = 0.0

    arrival_time: Optional[float] = None
    start_service_time: Optional[float] = None
    end_service_time: Optional[float] = None

    inventory_before: float = 0.0
    inventory_after: float = 0.0
    load: Optional[float] = None

    routing_cost: float = 0.0
    inventory_penalty: float = 0.0
    total_cost: float = 0.0

    is_initial: bool = False


@dataclass
class Solution:
    calls: List[Call]

    total_cost: float = 0.0
    routing_cost: float = 0.0
    inventory_penalty: float = 0.0

    vessel_times: Optional[dict] = None
    vessel_ports: Optional[dict] = None
    port_inventory: Optional[dict] = None
    port_violations: dict = None

    is_feasible: bool = True
    violated_horizon: bool = False
