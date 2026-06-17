from pathlib import Path
from models import Instance, Port, Vessel, VesselClass, Metadata
import json
import numpy as np


def load_instance(file_path: Path) -> Instance:

    with open(file_path, 'r') as f:
        data = json.load(f)

    m_data = data['metadata']
    metadata = Metadata(
        n_periods=m_data['n_periods'],
        n_commodities=m_data['n_commodities'],
        n_ports=m_data['n_ports'],
        n_ports_per_region=m_data['n_ports_per_region'],
        n_load_regions=m_data['n_load_regions'],
        n_unload_regions=m_data['n_unload_regions'],
        n_ports_per_load_region=m_data['n_ports_per_load_region'],
        n_ports_per_unload_region=m_data['n_ports_per_unload_region'],
        n_vessels=m_data['n_vessels'],
        n_vessel_classes=m_data['n_vessel_classes'],
        n_vessels_per_class=m_data['n_vessels_per_class'],
        hours_per_period=m_data['hours_per_period'],
        spot_market_price=m_data['spot_market_price'],
        spot_market_discount=m_data['spot_market_discount'],
        reward_finishing_early=m_data['reward_finishing_early'],
        single_period_alpha_slack=m_data['single_period_alpha_slack'],
    )
    
    def parse_port(p:dict) -> Port:
        return Port(
            id=p['id'],
            name=p['name'],
            port_type=p['type'],
            region_id=p['region_id'],
            x=p['x'],
            y=p['y'],
            port_fee=p['fee'],
            berth_limit=p['berth_limit'],
            max_inventory=p['capacity'],
            min_inventory=0,
            max_transfer_amt=p['max_amt'],
            min_transfer_amt=p['min_amt'],
            capacity=p['capacity'],
            initial_inventory=p['inventory'],
            production=p.get('rates', []) if p['Δ'] == 1 else None,
            consumption=p.get('rates', []) if p['Δ'] == -1 else None,
            revenue=p.get('prices')
            )
    
    ports = [parse_port(p) for p in data['ports']]
    load_ports = [parse_port(p) for p in data['load_ports']]
    unload_ports = [parse_port(p) for p in data['unload_ports']]

    vessels = []
    for v in data['vessels']:
        vessel = Vessel(
            id=v['id'],
            name=v['name'],
            vessel_class=v['class_id'],
            initial_inventory=v['inventory'],
            initial_port=v['initial_port_id'],
            first_available=v['first_time']
        )
        vessels.append(vessel)

    vessel_classes = []
    for vc in data['classes']:
        vessel_class = VesselClass(
            id=vc['id'],
            name=vc['name'],
            capacity=vc['capacity'],
            speed=vc['speed'],
            vessels=vc['vessel_ids'],
            travel_cost_per_km=vc['travel_cost_km'],
            empty_discount=vc['discount_empty'],
            travel_times=vc['travel_times']
        )
        vessel_classes.append(vessel_class)

    dist_matrix = np.array(data['distances'])

    instance = Instance(
        name=data['name'],
        type=data['type'],
        metadata=metadata,
        ports=ports,
        load_ports=load_ports,
        unload_ports=unload_ports,
        vessels=vessels,
        vessel_classes=vessel_classes,
        distances=dist_matrix,
        lb=data['lb'],
        ub=data['ub'],
    )

    return instance
