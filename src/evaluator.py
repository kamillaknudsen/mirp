from models import Instance, Solution


def update_call_pointers(solution: Solution):
    last_vessel_call = {}
    last_port_call = {}

    for i, call in enumerate(solution.calls):
        call.position = i
        call.prev_vessel_call_id = None
        call.next_vessel_call_id = None
        call.prev_port_call_id = None
        call.next_port_call_id = None

    for i, call in enumerate(solution.calls):
        vessel_id = call.vessel_id
        port_id = call.port_id

        if vessel_id in last_vessel_call:
            prev_idx = last_vessel_call[vessel_id]
            call.prev_vessel_call_id = prev_idx
            solution.calls[prev_idx].next_vessel_call_id = i

        if port_id in last_port_call:
            prev_idx = last_port_call[port_id]
            call.prev_port_call_id = prev_idx
            solution.calls[prev_idx].next_port_call_id = i

        last_vessel_call[vessel_id] = i
        last_port_call[port_id] = i


def evaluate_schedule(solution: Solution, instance: Instance, time_horizon, class_map, port_map, vessel_map):
    vessel_times = {}
    vessel_ports = {}
    berth_occupancy = {}

    for vessel in instance.vessels:
        vessel_times[vessel.id] = vessel.first_available
        vessel_ports[vessel.id] = vessel.initial_port

    for port in instance.ports:
        berth_occupancy[port.id] = {t: 0 for t in range(time_horizon + 1)}

    for call in solution.calls:
        vessel = vessel_map[call.vessel_id]
        vessel_class = class_map[vessel.vessel_class]

        port = port_map[call.port_id]
        previous_port_id = vessel_ports[call.vessel_id]

        previous_port = port_map[previous_port_id]

        travel_time = vessel_class.travel_times[previous_port_id - 1][port.id - 1]
        arrival_time = vessel_times[vessel.id] + travel_time

        start_service = arrival_time
        service_duration = 1 # assumption

        while start_service < time_horizon:
            if berth_occupancy[port.id][start_service] < port.berth_limit:
                break
            start_service += 1

        departure_time = start_service + service_duration

        if start_service >= time_horizon:
            solution.violated_horizon = True
            solution.is_feasible = False
            break

        berth_occupancy[port.id][start_service] += 1

        if previous_port.port_type == port.port_type and previous_port.id != port.id:
            solution.is_feasible = False
            break

        call.arrival_time = arrival_time
        call.start_service_time = start_service
        call.end_service_time = departure_time
        call.vessel_service_time = vessel_times[vessel.id]
        call.port_service_time = 0

        # ROUTING COST

        distance = instance.distances[previous_port_id - 1][port.id - 1]
        routing_cost = distance * vessel_class.travel_cost_per_km

        # make sure its the correct formula
        if previous_port.port_type == 'unloading' and port.port_type == 'loading':
            routing_cost *= (1 - vessel_class.empty_discount)

        routing_cost += port.port_fee
        call.routing_cost = routing_cost
        solution.routing_cost += routing_cost

        vessel_times[vessel.id] = departure_time
        vessel_ports[vessel.id] = port.id

    for v_id, last_departure in vessel_times.items():
        if last_departure < time_horizon:
            periods_saved = time_horizon - last_departure
            solution.total_cost -= periods_saved * instance.metadata.reward_finishing_early

    return vessel_times, vessel_ports


def evaluate_inventories(solution: Solution, instance: Instance, time_horizon , class_map, port_map, vessel_map):
    port_inventory = {}
    port_times = {}
    port_violations = {}

    for port in instance.ports:
        port_times[port.id] = 0
        port_inventory[port.id] = {t: 0 for t in range(time_horizon + 1)}
        port_inventory[port.id][0] = port.initial_inventory
        port_violations[port.id] = []

    chronological_calls = sorted([c for c in solution.calls if c.start_service_time is not None], key=lambda c: c.start_service_time)

    for call in chronological_calls:
        port = port_map[call.port_id]
        vessel = vessel_map[call.vessel_id]
        vessel_class = class_map[vessel.vessel_class]

        delta_j = 1 if port.port_type == 'loading' else -1
        last_t = port_times[port.id]
        start_service = call.start_service_time

        for t in range (last_t+1, start_service+1):
            d_jt = port.production[t-1] if delta_j == 1 else port.consumption[t-1]

            port_inventory[port.id][t] = port_inventory[port.id][t-1] + (delta_j * d_jt)

            if t < start_service:
                if port_inventory[port.id][t] < 0:
                    shortage = 0 - port_inventory[port.id][t]
                    solution.inventory_penalty += shortage * instance.metadata.spot_market_price
                    port_violations[port.id].append(t)

                elif port_inventory[port.id][t] > port.max_inventory:
                    overflow = port_inventory[port.id][t] - port.max_inventory
                    solution.inventory_penalty +=  overflow * instance.metadata.spot_market_price
                    port_violations[port.id].append(t)

                if port_inventory[port.id][t] < 0:
                    port_inventory[port.id][t] = 0
                elif port_inventory[port.id][t] > port.max_inventory:
                    port_inventory[port.id][t] = port.max_inventory

        quantity = vessel_class.capacity

        if port.port_type == 'loading':
            port_inventory[port.id][start_service] -= quantity
        else:
            port_inventory[port.id][start_service] += quantity

        if port_inventory[port.id][start_service] < 0:
            shortage = 0 - port_inventory[port.id][start_service]
            solution.inventory_penalty += shortage * instance.metadata.spot_market_price
            port_violations[port.id].append(start_service)
            port_inventory[port.id][start_service] = 0

        elif port_inventory[port.id][start_service] > port.max_inventory:
            overflow = port_inventory[port.id][start_service] - port.max_inventory
            solution.inventory_penalty += overflow * instance.metadata.spot_market_price
            port_violations[port.id].append(start_service)
            port_inventory[port.id][start_service] = port.max_inventory

        port_times[port.id] = start_service
    
    # track inventories after calls
    for port in instance.ports:
        delta_j = 1 if port.port_type == 'loading' else -1
        last_t = port_times[port.id]
        for t in range (last_t+1, time_horizon+1):
            d_jt = port.production[t-1] if delta_j == 1 else port.consumption[t-1]
            port_inventory[port.id][t] = port_inventory[port.id][t-1] + (delta_j * d_jt)

            if port_inventory[port.id][t] < 0:
                shortage = 0 - port_inventory[port.id][t]
                solution.inventory_penalty +=  shortage * instance.metadata.spot_market_price
                port_violations[port.id].append(t)
                port_inventory[port.id][t] = 0

            elif port_inventory[port.id][t] > port.max_inventory:
                overflow = port_inventory[port.id][t] - port.max_inventory
                solution.inventory_penalty += overflow * instance.metadata.spot_market_price
                port_violations[port.id].append(t)
                port_inventory[port.id][t] = port.max_inventory

    solution.port_violations = port_violations
    
    return port_inventory


def evaluate_solution(solution: Solution, instance: Instance) -> Solution:
    solution.total_cost = 0.0
    solution.routing_cost = 0.0
    solution.inventory_penalty = 0.0
    solution.is_feasible = True
    solution.violated_horizon = False
    solution.port_violations = {}

    port_map = {p.id: p for p in instance.ports}
    class_map = {vc.id: vc for vc in instance.vessel_classes}
    vessel_map = {v.id: v for v in instance.vessels}

    update_call_pointers(solution)

    time_horizon = instance.metadata.n_periods

    vessel_times, vessel_ports = evaluate_schedule(solution, instance, time_horizon, class_map, port_map, vessel_map)

    port_inventory = evaluate_inventories(solution, instance, time_horizon, class_map, port_map, vessel_map)

    solution.total_cost = solution.routing_cost + solution.inventory_penalty
    if solution.violated_horizon:
        solution.is_feasible = False
    
    solution.vessel_times = vessel_times
    solution.vessel_ports = vessel_ports
    solution.port_inventory = port_inventory

    return solution
