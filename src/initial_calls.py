from copy import deepcopy
from models import Solution

def create_initial_call(vessel, instance):
    call = type('Call', (object,), {})()
    call.vessel_id = vessel.id
    call.port_id = vessel.initial_port

    call.arrival_time = vessel.first_available
    call.start_service_time = vessel.first_available
    call.end_service_time = vessel.first_available + 1

    port = next(p for p in instance.ports if p.id == vessel.initial_port)

    call.routing_cost = port.port_fee

    call.prev_vessel_call_id = None
    call.next_vessel_call_id = None
    call.prev_port_call_id = None
    call.next_port_call_id = None

    call.is_initial_call = True

    return call
 

def add_initial_calls_to_solution(instance, solution):
    solution_copy = solution

    initial_calls = []
    for vessel in instance.vessels:
        call = create_initial_call(vessel, instance)
        initial_calls.append(call)

    solution_copy.calls = initial_calls + solution_copy.calls
    
    return solution_copy


def create_empty_solution_with_initial_calls(instance):
    solution = Solution(calls=[], total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True, violated_horizon = False, port_violations = {})

    for vessel in instance.vessels:
        call = create_initial_call(vessel, instance)
        solution.calls.append(call)

    return solution