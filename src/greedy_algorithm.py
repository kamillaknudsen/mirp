import random
import math
from copy import deepcopy, copy
import numpy as np

from models import Call, Solution
from evaluator import evaluate_solution


def find_next_violating_ports(instance, port_violations): 
    violating_ports = []

    for port in instance.ports:
        violations = port_violations.get(port.id, [])

        if violations:
            earliest_violation = violations[0]
            violating_ports.append((port, earliest_violation))

    violating_ports.sort(key=lambda x: x[1])

    return violating_ports


def find_earliest_vessels(port, vessel_times, vessel_ports, instance):
    candidate_vessels = []
    class_map = {vc.id: vc for vc in instance.vessel_classes}
    port_map = {p.id: p for p in instance.ports}

    for vessel in instance.vessels:
       current_port_id = vessel_ports[vessel.id]
       current_port = port_map[current_port_id]

       if current_port.port_type == port.port_type:
           continue
       
       if current_port_id == port.id:
           continue

       vessel_class = class_map[vessel.vessel_class]

       travel_time = vessel_class.travel_times[current_port_id - 1][port.id - 1]
       arrival = vessel_times[vessel.id] + travel_time

       candidate_vessels.append((vessel, arrival))
    
    candidate_vessels.sort(key=lambda x: x[1])

    return candidate_vessels


def select_normal_index(num_candidates, standard_deviation):
    if num_candidates <= 1:
        return 0
    
    sampled_value = abs(np.random.normal(loc=0, scale=standard_deviation))
    index = int(np.floor(sampled_value))

    return min(index, num_candidates - 1)


def select_port_probabilistically(violating_ports, std_deviation):
    choosen_index = select_normal_index(len(violating_ports), std_deviation)

    return violating_ports[choosen_index][0]


def select_vessel_probabilistically(candidate_vessels, std_deviation):
    choosen_index = select_normal_index(len(candidate_vessels), std_deviation)

    return candidate_vessels[choosen_index][0]


def greedy_ranodmized_algorithm(instance, std_deviation, partial_solution=None, is_randomized=True):
    if partial_solution is not None:
        working_calls = partial_solution.calls.copy()
    else:
        working_calls = []

    time_horizon = instance.metadata.n_periods
    solution_obj = Solution(calls=working_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
    evaluated = evaluate_solution(solution_obj, instance)

    while True:
        if all(t >= time_horizon for t in evaluated.vessel_times.values()):
            break

        violating_ports = find_next_violating_ports(instance, evaluated.port_violations)

        if not violating_ports:
            break
        
        if is_randomized:
            next_port = select_port_probabilistically(violating_ports, std_deviation)
        else:
            next_port = violating_ports[0][0]

        candidate_vessels = find_earliest_vessels(next_port, evaluated.vessel_times, evaluated.vessel_ports, instance)

        if not candidate_vessels:
            break

        if is_randomized:
            next_vessel = select_vessel_probabilistically(candidate_vessels, std_deviation)
        else:
            next_vessel = candidate_vessels[0][0]

        working_calls.append(Call(port_id=next_port.id, vessel_id=next_vessel.id))

        next_candidate = Solution(calls=working_calls)
        new_evaluated = evaluate_solution(next_candidate, instance)

        if not new_evaluated.is_feasible or new_evaluated.violated_horizon:
            working_calls.pop()
            break

        evaluated = new_evaluated

    return evaluated


def evaluate_successor_with_median(instance, partial_solution, std_deviation, q):
    completed_solutions = []

    deterministic = greedy_ranodmized_algorithm(instance, std_deviation, partial_solution, is_randomized=False)
    completed_solutions.append(deterministic)

    for _ in range(q-1):
        randomized = greedy_ranodmized_algorithm(instance, std_deviation, partial_solution, is_randomized=True)
        completed_solutions.append(randomized)

    costs = [s.total_cost for s in completed_solutions]
    return completed_solutions, np.median(costs)
