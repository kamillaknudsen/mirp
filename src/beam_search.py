from copy import deepcopy
from models import Instance, Solution, Call
from greedy_algorithm import find_next_violating_ports, find_earliest_vessels, greedy_ranodmized_algorithm, evaluate_successor_with_median
from evaluator import evaluate_solution


def generate_successors(partial_solution, instance, w):
    candidate_children = []

    violating_ports = find_next_violating_ports(instance, partial_solution.port_violations)
    if not violating_ports:
        return []
    
    for port, violation_time in violating_ports:
        candidate_vessels = find_earliest_vessels(port, partial_solution.vessel_times, partial_solution.vessel_ports, instance)

        for vessel, arrival_time in candidate_vessels:
            child_node = deepcopy(partial_solution)
            child_node.calls.append(Call(port_id=port.id, vessel_id=vessel.id))

            evaluated_child = evaluate_solution(child_node, instance)
            if evaluated_child.is_feasible and not evaluated_child.violated_horizon:
                candidate_children.append((evaluated_child, violation_time, arrival_time))

    candidate_children.sort(key=lambda x: (x[1], x[2]))
    successors = [child for child, _, _ in candidate_children[:w]]

    return successors


def beam_search(instance: Instance, initial_solution: Solution, N: int, q: int, w: int, std_deviation) -> Solution:
    
    initial_solution = evaluate_solution(initial_solution, instance)
    beam = [(deepcopy(initial_solution), initial_solution.total_cost)]

    global_best_solutions = []
    time_horizon = instance.metadata.n_periods

    while True: 
        successor_pool = []
        scores = set()
        for partial_solution, _ in beam:
            if all(t >= time_horizon for t in partial_solution.vessel_times.values()):
                continue

            children = generate_successors(partial_solution, instance, w)

            for child in children:
                gra_runs, median_score = evaluate_successor_with_median(instance, child, std_deviation, q)

                for full_solution in gra_runs:
                    if full_solution.is_feasible and not full_solution.violated_horizon:
                        global_best_solutions.append(full_solution)
                
                if median_score in scores:
                    continue

                scores.add(median_score)
                successor_pool.append((child, median_score))

        if not successor_pool:
            break

        successor_pool.sort(key=lambda x: x[1])
        beam = successor_pool[:N]
    
    for final_partial, _ in beam:
        final_complete = greedy_ranodmized_algorithm(instance, std_deviation, final_partial, is_randomized=False)
        if final_complete.is_feasible and not final_complete.violated_horizon:
            global_best_solutions.append(final_complete)

    feasible_candidates = [s for s in global_best_solutions if s.is_feasible and not s.violated_horizon]

    if not feasible_candidates:
        raise ValueError("No feasible solutions found during Beam Search")
    
    feasible_candidates.sort(key=lambda x: x.total_cost)

    return feasible_candidates[0]
