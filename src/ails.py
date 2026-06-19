import random
import math
import time
import logging
from copy import deepcopy

from evaluator import evaluate_solution
from neighborhoods import *

logger = logging.getLogger(__name__)


def select_neighborhood(neighborhoods, weights):
    total_weight = sum(weights)
    if total_weight == 0:
        chosen_idx = random.choice(range(len(neighborhoods)))
        return neighborhoods[chosen_idx], chosen_idx
    
    probabilities = [w / total_weight for w in weights]
    chosen_idx = random.choices(range(len(neighborhoods)), weights=probabilities, k=1)[0]
    return neighborhoods[chosen_idx], chosen_idx


def calculate_symmetric_difference(sol_a, sol_b):
    calls_a = set((call.port_id, call.vessel_id, call.position) for call in sol_a.calls)
    calls_b = set((call.port_id, call.vessel_id, call.position) for call in sol_b.calls)

    symmetric_diff = calls_a.symmetric_difference(calls_b)
    return len(symmetric_diff)


def rvnd(instance, initial_solution):
    current_solution = initial_solution
    neighborhoods = [swap, relocate, replace, insert, remove, swap_port]

    while True:
        improved = False
        random.shuffle(neighborhoods)

        for neighborhood in neighborhoods:
            candidate = neighborhood(instance, current_solution, random_move=False)

            if candidate is not None and candidate.total_cost < current_solution.total_cost:
                current_solution = candidate
                improved = True
                break
        
        if not improved:
            break
    
    return current_solution


def adaptive_iterated_local_search(instance, initial_solution, max_iterations, max_non_improving, 
                                   max_runtime, sigma1, sigma2, sigma3, segment_size, reaction_factor, 
                                   gamma, d_beta, target_acceptance_rate, initial_threshold, epsilon):
    instance_start_time = time.time()

    neighborhoods = [swap, relocate, replace, insert, remove, swap_port]
    num_neighborhoods = len(neighborhoods)

    weights = [1.0] * num_neighborhoods
    pi_scores = [0] * num_neighborhoods
    theta_counts = [0] * num_neighborhoods

    p = 2
    k_counter = 0
    d_bar_avg = 0.0

    best_solution = rvnd(instance, deepcopy(initial_solution))
    current_solution = best_solution

    non_improving_counter = 0

    f_bar = best_solution.total_cost
    rho = initial_threshold
    window_costs = []
    accepted_in_window = 0
    
    for iteration in range(1, max_iterations + 1):
        elapsed_timee = time.time() - instance_start_time
        if elapsed_timee > max_runtime:
            logger.warning("INSTANCE TIMEOUT TRIGGERED!")
            break

        reference_solution = current_solution
        perturbed_solution = deepcopy(current_solution)

        neighborhoods_used = set()

        num_perturbations = max(1, int(round(p)))
        for _ in range(num_perturbations):
            chosen_neighborhood, n_idx = select_neighborhood(neighborhoods, weights)

            theta_counts[n_idx] += 1
            neighborhoods_used.add(n_idx)

            candidate = chosen_neighborhood(instance, perturbed_solution, random_move=True)
            if candidate is not None:
                perturbed_solution = candidate
    
        perturbed_solution = evaluate_solution(perturbed_solution, instance)

        refined_solution = rvnd(instance, perturbed_solution)
        f_s = refined_solution.total_cost

        if iteration > gamma:
            f_bar = f_bar * (1 - (1 / gamma)) + (f_s / gamma)
        else:
            f_bar = (f_bar * (iteration - 1) + f_s) / iteration

        window_costs.append(f_s)
        if len(window_costs) > gamma:
            window_costs.pop(0)

        f_under = min(window_costs)

        b_bar = f_under + rho * (f_bar - f_under)

        distance = calculate_symmetric_difference(refined_solution, reference_solution)
        k_counter += 1
        d_bar_avg = ((d_bar_avg * (k_counter-1)) + distance) / k_counter

        if k_counter == gamma:
            if d_bar_avg > 0:
                p = (p * d_beta) / d_bar_avg
            n_p = len(refined_solution.calls)
            p = max(1.0, min(p, n_p))
            k_counter, d_bar_avg = 0, 0.0

        score = 0

        if f_s < b_bar:
            current_solution = refined_solution
            accepted_in_window += 1

            if f_s < reference_solution.total_cost:
                score = sigma2

                if f_s < best_solution.total_cost:
                    best_solution = deepcopy(refined_solution)
                    non_improving_counter = 0
                    score = sigma1
                else:
                    non_improving_counter += 1
            else:
                score = sigma3
                non_improving_counter += 1
        else: 
            non_improving_counter += 1

        if score > 0:
            for n_idx in neighborhoods_used:
                pi_scores[n_idx] += score

        if non_improving_counter > max_non_improving:
            current_solution = deepcopy(best_solution)
            non_improving_counter = 0

        if iteration % gamma == 0:
            kappa_r = accepted_in_window / gamma
            if kappa_r > 0:
                rho = (target_acceptance_rate * rho) / kappa_r
            else:
                rho = (target_acceptance_rate * rho) / (1 / gamma)

            rho = max(epsilon, rho)
            accepted_in_window=0

        if iteration % segment_size == 0:
            for i in range(num_neighborhoods):
                if theta_counts[i] > 0:
                    weights[i] = weights[i] * (1 - reaction_factor) + reaction_factor * (pi_scores[i] / theta_counts[i])
                else:
                    weights[i] = weights[i] * (1 - reaction_factor)
                weights[i] = max(0.01, weights[i])

            pi_scores = [0] * num_neighborhoods
            theta_counts = [0] * num_neighborhoods

    return best_solution
