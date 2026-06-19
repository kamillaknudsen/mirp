import random
import math
import time
import logging
from copy import deepcopy

from evaluator import evaluate_solution
from neighborhoods import *

logger = logging.getLogger(__name__)


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


def perturb(instance, solution):
    neighborhoods = [swap, relocate, replace, insert, remove, swap_port]
    perturbed_solution = solution

    for _ in range(2):
        chosen_neighborhood = random.choice(neighborhoods)
        candidate = chosen_neighborhood(instance, perturbed_solution, random_move=True)

        if candidate is not None:
            perturbed_solution = candidate
    
    perturbed_solution = evaluate_solution(perturbed_solution, instance)
    return perturbed_solution


def iterated_local_search(instance, initial_solution, max_iterations, max_non_improving, initial_temp, cooling_rate, max_runtime):
    instance_start_time = time.time()

    best_solution = rvnd(instance, deepcopy(initial_solution))
    current_solution = best_solution

    temperature = initial_temp
    non_improving_counter = 0
    
    for iteration in range(max_iterations):
        elapsed_timee = time.time() - instance_start_time
        if elapsed_timee > max_runtime:
            logger.warning("INSTANCE TIMEOUT TRIGGERED!")
            break

        perturbed_solution = perturb(instance, current_solution)
        refined_solution = rvnd(instance, perturbed_solution)

        cost_change = refined_solution.total_cost - current_solution.total_cost

        # Simulated annealing acceptance criterion
        if cost_change < 0:
            current_solution = refined_solution

            if current_solution.total_cost < best_solution.total_cost:
                best_solution = deepcopy(current_solution)
                non_improving_counter = 0
            else:
                non_improving_counter += 1
        else:
            probability = math.exp(-cost_change / temperature)

            if random.random() < probability:
                current_solution = refined_solution

            non_improving_counter += 1

        if non_improving_counter > max_non_improving:
            current_solution = deepcopy(best_solution)
            non_improving_counter = 0

        temperature *= cooling_rate

    return best_solution
