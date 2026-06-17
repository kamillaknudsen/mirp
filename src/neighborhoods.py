from copy import copy
import random
from models import Call, Solution
from evaluator import evaluate_solution


def swap(instance, solution, random_move=False):
    n = len(solution.calls)
    if n < 2:
        return None
    
    if random_move:
        i, j = random.sample(range(n), 2)
        new_calls = solution.calls.copy()
        new_calls[i], new_calls[j] = new_calls[j], new_calls[i]
        candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
        candidate = evaluate_solution(candidate, instance)

        if candidate.is_feasible and not candidate.violated_horizon:
            return candidate

    for i in range(n):
        for j in range(i+1, n):
            new_calls = solution.calls.copy()
            new_calls[i], new_calls[j] = new_calls[j], new_calls[i]
            candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
            candidate = evaluate_solution(candidate, instance)

            if candidate.is_feasible and not candidate.violated_horizon and candidate.total_cost < solution.total_cost:
                return candidate

    return None


def relocate(instance, solution, random_move=False):
    n = len(solution.calls)
    if n < 2:
        return None
    
    if random_move:
        i, j = random.sample(range(n), 2)
        new_calls = solution.calls.copy()
        call = new_calls.pop(i)
        new_calls.insert(j, call)

        candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
        candidate = evaluate_solution(candidate, instance)

        if candidate.is_feasible and not candidate.violated_horizon:
            return candidate

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            new_calls = solution.calls.copy()
            call = new_calls.pop(i)
            new_calls.insert(j, call)

            candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
            candidate = evaluate_solution(candidate, instance)

            if candidate.is_feasible and not candidate.violated_horizon and candidate.total_cost < solution.total_cost:
                return candidate

    return None


def replace(instance, solution, random_move=False):
    port_map = {p.id: p for p in instance.ports}
    n = len(solution.calls)
    if n ==0:
        return None
    
    if random_move:
        idx = random.randint(0, n-1)
        current_port = port_map[solution.calls[idx].port_id]
        alternative_ports = [p for p in instance.ports if p.port_type == current_port.port_type and p.id != current_port.id]
        if not alternative_ports:
            return None
        alternative_port = random.choice(alternative_ports)
        new_calls = solution.calls.copy()
        new_calls[idx] = copy(new_calls[idx])
        new_calls[idx].port_id = alternative_port.id

        candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
        candidate = evaluate_solution(candidate, instance)
        
        if candidate.is_feasible and not candidate.violated_horizon:
            return candidate

    for idx, call in enumerate(solution.calls):
        current_port = port_map[call.port_id]
        for alternative_port in instance.ports:
            if alternative_port.id == current_port.id:
                continue

            if alternative_port.port_type != current_port.port_type:
                continue

            new_calls = solution.calls.copy()
            new_calls[idx] = copy(new_calls[idx])
            new_calls[idx].port_id = alternative_port.id

            candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
            candidate = evaluate_solution(candidate, instance)

            if candidate.is_feasible and candidate.total_cost < solution.total_cost:
                return candidate
    
    return None


def insert(instance, solution, random_move=False):
    loading_ports = [p for p in instance.ports if p.port_type == 'loading']
    unloading_ports = [p for p in instance.ports if p.port_type == 'unloading']

    if random_move:
        if not loading_ports or not unloading_ports or not instance.vessels:
            return None
        l_port = random.choice(loading_ports)
        u_port = random.choice(unloading_ports)
        vessel = random.choice(instance.vessels)

        new_calls = solution.calls.copy()
        new_calls.append(Call(port_id=l_port.id, vessel_id=vessel.id))
        new_calls.append(Call(port_id=u_port.id, vessel_id=vessel.id))

        candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
        candidate = evaluate_solution(candidate, instance) 
        if candidate.is_feasible and not candidate.violated_horizon:
            return candidate

    for l_port in loading_ports:
        for u_port in unloading_ports:
            for vessel in instance.vessels:
                new_calls = solution.calls.copy()
                new_calls.append(Call(port_id=l_port.id, vessel_id=vessel.id))
                new_calls.append(Call(port_id=u_port.id, vessel_id=vessel.id))

                candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
                candidate = evaluate_solution(candidate, instance)

                if candidate.is_feasible and candidate.total_cost < solution.total_cost:
                    return candidate
    return None


def remove(instance, solution, random_move=False):
    n = len(solution.calls)
    if n < 2:
        return None
    
    if random_move:
        valid_indices = []
        for i, call in enumerate(solution.calls):
            for j in range(i + 1, len(solution.calls)):
                if solution.calls[j].vessel_id == call.vessel_id:
                    valid_indices.append((i, j))
        
        if not valid_indices:
            return None
        
        i, j = random.choice(valid_indices)
        new_calls = solution.calls.copy()
        new_calls.pop(j)
        new_calls.pop(i)

        candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
        candidate = evaluate_solution(candidate, instance)

        if candidate.is_feasible and not candidate.violated_horizon:
            return candidate

    for i, call in enumerate(solution.calls):
        next_same_vessel = None
        for j in range(i + 1, len(solution.calls)):
            if solution.calls[j].vessel_id == call.vessel_id:
                next_same_vessel = j
                break
        
        if next_same_vessel is None:
            continue

        new_calls = solution.calls.copy()
        new_calls.pop(next_same_vessel)
        new_calls.pop(i)

        candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
        candidate = evaluate_solution(candidate, instance)
        if candidate.is_feasible and candidate.total_cost < solution.total_cost:
            return candidate

    return None


def swap_port(instance, solution, random_move=False):
    port_map = {p.id: p for p in instance.ports}
    n = len(solution.calls)
    if n < 2:
        return None
    
    if random_move:
        valid_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if port_map[solution.calls[i].port_id].port_type == port_map[solution.calls[j].port_id].port_type:
                    valid_pairs.append((i, j))

        if not valid_pairs:
            return None

        i, j = random.choice(valid_pairs)
        new_calls = solution.calls.copy()
        new_calls[i] = copy(new_calls[i])
        new_calls[j] = copy(new_calls[j])
        new_calls[i].port_id, new_calls[j].port_id = new_calls[j].port_id, new_calls[i].port_id

        candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
        candidate = evaluate_solution(candidate, instance)
        
        if candidate.is_feasible and not candidate.violated_horizon:
            return candidate


    for i in range(n):
        for j in range(i+1, n):
            port_i = port_map[solution.calls[i].port_id]
            port_j = port_map[solution.calls[j].port_id]

            if port_i.port_type != port_j.port_type:
                continue

            new_calls = solution.calls.copy()
            new_calls[i] = copy(new_calls[i])
            new_calls[j] = copy(new_calls[j])

            new_calls[i].port_id, new_calls[j].port_id = new_calls[j].port_id, new_calls[i].port_id

            candidate = Solution(calls=new_calls, total_cost=0.0, routing_cost=0.0, inventory_penalty=0.0, is_feasible=True)
            candidate = evaluate_solution(candidate, instance)
            if candidate.is_feasible and candidate.total_cost < solution.total_cost:
                return candidate

    return None
