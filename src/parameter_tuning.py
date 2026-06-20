import time
import pandas as pd
import csv
from pathlib import Path
import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from data_loader import load_instance
from initial_calls import create_empty_solution_with_initial_calls
from beam_search import beam_search
from ails import adaptive_iterated_local_search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calc_gap(instance_name, final_cost):
    repo_root = Path(__file__).resolve().parent.parent
    bounds_path = repo_root / "data" / "bounds.txt"

    df_bounds = pd.read_csv(bounds_path, sep=r'\s+', header=None,
                            names=['instance_name', 'periods', 'lower_bound', 'z_ref'])
    
    instance_bound = df_bounds[df_bounds['instance_name'] == instance_name]

    z_ref = float(instance_bound['z_ref'].iloc[0])
    gap = round(((final_cost - z_ref) / z_ref) * 100, 2)

    return gap


def run_single_instance(instance_name, file_path, params, combo_idx, tuned_factor, output_csv, write_lock):
    try: 
        instance = load_instance(file_path)
        initial_solution = create_empty_solution_with_initial_calls(instance)

        # 1. Run Beam Search
        start_time_bs = time.time()
        bs_solution = beam_search(instance, initial_solution, params['N'], params['q'], params['w'], params['std_deviation'])
        runtime_bs = time.time() - start_time_bs

        remaining_runtime = params['instance_max_runtime'] - runtime_bs
        remaining_runtime = max(1.0, remaining_runtime)

        # 2. Run Iterated Local Search
        start_time_ils = time.time()
        final_solution = adaptive_iterated_local_search(instance, bs_solution, params['max_iterations'], 
                                                            params['max_non_improving'], remaining_runtime,
                                                            params['sigma1'], params['sigma2'],
                                                            params['sigma3'], params['segment_size'],
                                                            params['reaction_factor'],  params['gamma'],
                                                            params['d_beta'], params['target_acceptance_rate'],
                                                            params['initial_threshold'], params['epsilon'],
                                                            )
        runtime_ils = time.time() - start_time_ils
            
        total_runtime = runtime_bs + runtime_ils

        gap_percent = calc_gap(instance_name, final_solution.total_cost)

        row_payload = [combo_idx, tuned_factor, instance_name, final_solution.total_cost,
                       gap_percent, total_runtime,  params['d_beta'],
                       params['target_acceptance_rate'], params['initial_threshold'],
                       params['reaction_factor']]

        with write_lock:
            with open(output_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_payload)
                    f.flush()

            logger.info(f"SUCCESS | {file_path.name} | Config {combo_idx} | Instance: {instance_name} | Cost: {final_solution.total_cost} | Gap: {gap_percent}%")

    except Exception as e:
        import traceback
        logger.error(f"CRASHED on {file_path.name}: {str(e)}\n{traceback.format_exc()}")


    return gap_percent


def tune_parameters_async():
    instances = [
        'LR1_DR02_VC03_V8a',
        'LR1_DR02_VC01_V6a',
        #'LR1_DR02_VC03_V7a',
        #'LR1_DR02_VC02_V6a'
        ]

    repo_root = Path(__file__).resolve().parent.parent
    output_csv = Path("tuning_results.csv")

    base_parameters = {
        'N': 1000,
        'q': 3,
        'w': 2,
        'std_deviation': 1,
        'max_iterations': 640,
        'max_non_improving': 4,
        "instance_max_runtime": 43100,
        'sigma1': 5,
        'sigma2': 2,
        'sigma3': 1,
        'segment_size': 100,
        'reaction_factor': 0.1,
        'gamma': 20,
        'd_beta': 0.3,
        "target_acceptance_rate": 0.1,
        "initial_threshold": 0.05,
        "epsilon": 0.001
    }

    tune_params = {
        'reaction_factor': [0.05, 0.10, 0.15],
        'd_beta':[0.20, 0.30, 0.40],
        "target_acceptance_rate": [0.05, 0.10, 0.15],
        "initial_threshold": [0.01, 0.05, 0.10],
    }

    configurations_to_run = []

    configurations_to_run.append(("baseline", base_parameters.copy()))

    for factor_key, test_values in tune_params.items():
        for value in test_values:
            if value == base_parameters[factor_key]:
                continue

            modified_config = base_parameters.copy()
            modified_config[factor_key] = value
            configurations_to_run.append((factor_key, modified_config))

    total_tasks = len(configurations_to_run) * len(instances)
    logger.info(f"Initializing Async OFAT Tuning: {len(configurations_to_run)} variations x {len(instances)} instances = {total_tasks} jobs.")

    if not output_csv.exists():
        headers = ["combination_idx", "tuned_factor", "instance_name", "final_cost", "gap_percent", "runtime_sec" , "reaction_factor", "d_beta", "target_acceptance_rate", "initial_threshold"]
        with open(output_csv, mode='w', newline='') as f:
            csv.writer(f).writerow(headers)

    manager = multiprocessing.Manager()
    write_lock = manager.Lock()
    available_cores = os.cpu_count() or 2
    logger.info(f"Spawning async executor pool across {available_cores} CPU processes.")

    with ProcessPoolExecutor(max_workers=available_cores) as executor:
        futures = {}

        for combo_idx, (tuned_factor, active_config) in enumerate(configurations_to_run, 1):
            for instance_name in instances:
                file_path = repo_root / "data" / "instances" / "120" / f"{instance_name}.json"

                future = executor.submit(
                    run_single_instance, 
                    instance_name, file_path, active_config, 
                    combo_idx, tuned_factor, output_csv, write_lock
                )
                futures[future] = (combo_idx, instance_name)

        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            combo_idx, instance_name = futures[future]
            try:
                future.result()
                if completed_count % 10 == 0 or completed_count == total_tasks:
                    logger.info(f"Progress Status Update: [{completed_count}/{total_tasks}] total tuning tasks executed.")
            except Exception as e:
                logger.error(f"Task tracking error on configuration {combo_idx} for {instance_name}: {e}")

    try:
        logger.info("\n" + "="*80)
        logger.info("OFAT EXECUTION RUN METRICS SUMMARY")
        logger.info("="*80)

        df = pd.read_csv(output_csv)
        # Group by configuration index to evaluate average optimality gap across tested files
        agg_df = df.groupby('combination_idx').agg({
            'gap_percent': 'mean',
            'tuned_factor': 'first',
            'reaction_factor': 'first',
            'd_beta': 'first',
            'target_acceptance_rate': 'first',
            'initial_threshold': 'first'
        }).reset_index()

        best_idx = agg_df['gap_percent'].idxmin()
        best_row = agg_df.iloc[best_idx]

        logger.info(f"Top Performing Configuration Sweep Index: {best_row['combination_idx']} (Avg Gap: {best_row['gap_percent']:.2f}%)")
        logger.info(f"Triggered by factor adjustments on: '{best_row['tuned_factor']}'")
        logger.info(f"  reaction_factor:        {best_row['reaction_factor']}")
        logger.info(f"  d_beta:                 {best_row['d_beta']}")
        logger.info(f"  target_acceptance_rate: {best_row['target_acceptance_rate']}")
        logger.info(f"  initial_threshold:      {best_row['initial_threshold']}")

    except Exception as analysis_err:
        logger.error(f"Post-run summary analysis step failed: {analysis_err}")

    logger.info(f"\nAll results saved successfully to: {output_csv.resolve()}")


if __name__ == "__main__":
    tune_parameters_async()
