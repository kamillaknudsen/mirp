from pathlib import Path
import time
import gc
import os
import logging
import csv
from data_loader import load_instance
from initial_calls import create_empty_solution_with_initial_calls
from beam_search import beam_search
from ails import adaptive_iterated_local_search
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_runtime(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.4f}s"
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {rem_seconds:.2f}s"
    hours = int(minutes // 60)
    rem_minutes = minutes % 60
    return f"{hours}h {rem_minutes}m {rem_seconds:.2f}s"

# ----------------------------------------------------
# SEQUENTIAL WORKER FUNCTION
# ----------------------------------------------------
def process_single_instance(file_path: Path, output_csv: Path, config: dict, write_lock):
    try:
        logger.info(f"Processing file: {file_path.name}")
        instance = load_instance(file_path)
        initial_solution = create_empty_solution_with_initial_calls(instance)

        # 1. Run Beam Search
        start_time_bs = time.time()
        bs_solution = beam_search(instance, initial_solution, config['N'], config['q'], config['w'], config['std_deviation'])
        runtime_bs = time.time() - start_time_bs

        # Deduct Beam Search runtime from the total allocated instance budget
        remaining_runtime = config['instance_max_runtime'] - runtime_bs
        remaining_runtime = max(1.0, remaining_runtime)

        # 2. Run Iterated Local Search
        start_time_ils = time.time()
        final_solution = adaptive_iterated_local_search(instance, bs_solution, config['max_iterations'], 
                                                        config['max_non_improving'], remaining_runtime,
                                                        config['sigma1'], config['sigma2'],
                                                        config['sigma3'], config['segment_size'],
                                                        config['reaction_factor'],  config['gamma'],
                                                        config['d_beta'], config['target_acceptance_rate'],
                                                        config['initial_threshold'], config['epsilon'],
                                                        )
        runtime_ils = time.time() - start_time_ils
        
        total_runtime = runtime_bs + runtime_ils
        improvement = ((bs_solution.total_cost - final_solution.total_cost) / bs_solution.total_cost * 100) if bs_solution.total_cost > 0 else 0.0

        row_payload = [
            instance.name, instance.metadata.n_periods,
            bs_solution.total_cost, bs_solution.routing_cost, bs_solution.inventory_penalty, bs_solution.is_feasible, round(runtime_bs, 4), len(bs_solution.calls),
            final_solution.total_cost, final_solution.routing_cost, final_solution.inventory_penalty, final_solution.is_feasible, round(runtime_ils, 4), len(final_solution.calls),
            round(total_runtime, 4), format_runtime(total_runtime), f"{improvement:.2f}%"
        ]

        with write_lock:
            with open(output_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_payload)
                f.flush()

            logger.info(f"SUCCESS | {file_path.name} | Improvement: {improvement:.2f}% | Time: {format_runtime(total_runtime)}")
            
            # 3. AUTOMATIC GIT PUSH BLOCK
            try:
                logger.info(f"Syncing data row for {file_path.name} to GitHub...")
                os.system("git config --global user.name 'Kaggle Automation Framework'")
                os.system("git config --global user.email 'actions@github.com'")
                os.system("git pull origin main --rebase")
                os.system(f"git add {output_csv}")
                os.system(f"git commit -m 'Automated sync: completed {file_path.name} [skip ci]'")
                os.system("git push origin main")
                logger.info("GitHub synchronization successful!")
            except Exception as git_err:
                logger.error(f"Git push failed but continuing execution queue: {str(git_err)}")

        # Strict memory cleanup to avoid RAM leaks
        del instance, initial_solution, bs_solution, final_solution
        gc.collect()

    except Exception as e:
        import traceback
        logger.error(f"CRASHED on {file_path.name}: {str(e)}\n{traceback.format_exc()}")

def run_horizon_batch(horizon: int, base_dir: Path, config: dict, write_lock, max_workers: int): 
    """Executes experiments ONE BY ONE sequentially to safeguard memory usage."""
    logger.info(f"\n==================================================")
    logger.info(f"STARTING BATCH RUN FOR HORIZON: {horizon}")
    logger.info(f"==================================================")

    instances_dir = base_dir / "data" / "instances" / str(horizon)
    output_dir = base_dir / "results" / f"horizon_{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "extension_results_correct.csv"

    instance_files = sorted(list(instances_dir.glob('*.json')))
    if not instance_files:
        logger.warning(f"Skipping horizon {horizon}: No data models found at {instances_dir}")
        return

    completed_instances = set()
    if output_csv.exists():
        try:
            with open(output_csv, mode='r', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader, None)  # Skip the header row
                if headers is not None:
                    for row in reader:
                        if row:  # Ensure the row isn't blank
                            # Strip .json if extension exists, or store raw instance name
                            completed_instances.add(row[0].strip())
            logger.info(f"Found existing results file. Detected {len(completed_instances)} already completed instances for Horizon {horizon}.")
        except Exception as read_err:
            logger.error(f"Could not parse existing CSV file for recovery check (will proceed without skipping): {read_err}")
    else:
        # Create file and write headers if it does not exist yet
        csv_headers = [
            "instance_name", "periods",
            "bs_total_cost", "bs_routing_cost", "bs_inventory_penalty", "bs_feasible", "bs_runtime_sec", "bs_calls",
            "ils_total_cost", "ils_routing_cost", "ils_inventory_penalty", "ils_feasible", "ils_runtime_sec", "ils_calls",
            "total_runtime_sec", "total_runtime_formatted", "improvement_percent"
        ]
        with open(output_csv, mode='w', newline='') as f:
            csv.writer(f).writerow(csv_headers)

    runnable_files = [
        f for f in instance_files 
        if f.stem not in completed_instances and f.name not in completed_instances
    ]

    skipped_count = len(instance_files) - len(runnable_files)
    if skipped_count > 0:
        logger.info(f"Filtered execution queue: Skipping {skipped_count} completed tasks. Remaining payload: {len(runnable_files)} jobs.")

    if not runnable_files:
        logger.info(f"All instances for Horizon {horizon} are already processed. Moving to next horizon batch.")
        return

    start_batch_time = time.time()

    logger.info(f"Spawning pool with up to {max_workers} worker processes...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_instance, file_path, output_csv, config, write_lock): file_path 
            for file_path in runnable_files
        }
        
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error(f"Worker generated an unhandled exception for file {file_path.name}: {exc}")

    total_batch_time = time.time() - start_batch_time
    logger.info(f"Horizon {horizon} metrics compiled sequentially in: {format_runtime(total_batch_time)}")

def run():
    repo_root = Path(__file__).resolve().parent.parent

    config = {
        'N': 1000,
        'q': 3,
        'w': 2,
        'std_deviation': 1,
        'max_iterations': 640,
        'max_non_improving': 50,
        "instance_max_runtime": 43100,
        'sigma1': 5,
        'sigma2': 2,
        'sigma3': 1,
        'segment_size': 100,
        'reaction_factor': 0.1,
        'gamma': 20,
        'd_beta': 0.3,
        "target_acceptance_rate": 0.15,
        "initial_threshold": 0.05,
        "epsilon": 0.001
    }
    available_cores = os.cpu_count() or 2
    logger.info(f"System detected {available_cores} CPU cores available.")

    # Initialize a multiprocessing-safe lock to synchronize disk write actions
    manager = multiprocessing.Manager()
    write_lock = manager.Lock()

    time_horizons = [120, 180, 360]
    for horizon in time_horizons:
        run_horizon_batch(horizon, repo_root, config, write_lock, max_workers=available_cores)

if __name__ == "__main__":
    run()
