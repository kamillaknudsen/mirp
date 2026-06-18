from pathlib import Path
import time
import gc
import os
import logging
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_loader import load_instance
from models import Call, Solution
from initial_calls import create_empty_solution_with_initial_calls
from beam_search import beam_search
from ils import iterated_local_search

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
# WORKER FUNCTION (Executes on an isolated CPU core)
# ----------------------------------------------------
def process_single_instance(file_path: Path, config: dict):
    try:
        logger.info(f"Processing single file: {file_path.name}")
        instance = load_instance(file_path)
        initial_solution = create_empty_solution_with_initial_calls(instance)

        start_time_bs = time.time()
        bs_solution = beam_search(instance, initial_solution, config['N'], config['q'], config['w'], config['std_deviation'])
        runtime_bs = time.time() - start_time_bs

        remaining_runtime = config['instance_max_runtime'] - runtime_bs
        remaining_runtime = max(1, remaining_runtime)

        start_time_ils = time.time()
        final_solution = iterated_local_search(instance, bs_solution, config['max_iterations'], config['max_non_improving'], config['initial_temp'], config['cooling_rate'], remaining_runtime)
        runtime_ils = time.time() - start_time_ils
        
        total_runtime = runtime_bs + runtime_ils
        improvement = ((bs_solution.total_cost - final_solution.total_cost) / bs_solution.total_cost * 100) if bs_solution.total_cost > 0 else 0.0

        row_payload = [
            instance.name, instance.metadata.n_periods,
            bs_solution.total_cost, bs_solution.routing_cost, bs_solution.inventory_penalty, bs_solution.is_feasible, round(runtime_bs, 4), len(bs_solution.calls),
            final_solution.total_cost, final_solution.routing_cost, final_solution.inventory_penalty, final_solution.is_feasible, round(runtime_ils, 4), len(final_solution.calls),
            round(total_runtime, 4), format_runtime(total_runtime), f"{improvement:.2f}%"
        ]

        # Clear memory variables to stay within Kaggle limits
        del instance, initial_solution, bs_solution, final_solution
        gc.collect()

        return {
            "status": "success",
            "file_name": file_path.name,
            "improvement": f"{improvement:.2f}",
            "total_runtime_str": format_runtime(total_runtime),
            "row_data": row_payload
        }

    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"CRASHED on {file_path.name}: {error_details}")
        return {"status": "error", "file_name": file_path.name, "error_msg": error_details}

def run_horizon_batch(horizon: int, base_dir: Path, config: dict):
    """Executes experiments for a specific time horizon batch."""
    logger.info(f"\n==================================================")
    logger.info(f"STARTING BATCH RUN FOR HORIZON: {horizon}")
    logger.info(f"==================================================")

    # Repository reference routing (avoids hardcoded local Mac paths)
    instances_dir = base_dir / "data" / "instances" / str(horizon)
    output_dir = base_dir / "results" / f"horizon_{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "results_summary.csv"

    instance_files = sorted(list(instances_dir.glob('*.json')))
    if not instance_files:
        logger.warning(f"Skipping horizon {horizon}: No data models found at {instances_dir}")
        return

    if not output_csv.exists():
        csv_headers = [
            "instance_name", "periods",
            "bs_total_cost", "bs_routing_cost", "bs_inventory_penalty", "bs_feasible", "bs_runtime_sec", "bs_calls",
            "ils_total_cost", "ils_routing_cost", "ils_inventory_penalty", "ils_feasible", "ils_runtime_sec", "ils_calls",
            "total_runtime_sec", "total_runtime_formatted", "improvement_percent"
        ]
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    start_batch_time = time.time()
    completed_count = 0

    with ProcessPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_single_instance, file_path, config): file_path 
            for file_path in instance_files
        }
        
        for future in as_completed(future_to_file):
            completed_count += 1
            file_path = future_to_file[future]
            
            try:
                result = future.result()
                if result["status"] == "success":
                    with open(output_csv, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(result["row_data"])
                    logger.info(
                        f"[{completed_count}/{len(instance_files)}] Horizon {horizon} | Unified {result['file_name']} "
                        f"| Improvement: {result['improvement']}% | Time: {result['total_runtime_str']}"
                    )
                    try:
                        logger.info(f"Syncing data row for {result['file_name']} to GitHub...")
                        os.system("git config --global user.name 'Kaggle Automation Framework'")
                        os.system("git config --global user.email 'actions@github.com'")
                        os.system("git pull origin main --rebase")
                        os.system(f"git add {output_csv}")
                        os.system(f"git commit -m 'Automated sync: completed {result['file_name']} [skip ci]'")
                        os.system("git push origin main")
                    except Exception as git_err:
                        logger.error(f"Git sync skipped due to network block: {str(git_err)}")

                else:
                    logger.error(f"Error on {result['file_name']}: {result['error_msg']}")
            except Exception as exc:
                logger.error(f"Worker crashed for file {file_path.name}: {exc}")

    total_batch_time = time.time() - start_batch_time
    logger.info(f"Horizon {horizon} metrics compiled in: {format_runtime(total_batch_time)}")

def run():
    # Resolve the repository root relative to where this running script lives
    repo_root = Path(__file__).resolve().parent.parent

    config = {
        'N': 1000, 'q': 3, 'w': 2, 'std_deviation': 1,
        'max_iterations': 640, 'max_non_improving': 4,
        'initial_temp': 10000, 'cooling_rate': 0.995,
        "instance_max_runtime": 90000
    }

    # Loops through folders sequentially 
    time_horizons = [360, 180, 120]
    for horizon in time_horizons:
        run_horizon_batch(horizon, repo_root, config)

if __name__ == "__main__":
    run()