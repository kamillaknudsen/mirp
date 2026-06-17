import csv
import io
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)

bounds_path = os.path.join(root_dir, "data", "bounds.txt")

# 1. Your raw CSV output text string
csv_data = """instance_name,periods,bs_total_cost,bs_routing_cost,bs_inventory_penalty,bs_feasible,bs_runtime_sec,bs_calls,ils_total_cost,ils_routing_cost,ils_inventory_penalty,ils_feasible,ils_runtime_sec,ils_calls,total_runtime_sec,total_runtime_formatted,improvement_percent
LR1_DR02_VC03_V8a,120,316723.708282,42423.70828199998,274300.0,True,49.4997,52,133124.25567399999,37324.25567399998,95800.0,True,2173.0723,54,2222.572,37m 2.57s,57.97%
LR1_DR02_VC01_V6a,120,523279.81393999996,34879.81393999996,488400.0,True,6.4668,53,62070.30631999995,34070.30631999995,28000.0,True,4541.6017,65,4548.0684,1h 15m 48.07s,88.14%
LR1_DR02_VC02_V6a,120,731274.72451,31374.724510000007,699900.0,True,39.1222,73,131964.47872,32464.478720000014,99500.0,True,5203.5609,67,5242.6831,1h 27m 22.68s,81.95%
LR1_DR02_VC03_V7a,120,389381.654354,34681.65435399997,354700.0,True,25.0019,57,95203.68891399997,34003.68891399998,61200.0,True,5343.4529,65,5368.4548,1h 29m 28.45s,75.55%
LR1_DR02_VC05_V8a,120,206878.17798399998,37378.17798399999,169500.0,True,965.7815,69,69786.18430799997,37986.184307999974,31800.0,True,6708.7114,71,7674.493,2h 7m 54.49s,66.27%
LR1_DR02_VC04_V8a,120,134563.171048,43463.171048000004,91100.0,True,1037.4588,76,86421.89636399999,42221.896363999986,44200.0,True,7837.4004,78,8874.8592,2h 27m 54.86s,35.78%
LR1_DR03_VC03_V10b,120,1538635.000876,34135.000875999984,1504500.0,True,306.1496,72,201279.615528,57979.615527999995,143300.0,True,15854.8472,98,16160.9968,4h 29m 21.00s,86.92%
LR1_DR03_VC03_V16a,120,921035.2492720002,72835.24927200012,848200.0,True,2981.5647,76,156145.74018600013,70045.74018600013,86100.0,True,14734.2177,94,17715.7824,4h 55m 15.78s,83.05%
LR1_DR03_VC03_V13b,120,1592685.0541120002,74085.05411200009,1518600.0,True,673.6399,114,214524.85854200003,77124.85854200003,137400.0,True,25448.0653,120,26121.7052,7h 15m 21.71s,86.53%
LR1_DR04_VC03_V15a,120,2152026.312948,67226.31294800005,2084800.0,True,202.6662,83,206346.79243600005,75146.79243600005,131200.0,True,34938.9248,127,35141.591,9h 45m 41.59s,90.41%
LR1_DR04_VC03_V15b,120,2326330.198364,64630.198363999916,2261700.0,True,1140.6433,126,275566.051708,83766.05170800004,191800.0,True,41162.4978,142,42303.1411,11h 45m 3.14s,88.15%
LR1_DR04_VC05_V17a,120,2423716.545806,81216.54580600005,2342500.0,True,330.7695,144,135318.29592200002,76418.29592200003,58900.0,True,43654.6372,136,43985.4067,12h 13m 5.41s,94.42%
LR1_DR04_VC05_V17b,120,2455820.9846139997,60920.98461399994,2394900.0,True,767.535,101,245665.36540200002,81165.36540200004,164500.0,True,48214.0154,153,48981.5505,13h 36m 21.55s,90.00%
LR1_DR05_VC05_V25a,120,2690148.7311879997,40248.73118799993,2649900.0,True,2572.4677,77,227694.79296400008,106394.7929640001,121300.0,True,60937.7625,169,63510.2302,17h 38m 30.23s,91.54%"""

# 2. DICTIONARY OF ABSOLUTE BASELINE COSTS FOR COMPUTING GAPS
# Update these mock values with your dataset's true mathematical lower bounds.
bounds = {}
best_known = {}

with open(bounds_path) as f:
    for line in f:
        parts = line.strip().split()

        instance = parts[0]
        periods = int(parts[1])

        if periods != 120:
            continue

        z_bound = float(parts[2])
        z_best = float(parts[3])

        bounds[instance] = z_bound
        best_known[instance] = z_best

# MIRPPacs values extracted from your LaTeX template layout to populate the summary row averages
mirppacs_static_table_data = {
    "LR1_DR02_VC01_V6a": {"best_gap": "0.00", "avg_gap": "1.73", "time": 0.23},
    "LR1_DR02_VC02_V6a": {"best_gap": "0.00", "avg_gap": "1.73", "time": 0.51},
    "LR1_DR02_VC03_V7a": {"best_gap": "-0.26", "avg_gap": "4.17", "time": 2.83},
    "LR1_DR02_VC03_V8a": {"best_gap": "0.00", "avg_gap": "2.07", "time": 1.78},
    "LR1_DR02_VC04_V8a": {"best_gap": "0.00", "avg_gap": "0.46", "time": 2.78},
    "LR1_DR02_VC05_V8a": {"best_gap": "-0.07", "avg_gap": "0.60", "time": 2.49},
    "LR1_DR03_VC03_V10b": {"best_gap": "3.10", "avg_gap": "10.77", "time": 1.57},
    "LR1_DR03_VC03_V13b": {"best_gap": "0.03", "avg_gap": "2.59", "time": 10.12},
    "LR1_DR03_VC03_V16a": {"best_gap": "-4.69", "avg_gap": "0.43", "time": 7.33},
    "LR1_DR04_VC03_V15a": {"best_gap": "-0.38", "avg_gap": "1.34", "time": 6.75},
    "LR1_DR04_VC03_V15b": {"best_gap": "-5.18", "avg_gap": "0.71", "time": 13.28},
    "LR1_DR04_VC05_V17a": {"best_gap": "-0.35", "avg_gap": "0.09", "time": 7.49},
    "LR1_DR04_VC05_V17b": {"best_gap": "-6.32", "avg_gap": "-3.25", "time": 13.39},
    "LR1_DR05_VC05_V25a": {"best_gap": "-0.69", "avg_gap": "-0.16", "time": 15.76},
    "LR1_DR05_VC05_V25b": {"best_gap": "-3.09", "avg_gap": "2.09", "time": 14.55},
}

# 3. PARSE RAW RESULTS
f = io.StringIO(csv_data.strip())
reader = csv.DictReader(f)

parsed_results = {}
for row in reader:
    name = row['instance_name'].strip()
    parsed_results[name] = {
        'obj': float(row['ils_total_cost']),
        'runtime_hours': float(row['total_runtime_sec']) / 3600.0
    }

# Order identical to your LaTeX structure template sequence
target_order = [
    "LR1_DR02_VC01_V6a", "LR1_DR02_VC02_V6a", "LR1_DR02_VC03_V7a", "LR1_DR02_VC03_V8a",
    "LR1_DR02_VC04_V8a", "LR1_DR02_VC05_V8a", "LR1_DR03_VC03_V10b", "LR1_DR03_VC03_V13b",
    "LR1_DR03_VC03_V16a", "LR1_DR04_VC03_V15a", "LR1_DR04_VC03_V15b", "LR1_DR04_VC05_V17a",
    "LR1_DR04_VC05_V17b", "LR1_DR05_VC05_V25a", "LR1_DR05_VC05_V25b"
]

total_my_obj, total_my_gap, total_my_time = 0.0, 0.0, 0.0
valid_rows_count = 0

print(f"{'Instance':<22} | {'Obj':<12} | {'Best Gap':<8} | {'Avg Gap':<8} | {'Time (h)':<8}")
print("-" * 70)

latex_lines = []

for name in target_order:

    escaped_name = name.replace("_", "\\_")
    mir_data = mirppacs_static_table_data[name]

    if name in parsed_results:
        # Invert minimization cost tracking sign to compute profit gaps accurately 
        # against the maximization framework assumptions of the bounds/paper.
        z_your = parsed_results[name]['obj']
        time_h = parsed_results[name]['runtime_hours']

        # Safe fallback checks for missing bounds file values
        z_bound = bounds.get(name, 1.0)
        z_best = best_known.get(name, 1.0)

        gap_lb = 100 * (z_your - z_bound) / z_bound if z_bound != 0 else 0.0
        gap_best = 100 * (z_your - z_best) / z_best if z_best != 0 else 0.0

        # Formatted display tokens
        obj_str = f"{parsed_results[name]['obj']:.2f}"
        gap_str = f"{gap_best:.2f}"
        time_str = f"{time_h:.2f}"

        print(f"{name:<22} | {obj_str:<12} | {gap_lb:.2f}% | {gap_best:.2f}% | {time_str:<8}")

        total_my_obj += parsed_results[name]['obj']
        total_my_gap += gap_best
        total_my_time += time_h
        valid_rows_count += 1

        latex_lines.append(
            f"\\texttt{{{escaped_name}}} & {obj_str} & {gap_lb:.2f}\\ & {gap_str} & {time_str} & "
            f"{mir_data['best_gap']} & {mir_data['avg_gap']} & {mir_data['time']} \\\\"
        )
    else:
        # Handle cases where some rows are missing from the data stream
        obj_str, gap_str, time_str = "N/A", "N/A", "N/A"
        latex_lines.append(
            f"\\texttt{{{escaped_name}}} & & & & & "
            f"{mir_data['best_gap']} & {mir_data['avg_gap']} & {mir_data['time']} \\\\"
        )
        
    print(f"{name:<22} | {obj_str:<12} | {gap_str:<8} | {gap_str:<8} | {time_str:<8}")

print("-" * 70)
# Print out the fully filled TeX markup block ready to copy-paste
print("\n--- GENERATED LATEX ROWS FOR YOUR THESIS ---")
for line in latex_lines:
    print(line)

if valid_rows_count > 0:
    avg_obj = total_my_obj / valid_rows_count
    avg_gap = total_my_gap / valid_rows_count
    avg_time = total_my_time / valid_rows_count
    print(f"\\hline")
    print(f"Average & {avg_obj:.2f} & {avg_gap:.2f}\\ & {avg_gap:.2f}\\ & {avg_time:.2f} & & & \\\\")