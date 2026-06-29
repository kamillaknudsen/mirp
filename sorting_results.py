import pandas as pd
import io

# 1. Define your exact target instance ordering
target_order = [
    "LR1_DR02_VC01_V6a", "LR1_DR02_VC02_V6a", "LR1_DR02_VC03_V7a", "LR1_DR02_VC03_V8a",
    "LR1_DR02_VC04_V8a", "LR1_DR02_VC05_V8a", "LR1_DR03_VC03_V10b", "LR1_DR03_VC03_V13b",
    "LR1_DR03_VC03_V16a", "LR1_DR04_VC03_V15a", "LR1_DR04_VC03_V15b", "LR1_DR04_VC05_V17a",
    "LR1_DR04_VC05_V17b", "LR1_DR05_VC05_V25a", "LR1_DR05_VC05_V25b"
]

def process_and_minimize_results(results_csv_path: str, bounds_path: str, final_output_path: str):
    # 1. Load and parse the benchmark bounds data (delimited by tabs or spaces)
    df_bounds = pd.read_csv(bounds_path, sep=r'\s+', header=None,
                            names=['instance_name', 'periods', 'lower_bound', 'z_ref'])
    df_bounds['instance_name'] = df_bounds['instance_name'].str.strip()
    df_bounds['periods'] = df_bounds['periods'].astype(int)
    df_bounds['z_ref'] = df_bounds['z_ref'].astype(float)

    # 2. Load and deduplicate your raw algorithm results
    df_results = pd.read_csv(results_csv_path)
    df_results['instance_name'] = df_results['instance_name'].astype(str).str.strip()
    df_results['periods'] = df_results['periods'].astype(int)
    df_results['ils_total_cost'] = df_results['ils_total_cost'].astype(float)
    df_results['total_runtime_sec'] = df_results['total_runtime_sec'].astype(float)

    # Deduplicate results: keep the row with the lowest cost per instance and period
    df_best_results = df_results.sort_values(by='ils_total_cost', ascending=True)\
                                .drop_duplicates(subset=['instance_name', 'periods'], keep='first').copy()

    # 3. Merge results with benchmark dataset on instance name and periods
    df_merged = pd.merge(df_best_results, df_bounds[['instance_name', 'periods', 'z_ref']], 
                         on=['instance_name', 'periods'], how='inner')

    # 4. Calculate the percentage gap using the formal formula: Gap = ((z - z_ref) / z_ref) * 100
    z = df_merged['ils_total_cost']
    z_ref = df_merged['z_ref']
    df_merged['gap_percent'] = round(((z - z_ref) / z_ref) * 100, 2)

    # 5. Convert runtime to hours
    df_merged['runtime_in_hours'] = round(df_merged['total_runtime_sec'] / 3600, 2)

    # 6. Apply custom structural sort order
    df_merged['instance_name'] = pd.Categorical(df_merged['instance_name'], categories=target_order, ordered=True)
    df_final = df_merged.sort_values(by=['periods', 'instance_name']).reset_index(drop=True)

    df_final['objective_value'] = round(df_final['ils_total_cost'], 2)

    df_final['z_ref'] = df_final['z_ref'].map(lambda x: f"{x:.2f}")
    df_final['objective_value'] = df_final['objective_value'].map(lambda x: f"{x:,.2f}")
    df_final['runtime_in_hours'] = df_final['runtime_in_hours'].map(lambda x: f"{x:,.2f}")
    df_final['gap_percent'] = df_final['gap_percent'].map(lambda x: f"{x:.2f}")

    # 7. Select only the necessary comparison columns
    output_columns = ['instance_name', 'periods', 'z_ref', 'objective_value', 'gap_percent', 'runtime_in_hours']
    df_output = df_final[output_columns]

    # Save to your output file
    df_output.to_csv(final_output_path, index=False)
    print(f"Data joined and gaps compiled successfully! Saved to: {final_output_path}")

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Change these filenames to match your local setup
    # REPLICATION
    replication_raw_csv = "/Users/kamillaknudsen-bielaszewska/Documents/EUR/Thesis/code_thesis/data/results_all_together.csv"
    bounds = "/Users/kamillaknudsen-bielaszewska/Documents/EUR/Thesis/code_thesis/data/bounds.txt"
    repli_clean_csv = "/Users/kamillaknudsen-bielaszewska/Documents/EUR/Thesis/code_thesis/data/final_best_sorted_results.csv"
    process_and_minimize_results(replication_raw_csv, bounds, repli_clean_csv)
 
    #EXTENSION
    extension_raw_csv = "/Users/kamillaknudsen-bielaszewska/Documents/EUR/Thesis/code_thesis/data/results_all_together_extension.csv"
    extension_output = '/Users/kamillaknudsen-bielaszewska/Documents/EUR/Thesis/code_thesis/data/final_best_sorted_extension_results.csv'
    process_and_minimize_results(extension_raw_csv, bounds, extension_output)
