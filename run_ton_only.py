import os
import pandas as pd
from model_comparison import (
    load_ton_dataset, run_pipeline_for_dataset, ResourceMonitor,
    plot_cross_dataset_comparison, plot_resource_comparison_all,
    MODEL_CATEGORY
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def main():
    monitor = ResourceMonitor()
    
    # 1. Run TON_IoT
    print("Running TON_IoT dataset...")
    ton_results = run_pipeline_for_dataset('TON_IoT', load_ton_dataset, monitor)
    
    # 2. Re-create the full results dictionary from CSV and TON results
    all_dataset_results = {'TON_IoT': ton_results}
    
    # Load existing CSV to rebuild the structure for the other 2 datasets
    csv_path = os.path.join(RESULTS_DIR, 'all_results.csv')
    df_existing = pd.read_csv(csv_path)
    
    # Add Edge and NSL-KDD
    for ds_name in ['Edge-IIoT', 'NSL-KDD']:
        ds_df = df_existing[df_existing['Dataset'] == ds_name]
        ds_results = {}
        for _, row in ds_df.iterrows():
            model = row['Model']
            ds_results[model] = {
                'metrics': {
                    'accuracy': row['accuracy'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1': row['f1'],
                    'roc_auc': row['roc_auc']
                },
                'resource': {
                    'ram_usage_mb': row['resource_ram_usage_mb'],
                    'cpu_percent': row['resource_cpu_percent'],
                    'inference_time_ms': row['resource_inference_time_ms'],
                    'model_size_kb': row['resource_model_size_kb']
                }
            }
        all_dataset_results[ds_name] = ds_results
        
    # Re-order dict to be logical
    ordered_results = {
        'Edge-IIoT': all_dataset_results['Edge-IIoT'],
        'NSL-KDD': all_dataset_results['NSL-KDD'],
        'TON_IoT': all_dataset_results['TON_IoT']
    }
    
    # 3. Generate cross-dataset charts
    plot_cross_dataset_comparison(ordered_results)
    plot_resource_comparison_all(ordered_results)
    
    # 4. Save merged CSV
    rows = []
    for ds_name, ds_results in ordered_results.items():
        for model_name, res in ds_results.items():
            row = {
                'Dataset': ds_name,
                'Model': model_name,
                'Category': MODEL_CATEGORY.get(model_name, 'Unknown'),
                **res['metrics'],
                **{f'resource_{k}': v for k, v in res['resource'].items()},
            }
            rows.append(row)
    
    df_new = pd.DataFrame(rows)
    df_new.to_csv(csv_path, index=False)
    print(f"Updated all_results.csv with {len(df_new)} rows.")
    print("ALL DONE!")

if __name__ == '__main__':
    main()
