import pandas as pd

def inspect_parquet(path):
    print(f"--- Inspecting {path} ---")
    try:
        df = pd.read_parquet(path)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("Head:")
        print(df.head())
    except Exception as e:
        print(f"Error reading {path}: {e}")

inspect_parquet("/Users/mac/demo/ares-platform/data/stack_the_cups_lerobot_v3/meta/tasks.parquet")
inspect_parquet("/Users/mac/demo/ares-platform/data/stack_the_cups_lerobot_v3/data/chunk-000/file-000.parquet")
