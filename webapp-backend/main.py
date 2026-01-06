"""
FastAPI backend for the Ares platform web application.
Provides REST API endpoints for all data processing and visualization functionality.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import traceback

import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

load_dotenv()

# Add project root to path for scripts import
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

app = FastAPI(title="Ares Platform API", version="1.0.0")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Backend is running"}

@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint to verify backend is working."""
    return {"status": "ok", "message": "Backend API is working"}

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configuration
POD_API_URL = "https://g6hxoyusgab5l4-8000.proxy.runpod.net/run-json"
S3_BUCKET = "6d-temp-storage"
S3_REGION = "us-west-2"
s3 = boto3.client("s3", region_name=S3_REGION)
tmp_dump_dir = os.path.join(os.path.expanduser("~"), ".ares_webapp_tmp")
os.makedirs(tmp_dump_dir, exist_ok=True)

# Store dataset paths in memory (dataset_name -> dataset_path)
dataset_paths: Dict[str, str] = {}

# Request/Response models
class DatasetLoadRequest(BaseModel):
    dataset_path: str
    dataset_name: str

class AugmentationRequest(BaseModel):
    dataset_name: str
    prompt: str
    task_description: Optional[str] = ""

class OptimizationRequest(BaseModel):
    dataset_name: str

class TestingUploadRequest(BaseModel):
    dataset_name: str
    test_directory: str

class DatasetInfo(BaseModel):
    total_episodes: int
    robot_type: str
    dataset_name: str

# Helper functions
def upload_folder_to_s3(local_folder_path: str, bucket_name: str, s3_prefix: str = "") -> None:
    """Upload all files in a folder to S3."""
    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            try:
                s3.upload_file(local_file_path, bucket_name, s3_key)
            except Exception as e:
                print(f"Error uploading {local_file_path}: {e}")

def load_dataset_info(dataset_path: str) -> Optional[Dict[str, Any]]:
    """Load dataset info from meta/info.json."""
    try:
        info_path = Path(dataset_path) / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def count_episodes_from_parquet(dataset_path: str) -> int:
    """Count total episodes from parquet files in the dataset."""
    try:
        data_path = Path(dataset_path) / "data"
        if not data_path.exists():
            return 0
        
        all_episode_indices = set()
        for chunk_path in data_path.glob("chunk-*"):
            if chunk_path.is_dir():
                for parquet_file in chunk_path.glob("*.parquet"):
                    try:
                        df = pd.read_parquet(parquet_file)
                        if 'episode_index' in df.columns:
                            # Get unique episode indices from this file
                            unique_episodes = df['episode_index'].unique()
                            all_episode_indices.update(unique_episodes)
                    except Exception as e:
                        print(f"Error reading {parquet_file}: {e}")
                        continue
        
        return len(all_episode_indices)
    except Exception as e:
        print(f"Error counting episodes: {e}")
        return 0

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Ares Platform API", "version": "1.0.0"}

@app.post("/api/datasets/load")
async def load_dataset(request: DatasetLoadRequest):
    """Load a dataset from a local path."""
    try:
        dataset_path = Path(request.dataset_path)
        if not dataset_path.exists():
            raise HTTPException(status_code=400, detail=f"Path does not exist: {request.dataset_path}")
        
        if not (dataset_path / "data").exists() or not (dataset_path / "meta").exists():
            raise HTTPException(
                status_code=400,
                detail="Invalid LeRobot dataset structure. Directory must contain 'data/' and 'meta/' folders."
            )
        
        # Run basic ingestion (no database creation)
        # Ensure project root is in path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from scripts.ingest_lerobot_dataset import ingest_dataset
        # Using None for engine_url since database creation is skipped
        count = ingest_dataset(str(dataset_path), None, request.dataset_name)
        
        # Store the dataset path for later use
        dataset_paths[request.dataset_name] = str(dataset_path)
        
        return {
            "success": True,
            "dataset_name": request.dataset_name,
            "dataset_path": str(dataset_path),
            "episodes_ingested": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

@app.get("/api/datasets/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    """Get dataset information and metrics."""
    try:
        # Check if dataset path is stored
        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        dataset_path = dataset_paths[dataset_name]
        dataset_info = load_dataset_info(dataset_path)
        
        # Get episode count from info.json first, fallback to counting parquet files
        if dataset_info and 'total_episodes' in dataset_info:
            total_episodes = dataset_info['total_episodes']
        else:
            # Only count from parquet if not in info.json
            total_episodes = count_episodes_from_parquet(dataset_path)
        
        robot_type = "N/A"
        if dataset_info and 'robot_type' in dataset_info:
            robot_type = dataset_info['robot_type']
        
        return {
            "dataset_name": dataset_name,
            "total_episodes": total_episodes,
            "robot_type": robot_type
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset info: {str(e)}")

@app.get("/api/datasets/{dataset_name}/data")
async def get_dataset_data(dataset_name: str):
    """Get dataset data as JSON."""
    try:
        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        # Read from parquet files
        dataset_path = dataset_paths[dataset_name]
        data_path = Path(dataset_path) / "data"
        
        # Get video directory path
        from ares.constants import ARES_VIDEO_DIR
        video_base_dir = Path(ARES_VIDEO_DIR) / dataset_name
        
        episodes = []
        if data_path.exists():
            for chunk_path in data_path.glob("chunk-*"):
                if chunk_path.is_dir():
                    for parquet_file in chunk_path.glob("*.parquet"):
                        try:
                            df = pd.read_parquet(parquet_file)
                            if 'episode_index' in df.columns:
                                for episode_idx in df['episode_index'].unique():
                                    episode_df = df[df['episode_index'] == episode_idx]
                                    
                                    # Check if video exists
                                    video_path = video_base_dir / f"episode_{episode_idx}.mp4"
                                    video_url = None
                                    if video_path.exists():
                                        video_url = f"/api/datasets/{dataset_name}/videos/{episode_idx}"
                                    
                                    # Get task instruction if available
                                    task_instruction = None
                                    if 'task_language_instruction' in episode_df.columns:
                                        task_instruction = episode_df['task_language_instruction'].iloc[0] if len(episode_df) > 0 else None
                                    
                                    episodes.append({
                                        "id": f"episode_{episode_idx}",
                                        "episode_index": int(episode_idx),
                                        "length": len(episode_df),
                                        "video_url": video_url,
                                        "task_language_instruction": task_instruction
                                    })
                        except Exception:
                            continue
        
        return {
            "dataset_name": dataset_name,
            "data": episodes,
            "count": len(episodes)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset data: {str(e)}")

@app.get("/api/datasets/{dataset_name}/distributions")
async def get_dataset_distributions(dataset_name: str):
    """Get dataset distribution visualizations."""
    try:
        # Ares database access removed - return placeholder data
        # The actual visualization generation is done client-side
        return {
            "dataset_name": dataset_name,
            "distributions": {
                "environment": {
                    "type": "scatter",
                    "data": []
                },
                "trajectory": {
                    "type": "line",
                    "data": []
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting distributions: {str(e)}")

@app.post("/api/augmentation/run")
async def run_augmentation(request: AugmentationRequest):
    """Run Cosmos augmentation on a dataset."""
    try:
        # Ares database access removed
        # Dataset path would need to be stored separately or passed in the request
        # For now, we'll just handle the prompt upload and pod API call
        
        # Use task_description if provided, otherwise use prompt (they are equivalent)
        prompt_content = request.task_description if request.task_description else request.prompt
        
        # Create prompt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt_content)
            prompt_path = f.name
        
        try:
            # Upload prompt to S3
            s3.upload_file(
                prompt_path,
                S3_BUCKET,
                f"input_data/{request.dataset_name}/prompt.txt"
            )
            
            # Call pod API
            response = requests.post(
                POD_API_URL,
                json={"dataset_name": request.dataset_name},
                timeout=3600
            )
            response.raise_for_status()
            
            result = response.json()
            s3_output_url = result.get("s3_url")
            
            if not s3_output_url:
                raise RuntimeError("Pod did not return an S3 URL")
            
            return {
                "success": True,
                "s3_url": s3_output_url,
                "dataset_name": request.dataset_name
            }
        finally:
            # Clean up temp file
            if os.path.exists(prompt_path):
                os.unlink(prompt_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running augmentation: {str(e)}")

@app.post("/api/optimization/run")
async def run_optimization(request: OptimizationRequest):
    """Run optimization on a dataset."""
    try:
        # Check if dataset exists
        if request.dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_name}' not found")
        
        # Placeholder for optimization logic
        # In a real implementation, this would run optimization algorithms
        # on the dataset and return results
        
        return {
            "success": True,
            "result": f"Optimization completed for dataset '{request.dataset_name}'",
            "dataset_name": request.dataset_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running optimization: {str(e)}")

@app.get("/api/datasets/{dataset_name}/videos/{episode_index}")
async def get_episode_video(dataset_name: str, episode_index: int):
    """Serve episode video file."""
    try:
        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        from ares.constants import ARES_VIDEO_DIR
        video_path = Path(ARES_VIDEO_DIR) / dataset_name / f"episode_{episode_index}.mp4"
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found for episode {episode_index}")
        
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=f"episode_{episode_index}.mp4"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving video: {str(e)}")

@app.post("/api/testing/upload")
async def upload_test_data(request: TestingUploadRequest):
    """Upload test data directory for a dataset."""
    try:
        if request.dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_name}' not found")
        
        test_dir = Path(request.test_directory)
        if not test_dir.exists():
            raise HTTPException(status_code=400, detail=f"Test directory does not exist: {request.test_directory}")
        
        if not test_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.test_directory}")
        
        # Placeholder: In a real implementation, you would process/upload the test data
        # For now, just validate and return success
        
        return {
            "success": True,
            "dataset_name": request.dataset_name,
            "test_directory": str(test_dir),
            "message": "Test data directory validated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading test data: {str(e)}")

@app.get("/api/datasets/{dataset_name}/export")
async def export_dataset(dataset_name: str, format: str = "csv"):
    """Export dataset in the specified format."""
    try:
        # Ares database access removed - return empty export
        # In a real implementation, you'd read from dataset files directly
        if format == "csv":
            csv_content = ""  # Empty CSV since no database
            return JSONResponse(
                content={"content": csv_content, "filename": f"{dataset_name}.csv"},
                media_type="application/json"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting dataset: {str(e)}")

# ARES Dashboard API Endpoints (replacing Streamlit functionality)
@app.post("/api/ares/initialize")
async def initialize_ares():
    """Initialize ARES data (replaces Streamlit's initialize_data)."""
    try:
        import traceback
        print("[DEBUG] Starting initialize_ares endpoint...")
        from ares_api import initialize_ares_data
        print("[DEBUG] Calling initialize_ares_data...")
        df = initialize_ares_data()
        print(f"[DEBUG] initialize_ares_data completed, DataFrame shape: {df.shape}")
        
        # Convert columns to list of strings to ensure JSON serialization
        try:
            columns_list = [str(col) for col in df.columns]
            print(f"[DEBUG] Converted {len(columns_list)} columns to strings")
        except Exception as col_error:
            print(f"[ERROR] Failed to convert columns: {col_error}")
            columns_list = []
        
        result = {
            "success": True,
            "total_rows": int(len(df)),  # Ensure it's a Python int, not numpy int
            "columns": columns_list,
        }
        print(f"[DEBUG] Returning result with {result['total_rows']} rows and {len(result['columns'])} columns")
        
        # Test JSON serialization before returning
        try:
            import json
            json.dumps(result)
            print("[DEBUG] Result is JSON serializable")
        except Exception as json_error:
            print(f"[ERROR] Result is not JSON serializable: {json_error}")
            raise ValueError(f"Response is not JSON serializable: {json_error}")
        
        return result
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_detail = f"Error initializing ARES: {type(e).__name__}: {str(e)}\n\nTraceback:\n{error_trace}"
        print(f"[ERROR] {error_detail}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        print(f"[ERROR] Exception args: {e.args if hasattr(e, 'args') else 'N/A'}")
        # Ensure we return JSON with proper error format
        try:
            return JSONResponse(
                status_code=500,
                content={"detail": error_detail, "error_type": type(e).__name__, "error_message": str(e)}
            )
        except Exception as json_error:
            print(f"[ERROR] Failed to create JSONResponse: {json_error}")
            # Fallback - use HTTPException
            raise HTTPException(status_code=500, detail=f"Error initializing ARES: {str(e)}")

@app.get("/api/ares/state")
async def get_ares_state():
    """Get ARES state information."""
    try:
        from ares_api import get_state_info
        return get_state_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting state: {str(e)}")

@app.get("/api/ares/filters/metadata")
async def get_filter_metadata():
    """Get metadata for building filter UI (columns, types, ranges, unique values)."""
    try:
        from ares_api import get_dataframe
        from ares.app.data_analysis import infer_visualization_type
        from ares.constants import IGNORE_COLS
        import pandas as pd
        
        df = get_dataframe()
        metadata = []
        
        for col in df.columns:
            if col.lower() in IGNORE_COLS:
                continue
                
            viz_info = infer_visualization_type(col, df)
            if viz_info["viz_type"] is None:
                continue
            
            col_metadata = {
                "column": col,
                "viz_type": viz_info["viz_type"],
                "dtype": str(df[col].dtype),
                "nunique": int(viz_info["nunique"]),
            }
            
            # Add numeric range info
            if viz_info["viz_type"] in ["histogram", "bar"] and pd.api.types.is_numeric_dtype(df[col]):
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                valid_values = numeric_col.dropna()
                if len(valid_values) > 0:
                    min_val = valid_values.min()
                    max_val = valid_values.max()
                    # Handle infinity and NaN values - convert to None for JSON
                    if pd.notna(min_val) and not np.isinf(min_val):
                        col_metadata["min"] = float(min_val)
                    else:
                        col_metadata["min"] = None
                    if pd.notna(max_val) and not np.isinf(max_val):
                        col_metadata["max"] = float(max_val)
                    else:
                        col_metadata["max"] = None
                else:
                    col_metadata["min"] = None
                    col_metadata["max"] = None
                col_metadata["has_nan"] = bool(df[col].isna().any())
            
            # Add categorical options
            if viz_info["viz_type"] == "bar":
                unique_vals = df[col].unique()
                # Limit to 25 options for UI
                if len(unique_vals) <= 25:
                    try:
                        # Try to sort, but handle mixed types
                        sorted_vals = sorted([v for v in unique_vals if pd.notna(v)], key=lambda x: str(x))
                        col_metadata["options"] = [str(v) if pd.notna(v) else "(None)" for v in sorted_vals]
                        # Add None option if it exists
                        if df[col].isna().any():
                            col_metadata["options"].append("(None)")
                    except Exception as e:
                        print(f"[WARNING] Could not sort options for {col}: {e}")
                        col_metadata["options"] = [str(v) if pd.notna(v) else "(None)" for v in unique_vals[:25]]
                else:
                    col_metadata["options"] = None  # Too many options
            
            metadata.append(col_metadata)
        
        print(f"[DEBUG] Returning {len(metadata)} filterable columns")
        return {"columns": metadata}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting filter metadata: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.post("/api/ares/filters/structured")
async def apply_structured_filters(filters: Dict[str, Any] = None):
    """Apply structured data filters."""
    try:
        from ares_api import get_dataframe, apply_structured_filters
        df = get_dataframe()
        filtered_df, active_filters = apply_structured_filters(df, filters or {})
        return {
            "filtered_count": len(filtered_df),
            "total_count": len(df),
            "active_filters": active_filters,
            "data": filtered_df.to_dict('records')[:100],  # Limit to first 100 rows
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error applying filters: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.get("/api/ares/data-sample")
async def get_data_sample(n: int = 5):
    """Get a random sample of the filtered data."""
    try:
        from ares_api import get_dataframe
        df = get_dataframe()
        if len(df) == 0:
            return {"data": []}
        sample = df.sample(min(n, len(df)))
        return {"data": sample.to_dict('records')}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting data sample: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.get("/api/ares/embeddings/{embedding_key}")
async def get_embedding_data(embedding_key: str):
    """Get embedding data for visualization (reduced embeddings, labels, IDs)."""
    try:
        from ares_api import get_dataframe, _global_state
        from ares.databases.embedding_database import META_INDEX_NAMES
        import plotly.graph_objects as go
        
        if embedding_key not in META_INDEX_NAMES:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid embedding key. Must be one of: {META_INDEX_NAMES}"}
            )
        
        reduced_key = f"{embedding_key}_reduced"
        labels_key = f"{embedding_key}_labels"
        ids_key = f"{embedding_key}_ids"
        
        print(f"[DEBUG] Checking for embedding key: {reduced_key}")
        print(f"[DEBUG] Available keys in _global_state: {list(_global_state.keys())[:20]}...")  # Show first 20 keys
        
        # Check if reduced embeddings exist
        if reduced_key not in _global_state:
            available_keys = [k for k in _global_state.keys() if '_reduced' in k]
            all_keys = list(_global_state.keys())
            print(f"[DEBUG] Available _reduced keys: {available_keys}")
            print(f"[DEBUG] All _global_state keys: {all_keys}")
            
            # Try to initialize embeddings if they don't exist
            print(f"[DEBUG] Attempting to ensure embeddings are initialized...")
            try:
                from ares_api import initialize_ares_data
                # Re-initialize to ensure embeddings are loaded
                initialize_ares_data()
                # Check again
                if reduced_key not in _global_state:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "detail": f"Embedding data not found for {embedding_key} after initialization. "
                                     f"Available _reduced keys: {available_keys}. "
                                     f"This may mean embeddings were not generated during data ingestion."
                        }
                    )
            except Exception as init_err:
                print(f"[ERROR] Failed to re-initialize: {init_err}")
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Embedding data not found and re-initialization failed: {str(init_err)}"}
                )
        
        reduced = _global_state[reduced_key]
        labels = _global_state.get(labels_key, None)
        ids = _global_state.get(ids_key, None)
        
        df = get_dataframe()
        
        # Convert numpy arrays to lists for JSON serialization
        # reduced should be 2D array (n_points, 2) for 2D visualization
        if reduced.shape[1] > 2:
            # Use first 2 dimensions
            reduced_2d = reduced[:, :2]
        else:
            reduced_2d = reduced
        
        # Get IDs and raw data
        if ids is not None:
            point_ids = [str(id_val) for id_val in ids]
        elif "id" in df.columns:
            point_ids = df["id"].apply(str).tolist()
        else:
            point_ids = [str(i) for i in range(len(reduced_2d))]
        
        raw_data = df[embedding_key].tolist() if embedding_key in df.columns else [""] * len(reduced_2d)
        
        # Create a simple scatter plot
        fig = go.Figure()
        
        # Add scatter trace
        fig.add_trace(go.Scatter(
            x=reduced_2d[:, 0].tolist(),
            y=reduced_2d[:, 1].tolist(),
            mode='markers',
            marker=dict(
                size=5,
                color='#154e72',
                opacity=0.6,
            ),
            customdata=[[raw_data[i], point_ids[i]] for i in range(len(reduced_2d))],
            hovertemplate='<b>%{customdata[0]}</b><br>ID: %{customdata[1]}<extra></extra>',
            showlegend=False,
        ))
        
        fig.update_layout(
            title='',
            xaxis_title='',
            yaxis_title='',
            plot_bgcolor='transparent',
            paper_bgcolor='transparent',
            font=dict(color='#8a8a8a', size=10),
            height=500,
            margin=dict(l=40, r=40, t=20, b=40),
            xaxis=dict(showgrid=True, gridcolor='#2a2a2a'),
            yaxis=dict(showgrid=True, gridcolor='#2a2a2a'),
        )
        
        # Convert to dict
        fig_dict = fig.to_dict()
        
        return {
            "embedding_key": embedding_key,
            "figure": fig_dict,
            "point_count": len(reduced_2d),
            "point_ids": point_ids,
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting embedding data: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.post("/api/ares/filters/embedding")
async def apply_embedding_filters(selections: Dict[str, Any]):
    """Apply embedding-based filters from user selections."""
    try:
        from ares_api import get_dataframe, apply_structured_filters, get_embedding_filters
        df = get_dataframe()
        
        # First apply structured filters if any
        structured_filtered_df = df
        if selections.get("structured_filters"):
            structured_filtered_df, _ = apply_structured_filters(df, selections["structured_filters"])
        
        # Then apply embedding filters
        embedding_selections = selections.get("embedding_selections", {})
        filtered_df, embedding_figs = get_embedding_filters(df, structured_filtered_df, embedding_selections)
        
        return {
            "filtered_count": len(filtered_df),
            "total_count": len(df),
            "data": filtered_df.to_dict('records')[:100],
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error applying embedding filters: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.get("/api/ares/distributions")
async def get_distributions():
    """Get data distribution visualizations."""
    try:
        import traceback
        from ares_api import get_dataframe, get_data_distributions
        print("[DEBUG] Getting distributions...")
        df = get_dataframe()
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        visualizations = get_data_distributions(df)
        print(f"[DEBUG] Got {len(visualizations)} visualizations")
        return {"visualizations": visualizations}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_detail = f"Error getting distributions: {str(e)}\n\nTraceback:\n{error_trace}"
        print(f"[ERROR] {error_detail}")
        return JSONResponse(
            status_code=500,
            content={"detail": error_detail}
        )

@app.get("/api/ares/success-rate")
async def get_success_rate():
    """Get success rate analytics."""
    try:
        from ares_api import get_dataframe, get_success_rate_analytics
        df = get_dataframe()
        visualizations = get_success_rate_analytics(df)
        return {"visualizations": visualizations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting success rate: {str(e)}")

@app.get("/api/ares/time-series")
async def get_time_series():
    """Get time series analytics."""
    try:
        import traceback
        from ares_api import get_dataframe, get_time_series_analytics
        print("[DEBUG] Getting time series...")
        df = get_dataframe()
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        visualizations = get_time_series_analytics(df)
        print(f"[DEBUG] Got {len(visualizations)} time series visualizations")
        return {"visualizations": visualizations}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_detail = f"Error getting time series: {str(e)}\n\nTraceback:\n{error_trace}"
        print(f"[ERROR] {error_detail}")
        return JSONResponse(
            status_code=500,
            content={"detail": error_detail}
        )

@app.get("/api/ares/videos")
async def get_videos(n: int = 5):
    """Get video grid data."""
    try:
        from ares_api import get_dataframe, get_video_grid_data
        df = get_dataframe()
        videos = get_video_grid_data(df, n_videos=n)
        return {"videos": videos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting videos: {str(e)}")

@app.get("/api/ares/hero/{row_id}")
async def get_hero_display(row_id: str):
    """Get hero display data for a selected row."""
    try:
        from ares_api import get_dataframe, get_hero_display_data
        df = get_dataframe()
        hero_data = get_hero_display_data(row_id, df)
        return hero_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hero display: {str(e)}")

@app.get("/api/ares/robot-array/{row_id}")
async def get_robot_array(row_id: str):
    """Get robot array plots for a selected row."""
    try:
        from ares_api import get_robot_array_plots
        visualizations = get_robot_array_plots(row_id)
        return {"visualizations": visualizations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting robot array: {str(e)}")

@app.post("/api/ares/export")
async def export_ares_data(format: str = "csv", filters: Optional[Dict[str, Any]] = None):
    """Export ARES data."""
    try:
        from ares_api import get_dataframe, apply_structured_filters, export_data
        df = get_dataframe()
        
        # Apply filters if provided
        if filters:
            filtered_df, active_filters = apply_structured_filters(df, filters)
        else:
            filtered_df = df
            active_filters = {}
        
        # Get visualizations (simplified)
        visualizations = []
        
        export_result = export_data(filtered_df, active_filters, visualizations, format)
        return export_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

