"""
FastAPI backend module that replaces Streamlit's server-side functionality.
Provides REST API endpoints for all ARES dashboard features.

This module implements ARES functionality without Streamlit dependencies,
using a global state dictionary instead of Streamlit's session state.
"""

import os
import json
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from fastapi import HTTPException

# Import ares modules (non-Streamlit parts only)
from ares.app.data_analysis import generate_automatic_visualizations
from ares.app.viz_helpers import (
    generate_success_rate_visualizations,
    generate_time_series_visualizations,
    generate_robot_array_plot_visualizations,
    total_statistics,
    annotation_statistics,
)
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)
from ares.databases.embedding_database import (
    EMBEDDING_DB_PATH,
    META_INDEX_NAMES,
    FaissIndex,
    IndexManager,
)
from ares.utils.clustering import cluster_embeddings
from ares.constants import ARES_DATA_DIR

# Global state (replaces Streamlit session state)
_global_state: Dict[str, Any] = {}
tmp_dump_dir = os.path.join(ARES_DATA_DIR, "webapp_tmp")
os.makedirs(tmp_dump_dir, exist_ok=True)


def _load_cached_embeddings(
    tmp_dump_dir: str, index_name: str, stored_embeddings: np.ndarray
) -> tuple | None:
    """
    Load cached embedding visualizations if available.
    """
    embeddings_path = os.path.join(tmp_dump_dir, f"{index_name}_embeddings.npy")
    clusters_path = os.path.join(tmp_dump_dir, f"{index_name}_clusters.npz")
    ids_path = os.path.join(tmp_dump_dir, f"{index_name}_ids.npy")

    if not (
        os.path.exists(embeddings_path)
        and os.path.exists(clusters_path)
        and os.path.exists(ids_path)
    ):
        return None

    loaded_embeddings = np.load(embeddings_path)
    if not (
        len(loaded_embeddings) == len(stored_embeddings)
        and np.allclose(loaded_embeddings, stored_embeddings)
    ):
        # this means we have new embeddings
        return None

    # Valid cached data found - load everything
    clusters_data = np.load(clusters_path)
    loaded_ids = np.load(ids_path)
    return (
        loaded_embeddings,
        clusters_data["reduced"],
        clusters_data["labels"],
        loaded_ids,
    )


def _save_embeddings(
    tmp_dump_dir: str,
    index_name: str,
    embeddings: np.ndarray,
    reduced: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
) -> None:
    """
    Save reduced embeddings, clusters, and IDs to disk
    """
    embeddings_path = os.path.join(tmp_dump_dir, f"{index_name}_embeddings.npy")
    clusters_path = os.path.join(tmp_dump_dir, f"{index_name}_clusters.npz")
    ids_path = os.path.join(tmp_dump_dir, f"{index_name}_ids.npy")

    np.save(embeddings_path, embeddings)
    np.savez(clusters_path, reduced=reduced, labels=labels)
    np.save(ids_path, ids)


def _extract_file_path_from_sqlite_url(url: str) -> str:
    """Extract the actual file path from a SQLite connection string."""
    # Handle both sqlite:/// (relative) and sqlite://// (absolute) formats
    if url.startswith("sqlite:///"):
        # Remove sqlite:/// prefix
        file_path = url[len("sqlite:///"):]
        # For absolute paths, there's an extra slash: sqlite:////path -> /path
        # For relative paths: sqlite:///path -> path
        return file_path
    return url


def initialize_ares_data() -> pd.DataFrame:
    """Initialize ARES data without Streamlit dependencies."""
    global _global_state
    
    # Skip if already initialized
    if all(key in _global_state for key in ["ENGINE", "SESSION", "df", "INDEX_MANAGER"]):
        print("[INFO] ARES data already initialized, returning cached DataFrame.")
        return _global_state["df"]
    
    try:
        # Extract actual file paths from connection strings
        robot_db_file = _extract_file_path_from_sqlite_url(ROBOT_DB_PATH)
        embedding_db_dir = EMBEDDING_DB_PATH
        
        # Check if database files exist
        if not os.path.exists(robot_db_file):
            raise FileNotFoundError(
                f"Structured database not found at {robot_db_file}. "
                f"Please run ares_ingestion.py first to create the database."
            )
        
        if not os.path.exists(embedding_db_dir):
            raise FileNotFoundError(
                f"Embedding database directory not found at {embedding_db_dir}. "
                f"Please run ares_ingestion.py first to create the database."
            )
        
        # Initialize database and session
        print("Initializing database and session")
        engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
        sess = Session(engine)
        _global_state["ENGINE"] = engine
        _global_state["SESSION"] = sess
        
        # Load dataframe
        print("Loading dataframe")
        query = select(RolloutSQLModel)
        df = pd.read_sql(query, engine)
        # Filter out unnamed columns
        df = df[[c for c in df.columns if "unnamed" not in c.lower()]]
        _global_state["df"] = df
        
        # Initialize index manager
        print("Initializing index manager")
        index_manager = IndexManager(base_dir=EMBEDDING_DB_PATH, index_class=FaissIndex)
        _global_state["INDEX_MANAGER"] = index_manager
        
        # Get all vectors and their IDs
        print("Getting all vectors and their IDs")
        all_data = index_manager.get_all_matrices()
        _global_state["all_vecs"] = {
            name: data["arrays"] for name, data in all_data.items()
        }
        _global_state["all_ids"] = {name: data["ids"] for name, data in all_data.items()}
        
        # Create tmp directory if it doesn't exist
        os.makedirs(tmp_dump_dir, exist_ok=True)
        
        # Process each index type
        for index_name in META_INDEX_NAMES:
            print(f"Processing {index_name} index")
            if index_name not in index_manager.indices:
                print(f"Index {index_name} not found, skipping.")
                continue
            try:
                print(f"  Getting vectors and IDs for {index_name}...")
                stored_embeddings = index_manager.indices[index_name].get_all_vectors()
                stored_ids = index_manager.indices[index_name].get_all_ids()
                print(f"  Found {len(stored_embeddings)} embeddings for {index_name}")
                
                # Try loading from cache first
                print(f"  Checking cache for {index_name}...")
                cached_data = _load_cached_embeddings(
                    tmp_dump_dir, index_name, stored_embeddings
                )
                if cached_data is not None:
                    print(f"  Using cached data for {index_name}")
                    embeddings, reduced, labels, ids = cached_data
                else:
                    # Create new embeddings and clusters
                    print(f"  Computing clusters for {index_name} (this may take a while)...")
                    embeddings = stored_embeddings
                    reduced, labels, _ = cluster_embeddings(embeddings)
                    ids = stored_ids
                    print(f"  Saving embeddings for {index_name}...")
                    _save_embeddings(tmp_dump_dir, index_name, embeddings, reduced, labels, ids)
                    print(f"  Completed clustering for {index_name}")
                
                # Store in global state
                print(f"  Storing {index_name} in global state...")
                _global_state[f"{index_name}_embeddings"] = embeddings
                _global_state[f"{index_name}_reduced"] = reduced
                _global_state[f"{index_name}_labels"] = labels
                _global_state[f"{index_name}_ids"] = stored_ids
                print(f"  Successfully processed {index_name}")
            except Exception as e:
                print(f"  ERROR processing {index_name}: {type(e).__name__}: {str(e)}")
                traceback.print_exc()
                # Continue with other indices even if one fails
                continue
        
        # Empty database is valid after "New Task" - return empty DataFrame gracefully
        if _global_state["df"].empty:
            print(f"[INFO] DataFrame is empty (database cleared or no data ingested yet). This is expected after 'New Task'.")
            return _global_state["df"]
        
        print(f"[DEBUG] Successfully initialized. DataFrame shape: {_global_state['df'].shape}")
        return _global_state["df"]
    except Exception as e:
        error_msg = f"Error in initialize_ares_data: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        raise RuntimeError(error_msg) from e


def get_dataframe() -> pd.DataFrame:
    """Get the main dataframe."""
    if "df" not in _global_state:
        return initialize_ares_data()
    return _global_state["df"]


def _get_cache_key(environment: str | None, selected_axes: list[str] | None) -> str:
    """Generate a cache key for plots based on environment and axes."""
    env = environment or "default"
    axes_str = ",".join(sorted(selected_axes)) if selected_axes else "default"
    return f"distributions_{env}_{axes_str}"


def generate_and_cache_distributions(environment: str | None = None, selected_axes: list[str] | None = None) -> None:
    """
    Generate distribution plots for the specified environment and axes, and cache them.
    
    Args:
        environment: "Indoor" or "Outdoor". If None, defaults to "Indoor".
        selected_axes: List of axis names. If None, uses default axes for the environment.
    """
    global _global_state
    
    try:
        # Default to Indoor if not specified
        if environment is None:
            environment = "Indoor"
        
        # Get default axes for the environment if not specified
        if selected_axes is None:
            if environment == "Indoor":
                selected_axes = ["Objects", "Lighting", "Materials"]
            else:  # Outdoor
                selected_axes = ["Objects", "Lighting", "Weather", "Road Surface"]
        
        print(f"[INFO] Generating and caching {environment} distribution plots with axes: {selected_axes}...")
        
        # Get fresh dataframe
        df = get_dataframe()
        
        if df.empty:
            print("[WARNING] DataFrame is empty, skipping plot generation")
            return
        
        # Generate plots for the specified environment
        cache_key = _get_cache_key(environment, selected_axes)
        # Use use_cache=False to avoid recursion when generating initial cache
        plots = get_data_distributions(df, environment=environment, selected_axes=selected_axes, use_cache=False)
        _global_state[f"plot_cache_{cache_key}"] = plots
        print(f"[INFO] Cached {len(plots)} {environment} plots")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate and cache distributions: {type(e).__name__}: {str(e)}")
        traceback.print_exc()


def get_state_info() -> Dict[str, Any]:
    """Get state information for debugging."""
    try:
        df = get_dataframe()
        print(f"[DEBUG] get_state_info: DataFrame shape: {df.shape}")
        
        try:
            stats = total_statistics(df)
            print(f"[DEBUG] get_state_info: total_statistics completed")
        except Exception as e:
            print(f"[WARNING] total_statistics failed: {type(e).__name__}: {str(e)}")
            stats = {}
        
        try:
            ann_stats = annotation_statistics(_global_state.get("annotations_db"))
            print(f"[DEBUG] get_state_info: annotation_statistics completed")
        except Exception as e:
            print(f"[WARNING] annotation_statistics failed: {type(e).__name__}: {str(e)}")
            ann_stats = {}
        
        return {
            "total_rows": len(df),
            "total_statistics": stats,
            "annotation_statistics": ann_stats,
            "has_embeddings": "INDEX_MANAGER" in _global_state,
            "has_annotations": _global_state.get("annotations_db") is not None,
        }
    except Exception as e:
        print(f"[ERROR] get_state_info failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise


def apply_structured_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply structured data filters."""
    
    # If filters provided, apply them directly
    if filters:
        filtered_df = df.copy()
        active_filters = {}
        
        for col, filter_value in filters.items():
            if col not in filtered_df.columns:
                continue
            if not isinstance(filter_value, dict):
                continue
                
            if "range" in filter_value:
                # Numeric range filter
                try:
                    min_val, max_val = filter_value["range"]
                    if min_val is None or max_val is None:
                        continue  # Skip invalid range
                    numeric_col = pd.to_numeric(filtered_df[col], errors="coerce")
                    mask = (numeric_col >= min_val) & (numeric_col <= max_val)
                    if filter_value.get("include_none", True):
                        mask |= filtered_df[col].isna()
                    filtered_df = filtered_df[mask]
                    active_filters[col] = filter_value
                except (ValueError, TypeError) as e:
                    print(f"[WARNING] Invalid range filter for {col}: {e}")
                    continue
            elif "values" in filter_value:
                # Categorical filter
                try:
                    if not filter_value["values"] or len(filter_value["values"]) == 0:
                        continue  # Skip empty filter
                    filtered_df = filtered_df[filtered_df[col].isin(filter_value["values"])]
                    active_filters[col] = filter_value
                except Exception as e:
                    print(f"[WARNING] Invalid values filter for {col}: {e}")
                    continue
    else:
        # Use existing filter display logic (requires Streamlit UI, so we'll use a simplified version)
        # For API use, we need filters to be provided
        filtered_df = df
        active_filters = {}
    
    return filtered_df, active_filters


def get_embedding_filters(df: pd.DataFrame, structured_filtered_df: pd.DataFrame, embedding_selections: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Get embedding-based filters.
    
    Note: Embedding filter display functions require Streamlit UI components.
    For API use, this is a simplified version that only applies selections if provided.
    """
    embedding_figs = {}
    embedding_filtered_dfs = []
    
    for raw_data_key in META_INDEX_NAMES:
        if f"{raw_data_key}_reduced" not in _global_state:
            continue
        
        # If embedding selections provided, use them
        if embedding_selections and raw_data_key in embedding_selections:
            selected_ids = embedding_selections[raw_data_key].get("selected_ids", [])
            if selected_ids:
                filtered_df = structured_filtered_df[structured_filtered_df["id"].isin(selected_ids)]
                embedding_filtered_dfs.append(filtered_df)
                embedding_figs[raw_data_key] = None  # Would need to create figure from selection
    
    # Combine all filtered dataframes (AND operation)
    if embedding_filtered_dfs:
        all_filtered_ids = set(embedding_filtered_dfs[0]["id"])
        for filtered_df in embedding_filtered_dfs[1:]:
            all_filtered_ids &= set(filtered_df["id"])
        
        filtered_df = structured_filtered_df[
            structured_filtered_df["id"].isin(all_filtered_ids)
        ]
    else:
        filtered_df = structured_filtered_df
    
    return filtered_df, embedding_figs


def get_data_distributions(
    filtered_df: pd.DataFrame,
    environment: str | None = None,
    selected_axes: list[str] | None = None,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Get data distribution visualizations.
    
    Args:
        filtered_df: DataFrame to visualize
        environment: "Indoor" or "Outdoor" mode
        selected_axes: List of axis names to show (e.g., ["Objects", "Lighting", "Materials"])
        use_cache: If True, check cache first before generating on-demand
    """
    try:
        print(f"[DEBUG] get_data_distributions: DataFrame shape: {filtered_df.shape}")
        print(f"[DEBUG] Environment: {environment}, Selected axes: {selected_axes}, use_cache: {use_cache}")
        
        # Check cache first if enabled
        if use_cache:
            cache_key = _get_cache_key(environment, selected_axes)
            cache_storage_key = f"plot_cache_{cache_key}"
            if cache_storage_key in _global_state:
                cached_plots = _global_state[cache_storage_key]
                print(f"[DEBUG] Returning {len(cached_plots)} cached visualizations for key: {cache_key}")
                return cached_plots
            else:
                print(f"[DEBUG] No cached plots found for key: {cache_key}, generating on-demand")
        
        if len(filtered_df.columns) == 0:
            print("[WARNING] DataFrame has no columns, returning empty visualizations")
            return []
        print(f"[DEBUG] DataFrame columns: {list(filtered_df.columns)[:10]}...")  # Show first 10 columns
        # Always call generate_automatic_visualizations even if dataframe is empty
        # so it can return empty plots for selected axes
        visualizations = generate_automatic_visualizations(
            filtered_df,
            environment=environment,
            selected_axes=selected_axes,
        )
        print(f"[DEBUG] Generated {len(visualizations)} visualizations")
        
        # Convert Plotly figures to JSON-serializable format
        import plotly.io as pio
        import json
        result = []
        for viz in visualizations:
            try:
                fig = viz["figure"]
                # Use Plotly's JSON encoder to properly serialize the figure
                fig_dict = json.loads(pio.to_json(fig))
                result.append({
                    "title": viz.get("title", "Untitled"),
                    "figure": fig_dict,
                })
            except Exception as viz_error:
                print(f"[WARNING] Failed to convert visualization: {type(viz_error).__name__}: {str(viz_error)}")
                traceback.print_exc()
                continue
        
        print(f"[DEBUG] Returning {len(result)} visualizations")
        return result
    except Exception as e:
        print(f"[ERROR] get_data_distributions failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise


def get_success_rate_analytics(filtered_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get success rate analytics visualizations."""
    import plotly.io as pio
    import json
    visualizations = generate_success_rate_visualizations(filtered_df)
    
    result = []
    for viz in visualizations:
        try:
            fig = viz["figure"]
            # Use Plotly's JSON encoder to properly serialize the figure
            fig_dict = json.loads(pio.to_json(fig))
            result.append({
                "title": viz.get("title", "Untitled"),
                "figure": fig_dict,
            })
        except Exception as viz_error:
            print(f"[WARNING] Failed to convert success rate visualization: {type(viz_error).__name__}: {str(viz_error)}")
            traceback.print_exc()
            continue
    
    return result


def get_time_series_analytics(filtered_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get time series analytics visualizations."""
    try:
        print(f"[DEBUG] get_time_series_analytics: DataFrame shape: {filtered_df.shape}")
        print(f"[DEBUG] Checking for time_column 'ingestion_time' in columns: {'ingestion_time' in filtered_df.columns}")
        
        # Check if time column exists
        if "ingestion_time" not in filtered_df.columns:
            print("[WARNING] 'ingestion_time' column not found, using first datetime column or skipping")
            datetime_cols = filtered_df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                time_column = datetime_cols[0]
                print(f"[DEBUG] Using '{time_column}' as time column instead")
            else:
                print("[WARNING] No datetime columns found, returning empty visualizations")
                return []
        else:
            time_column = "ingestion_time"
        
        visualizations = generate_time_series_visualizations(
            filtered_df, time_column=time_column
        )
        print(f"[DEBUG] Generated {len(visualizations)} time series visualizations")
        
        # Convert Plotly figures to JSON-serializable format
        import plotly.io as pio
        import json
        result = []
        for viz in visualizations:
            try:
                fig = viz.get("figure")
                if fig is None:
                    print("[WARNING] Visualization has no figure, skipping")
                    continue
                # Use Plotly's JSON encoder to properly serialize the figure
                fig_dict = json.loads(pio.to_json(fig))
                result.append({
                    "title": viz.get("title", "Untitled"),
                    "figure": fig_dict,
                })
            except Exception as viz_error:
                print(f"[WARNING] Failed to convert time series visualization: {type(viz_error).__name__}: {str(viz_error)}")
                traceback.print_exc()
                continue
        
        print(f"[DEBUG] Returning {len(result)} time series visualizations")
        return result
    except Exception as e:
        print(f"[ERROR] get_time_series_analytics failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise


def get_video_grid_data(filtered_df: pd.DataFrame, n_videos: int = 5) -> List[Dict[str, Any]]:
    """Get video grid data."""
    if len(filtered_df) == 0:
        return []
    
    # Remove duplicates
    if 'id' in filtered_df.columns:
        unique_df = filtered_df.drop_duplicates(subset=['id'], keep='first')
    elif 'filename' in filtered_df.columns:
        unique_df = filtered_df.drop_duplicates(subset=['filename'], keep='first')
    else:
        unique_df = filtered_df.drop_duplicates(keep='first')
    
    # Select random episodes
    if len(unique_df) >= n_videos:
        display_rows = unique_df.sample(n=n_videos, random_state=None)
    else:
        display_rows = unique_df
    
    # Convert to JSON-serializable format
    videos = []
    for _, row in display_rows.iterrows():
        video_data = {
            "id": str(row.get('id', '')),
            "filename": str(row.get('filename', '')) if 'filename' in row else None,
        }
        # Add other relevant fields
        for col in ['episode_index', 'task_language_instruction', 'video_path']:
            if col in row:
                video_data[col] = str(row[col]) if pd.notna(row[col]) else None
        videos.append(video_data)
    
    return videos


def get_hero_display_data(selected_row_id: str, filtered_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Get hero display data for a selected row."""
    df = get_dataframe()
    
    # Find the selected row
    if filtered_df is None:
        filtered_df = df
    
    if 'id' in filtered_df.columns:
        selected_row = filtered_df[filtered_df['id'] == selected_row_id]
    else:
        selected_row = filtered_df.iloc[0:1] if len(filtered_df) > 0 else pd.DataFrame()
    
    if len(selected_row) == 0:
        raise HTTPException(status_code=404, detail="Selected row not found")
    
    row = selected_row.iloc[0]
    
    # Get similar examples using INDEX_MANAGER
    similar_examples = []
    index_manager = _global_state.get("INDEX_MANAGER")
    all_vecs = _global_state.get("all_vecs", {})
    
    if index_manager and all_vecs:
        # This would use the index manager to find similar examples
        # Simplified implementation
        try:
            from ares.app.hero_display import create_similarity_viz_objects
            similar_ids, similar_data = create_similarity_viz_objects(
                row,
                df,
                index_manager,
                retrieve_n_most_similar=10,
            )
            similar_examples = [
                {"id": sid, "data": sdata} for sid, sdata in zip(similar_ids, similar_data)
            ]
        except Exception as e:
            print(f"Error getting similar examples: {e}")
    
    return {
        "selected_row": row.to_dict(),
        "similar_examples": similar_examples,
    }


def get_robot_array_plots(selected_row_id: str) -> List[Dict[str, Any]]:
    """Get robot array plot visualizations."""
    df = get_dataframe()
    
    # Find the selected row
    if 'id' in df.columns:
        selected_row = df[df['id'] == selected_row_id]
    else:
        selected_row = df.iloc[0:1] if len(df) > 0 else pd.DataFrame()
    
    if len(selected_row) == 0:
        return []
    
    row = selected_row.iloc[0]
    
    visualizations = generate_robot_array_plot_visualizations(
        row,
        _global_state.get("all_vecs", {}),
        show_n=1000,
    )
    
    # Convert Plotly figures to JSON-serializable format
    import plotly.io as pio
    import json
    result = []
    for viz in visualizations:
        try:
            fig = viz.get("figure")
            if fig is None:
                print("[WARNING] Robot array visualization has no figure, skipping")
                continue
            # Use Plotly's JSON encoder to properly serialize the figure
            fig_dict = json.loads(pio.to_json(fig))
            result.append({
                "title": viz.get("title", "Untitled"),
                "figure": fig_dict,
            })
        except Exception as viz_error:
            print(f"[WARNING] Failed to convert robot array visualization: {type(viz_error).__name__}: {str(viz_error)}")
            traceback.print_exc()
            continue
    
    return result


def export_data(
    filtered_df: pd.DataFrame,
    active_filters: Dict[str, Any],
    visualizations: List[Dict[str, Any]],
    format: str = "csv"
) -> Dict[str, Any]:
    """Export filtered data and visualizations."""
    if format == "csv":
        csv_content = filtered_df.to_csv(index=False)
        return {
            "format": "csv",
            "content": csv_content,
            "filename": "ares_data_export.csv",
        }
    elif format == "json":
        json_content = filtered_df.to_json(orient="records", indent=2)
        return {
            "format": "json",
            "content": json_content,
            "filename": "ares_data_export.json",
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

