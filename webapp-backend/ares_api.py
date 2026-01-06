"""
FastAPI backend module that replaces Streamlit's server-side functionality.
Provides REST API endpoints for all ARES dashboard features.

This module wraps the existing ares.app functionality and removes Streamlit dependencies
by using a global state dictionary instead of Streamlit's session state.
"""

import os
import json
import traceback
import sys
import types
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from fastapi import HTTPException

# CRITICAL: Set up Streamlit mock BEFORE importing any ares modules
# This ensures that when ares modules import streamlit, they get our mock
class MockSessionState:
    def __init__(self):
        self._data = {}
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __contains__(self, key):
        return key in self._data
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def keys(self):
        return self._data.keys()
    
    # Support attribute access (e.g., st.session_state.ENGINE)
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self._data.get(name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

# Create and inject mock streamlit module
# Force replace to ensure our mock is used
streamlit_module = types.ModuleType('streamlit')
streamlit_module.session_state = MockSessionState()
sys.modules['streamlit'] = streamlit_module

# Also try to update any existing references
try:
    import streamlit
    streamlit.session_state = MockSessionState()
except:
    pass

# Now import ares modules (they will use our mock streamlit)
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
from ares.constants import ARES_DATA_DIR

# Global state (replaces Streamlit session state)
_global_state: Dict[str, Any] = {}
tmp_dump_dir = os.path.join(ARES_DATA_DIR, "webapp_tmp")
os.makedirs(tmp_dump_dir, exist_ok=True)


def _setup_streamlit_mock():
    """Return the mock Streamlit session state (already set up at module level)."""
    return sys.modules['streamlit'].session_state


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
    """Initialize ARES data (replaces initialize_data from init_data.py)."""
    global _global_state
    
    # Skip if already initialized
    if all(key in _global_state for key in ["ENGINE", "SESSION", "df", "INDEX_MANAGER"]):
        return _global_state["df"]
    
    try:
        # Extract actual file paths from connection strings
        robot_db_file = _extract_file_path_from_sqlite_url(ROBOT_DB_PATH)
        embedding_db_dir = EMBEDDING_DB_PATH  # This should already be a directory path
        
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
        
        # Ensure mock is set up
        mock_st = _setup_streamlit_mock()
        print(f"[DEBUG] Mock session state type: {type(mock_st)}")
        print(f"[DEBUG] Mock session state keys before init: {list(mock_st.keys())}")
        
        # Verify streamlit module has our mock
        import streamlit as st_check
        print(f"[DEBUG] streamlit module session_state type: {type(st_check.session_state)}")
        print(f"[DEBUG] Is our mock? {isinstance(st_check.session_state, MockSessionState)}")
        
        # Use existing initialize_data function but sync with our global state
        print("[DEBUG] Importing initialize_data...")
        
        # Ensure streamlit is mocked before importing init_data
        print("[DEBUG] Checking streamlit mock before import...")
        import streamlit as st_precheck
        print(f"[DEBUG] streamlit module loaded, session_state type: {type(st_precheck.session_state)}")
        if not isinstance(st_precheck.session_state, MockSessionState):
            print("[WARNING] streamlit.session_state is not our mock before import!")
            st_precheck.session_state = MockSessionState()
            sys.modules['streamlit'].session_state = MockSessionState()
        
        print("[DEBUG] About to import ares.app.init_data...")
        print("[DEBUG] Testing individual imports to find which one hangs...")
        import time
        
        # Test imports one by one to find the culprit
        # Note: ares.models.base is skipped because it hangs - we've made it lazy in init_data.py
        test_imports = [
            ("os", "os"),
            ("numpy", "np"),
            ("pandas", "pd"),
            ("streamlit", "st"),
            ("sqlalchemy", "sqlalchemy"),
            ("ares.databases.annotation_database", "ANNOTATION_DB_PATH"),
            ("ares.databases.embedding_database", "EMBEDDING_DB_PATH"),
            ("ares.databases.structured_database", "ROBOT_DB_PATH"),
            ("ares.utils.clustering", "cluster_embeddings"),
        ]
        
        for import_name, attr_name in test_imports:
            try:
                print(f"[DEBUG] Testing import: {import_name}...")
                start = time.time()
                if "." in import_name:
                    parts = import_name.split(".")
                    mod = __import__(import_name, fromlist=[parts[-1]])
                else:
                    mod = __import__(import_name)
                elapsed = time.time() - start
                if elapsed > 1.0:
                    print(f"[WARNING] {import_name} took {elapsed:.2f} seconds to import")
                else:
                    print(f"[DEBUG] {import_name} imported in {elapsed:.2f} seconds")
            except Exception as e:
                print(f"[ERROR] Failed to import {import_name}: {type(e).__name__}: {str(e)}")
        
        # Now try importing the actual module
        print("[DEBUG] Now importing ares.app.init_data module...")
        start_time = time.time()
        try:
            import ares.app.init_data as init_data_module
            elapsed = time.time() - start_time
            print(f"[DEBUG] Module imported (took {elapsed:.2f} seconds)")
            
            # Now get the function
            print("[DEBUG] Getting initialize_data function from module...")
            initialize_data = init_data_module.initialize_data
            print(f"[DEBUG] Successfully got initialize_data function")
            
        except ImportError as import_error:
            elapsed = time.time() - start_time
            print(f"[ERROR] Failed to import after {elapsed:.2f} seconds: {type(import_error).__name__}: {str(import_error)}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to import init_data module: {str(import_error)}") from import_error
        except Exception as import_error:
            elapsed = time.time() - start_time
            print(f"[ERROR] Unexpected error importing after {elapsed:.2f} seconds: {type(import_error).__name__}: {str(import_error)}")
            traceback.print_exc()
            raise RuntimeError(f"Unexpected error importing init_data: {str(import_error)}") from import_error
        
        # Verify streamlit is still mocked after import
        print("[DEBUG] Verifying streamlit mock after import...")
        import streamlit as st_check2
        print(f"[DEBUG] After import, streamlit session_state type: {type(st_check2.session_state)}")
        if not isinstance(st_check2.session_state, MockSessionState):
            print("[WARNING] streamlit.session_state is not our mock after import!")
            # Force replace it
            st_check2.session_state = MockSessionState()
            sys.modules['streamlit'].session_state = MockSessionState()
        
        # Also check if init_data module has its own st reference
        print("[DEBUG] Checking init_data module for st reference...")
        import ares.app.init_data as init_data_module
        if hasattr(init_data_module, 'st'):
            if not isinstance(init_data_module.st.session_state, MockSessionState):
                print("[WARNING] init_data.st.session_state is not our mock!")
                init_data_module.st.session_state = MockSessionState()
        
        print(f"[DEBUG] About to call initialize_data with tmp_dump_dir: {tmp_dump_dir}")
        try:
            initialize_data(tmp_dump_dir)
            print(f"[DEBUG] initialize_data completed successfully")
        except Exception as init_error:
            error_type = type(init_error).__name__
            error_msg = str(init_error)
            error_trace = traceback.format_exc()
            print(f"[ERROR] initialize_data failed:")
            print(f"  Type: {error_type}")
            print(f"  Message: {error_msg}")
            print(f"  Traceback:\n{error_trace}")
            # Re-raise with more context
            raise RuntimeError(
                f"Failed to initialize data: {error_type}: {error_msg}\n\n"
                f"Full traceback:\n{error_trace}"
            ) from init_error
        
        print(f"[DEBUG] Mock session state keys after init: {list(mock_st.keys())}")
        
        # Copy from mock session state to global state
        _global_state["ENGINE"] = mock_st.get("ENGINE")
        _global_state["SESSION"] = mock_st.get("SESSION")
        _global_state["df"] = mock_st.get("df")
        _global_state["INDEX_MANAGER"] = mock_st.get("INDEX_MANAGER")
        _global_state["all_vecs"] = mock_st.get("all_vecs", {})
        _global_state["all_ids"] = mock_st.get("all_ids", {})
        _global_state["annotations_db"] = mock_st.get("annotations_db")
        _global_state["annotation_db_stats"] = mock_st.get("annotation_db_stats", {})
        
        # Also copy embedding data
        print(f"[DEBUG] Copying embedding data for META_INDEX_NAMES: {META_INDEX_NAMES}")
        print(f"[DEBUG] Mock session state keys: {list(mock_st._data.keys())[:30]}...")  # Show first 30 keys
        
        for index_name in META_INDEX_NAMES:
            for key in [f"{index_name}_embeddings", f"{index_name}_reduced", f"{index_name}_labels", f"{index_name}_ids"]:
                if key in mock_st._data:
                    _global_state[key] = mock_st._data[key]
                    print(f"[DEBUG] Copied {key} to _global_state (shape: {getattr(_global_state[key], 'shape', 'N/A')})")
                else:
                    print(f"[DEBUG] Key {key} not found in mock_st._data")
        
        # Debug: show what embedding keys we have
        embedding_keys = [k for k in _global_state.keys() if any(name in k for name in META_INDEX_NAMES)]
        print(f"[DEBUG] Embedding keys in _global_state: {embedding_keys}")
        
        # Also check if embeddings exist in mock_st but weren't copied
        mock_embedding_keys = [k for k in mock_st._data.keys() if any(name in k for name in META_INDEX_NAMES)]
        print(f"[DEBUG] Embedding keys in mock_st._data: {mock_embedding_keys}")
        
        if "df" not in _global_state or _global_state["df"] is None:
            raise ValueError("DataFrame was not initialized properly")
        
        if _global_state["df"].empty:
            raise ValueError(
                f"DataFrame is empty. Database at {ROBOT_DB_PATH} exists but contains no data. "
                f"Please ensure ares_ingestion.py completed successfully."
            )
        
        print(f"[DEBUG] Successfully initialized. DataFrame shape: {_global_state['df'].shape}")
        print(f"[DEBUG] Returning DataFrame with {len(_global_state['df'])} rows")
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
    """Apply structured data filters using existing filter_helpers logic."""
    # Setup mock Streamlit for compatibility
    mock_st = _setup_streamlit_mock()
    
    # Sync global state to mock
    for key, value in _global_state.items():
        mock_st._data[key] = value
    
    # Use existing structured_data_filters_display but capture results
    from ares.app.filter_helpers import structured_data_filters_display
    
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
    """Get embedding-based filters."""
    # Setup mock Streamlit for compatibility
    mock_st = _setup_streamlit_mock()
    
    # Sync global state to mock
    for key, value in _global_state.items():
        mock_st._data[key] = value
    
    embedding_figs = {}
    embedding_filtered_dfs = []
    
    # Get filtered dataframes for each embedding
    from ares.app.filter_helpers import create_embedding_data_filter_display
    
    for raw_data_key in META_INDEX_NAMES:
        if f"{raw_data_key}_reduced" not in _global_state:
            continue
        
        # If embedding selections provided, use them
        if embedding_selections and raw_data_key in embedding_selections:
            # Apply selection directly
            selected_ids = embedding_selections[raw_data_key].get("selected_ids", [])
            filtered_df = df[df["id"].isin(selected_ids)]
            embedding_filtered_dfs.append(filtered_df)
            # Create a simple figure for the selection
            embedding_figs[raw_data_key] = None  # Would need to create figure from selection
        else:
            # Use existing filter display logic
            filtered_df, cluster_fig = create_embedding_data_filter_display(
                df=df,
                id_key="id",
                raw_data_key=raw_data_key,
                kept_ids=structured_filtered_df["id"].apply(str).tolist(),
            )
            embedding_filtered_dfs.append(filtered_df)
            embedding_figs[raw_data_key] = cluster_fig
    
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


def get_data_distributions(filtered_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get data distribution visualizations."""
    try:
        print(f"[DEBUG] get_data_distributions: DataFrame shape: {filtered_df.shape}")
        if filtered_df.empty:
            print("[WARNING] DataFrame is empty, returning empty visualizations")
            return []
        if len(filtered_df.columns) == 0:
            print("[WARNING] DataFrame has no columns, returning empty visualizations")
            return []
        print(f"[DEBUG] DataFrame columns: {list(filtered_df.columns)[:10]}...")  # Show first 10 columns
        visualizations = generate_automatic_visualizations(filtered_df)
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

