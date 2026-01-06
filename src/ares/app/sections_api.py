"""
API version of sections.py - removes Streamlit dependencies and returns data instead of rendering UI.
This replaces the Streamlit-based sections.py for use with TSX frontend.
"""

import typing as t
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from ares.app.data_analysis import generate_automatic_visualizations
from ares.app.filter_helpers import (
    create_embedding_data_filter_display,
    structured_data_filters_display,
)
from ares.app.viz_helpers import (
    annotation_statistics,
    display_video_grid,
    generate_robot_array_plot_visualizations,
    generate_success_rate_visualizations,
    generate_time_series_visualizations,
    total_statistics,
)
from ares.databases.embedding_database import META_INDEX_NAMES


# Global state storage (replaces Streamlit session state)
_global_state: Dict[str, Any] = {}


def set_global_state(key: str, value: Any) -> None:
    """Set a value in global state."""
    _global_state[key] = value


def get_global_state(key: str, default: Any = None) -> Any:
    """Get a value from global state."""
    return _global_state.get(key, default)


def load_data(tmp_dump_dir: str, force_reload: bool = False) -> pd.DataFrame:
    """Load data from global state or initialize if needed."""
    if not force_reload and "df" in _global_state:
        return _global_state["df"]
    
    # Initialize data if not already done
    from ares.app.init_data import initialize_data
    initialize_data(tmp_dump_dir)
    
    # Store in global state
    import streamlit as st
    if hasattr(st, 'session_state') and 'df' in st.session_state:
        _global_state["df"] = st.session_state.df
        _global_state["ENGINE"] = st.session_state.get("ENGINE")
        _global_state["SESSION"] = st.session_state.get("SESSION")
        _global_state["INDEX_MANAGER"] = st.session_state.get("INDEX_MANAGER")
        _global_state["all_vecs"] = st.session_state.get("all_vecs", {})
        _global_state["all_ids"] = st.session_state.get("all_ids", {})
        _global_state["annotations_db"] = st.session_state.get("annotations_db")
        return st.session_state.df
    
    raise RuntimeError("Failed to initialize data")


def loading_data_section(title: str, tmp_dump_dir: str) -> pd.DataFrame:
    """Load data section - returns dataframe."""
    return load_data(tmp_dump_dir)


def state_info_section(df: pd.DataFrame) -> Dict[str, Any]:
    """Get state information - returns dict instead of displaying."""
    from ares.app.init_data import display_state_info
    import streamlit as st
    
    # Get state info
    state_info = {}
    if hasattr(st, 'session_state'):
        state_info["session_keys"] = list(st.session_state.keys())
    
    # Get statistics
    stats = total_statistics(df)
    #ann_stats = annotation_statistics(st.session_state.get("annotations_db") if hasattr(st, 'session_state') else None)
    
    return {
        "state_info": state_info,
        "total_statistics": stats,
        #"annotation_statistics": ann_stats,
    }


def structured_data_filters_section(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, t.Any]]:
    """
    Apply structured data filters - returns filtered dataframe and active filters.
    
    Args:
        df: Input dataframe
        filters: Optional filter dictionary (if None, uses default filter logic)
    
    Returns:
        Tuple of (filtered_dataframe, active_filters_dict)
    """
    # For API use, we need to apply filters programmatically
    # This is a simplified version - full implementation would handle all filter types
    if filters is None:
        # Use the existing filter display logic but capture results
        import streamlit as st
        if hasattr(st, 'session_state'):
            # Initialize temp filter values if needed
            if 'temp_filter_values' not in st.session_state:
                st.session_state.temp_filter_values = {}
            if 'active_filter_values' not in st.session_state:
                st.session_state.active_filter_values = {}
        
        structured_filtered_df, active_filters = structured_data_filters_display(df, debug=False)
    else:
        # Apply filters directly
        structured_filtered_df = df.copy()
        active_filters = {}
        
        for col, filter_value in filters.items():
            if col in structured_filtered_df.columns:
                if isinstance(filter_value, dict):
                    if "range" in filter_value:
                        # Numeric range filter
                        min_val, max_val = filter_value["range"]
                        numeric_col = pd.to_numeric(structured_filtered_df[col], errors="coerce")
                        mask = (numeric_col >= min_val) & (numeric_col <= max_val)
                        if filter_value.get("include_none", True):
                            mask |= structured_filtered_df[col].isna()
                        structured_filtered_df = structured_filtered_df[mask]
                        active_filters[col] = filter_value
                    elif "values" in filter_value:
                        # Categorical filter
                        structured_filtered_df = structured_filtered_df[structured_filtered_df[col].isin(filter_value["values"])]
                        active_filters[col] = filter_value
    
    return structured_filtered_df, active_filters


def embedding_data_filters_section(
    df: pd.DataFrame,
    structured_filtered_df: pd.DataFrame,
    embedding_selections: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply embedding data filters - returns filtered dataframe and embedding figures.
    
    Args:
        df: Original dataframe
        structured_filtered_df: Already filtered dataframe from structured filters
        embedding_selections: Optional embedding selection data
    
    Returns:
        Tuple of (filtered_dataframe, embedding_figures_dict)
    """
    embedding_figs = {}
    embedding_filtered_dfs = []
    
    # Get filtered dataframes for each embedding
    import streamlit as st
    for raw_data_key in META_INDEX_NAMES:
        if hasattr(st, 'session_state') and f"{raw_data_key}_reduced" not in st.session_state:
            continue
        
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


def data_distributions_section(filtered_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get data distribution visualizations - returns list of visualization dicts."""
    visualizations = generate_automatic_visualizations(filtered_df)
    
    # Convert to JSON-serializable format
    result = []
    for viz in visualizations:
        fig = viz.get("figure")
        result.append({
            "title": viz.get("title", ""),
            "figure": fig.to_dict() if hasattr(fig, "to_dict") else {},
        })
    
    return result


def success_rate_analytics_section(filtered_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get success rate analytics - returns list of visualization dicts."""
    success_visualizations = generate_success_rate_visualizations(filtered_df)
    
    result = []
    for viz in success_visualizations:
        fig = viz.get("figure")
        result.append({
            "title": viz.get("title", ""),
            "figure": fig.to_dict() if hasattr(fig, "to_dict") else {},
        })
    
    return result


def time_series_analytics_section(filtered_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get time series analytics - returns list of visualization dicts."""
    time_series_visualizations = generate_time_series_visualizations(
        filtered_df, time_column="ingestion_time"
    )
    
    result = []
    for viz in time_series_visualizations:
        fig = viz.get("figure")
        result.append({
            "title": viz.get("title", ""),
            "figure": fig.to_dict() if hasattr(fig, "to_dict") else {},
        })
    
    return result


def video_grid_section(filtered_df: pd.DataFrame, n_videos: int = 5) -> List[Dict[str, Any]]:
    """Get video grid data - returns list of video data dicts."""
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
        }
        # Add relevant fields
        for col in ['filename', 'episode_index', 'task_language_instruction', 'video_path']:
            if col in row and pd.notna(row[col]):
                video_data[col] = str(row[col])
        videos.append(video_data)
    
    return videos


def plot_hero_section(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    selected_row_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Get hero display data for selected row - returns dict instead of displaying."""
    import streamlit as st
    
    # Get or set selected row
    if selected_row_id:
        if 'id' in filtered_df.columns:
            selected_row = filtered_df[filtered_df['id'] == selected_row_id]
        else:
            selected_row = filtered_df.iloc[0:1] if len(filtered_df) > 0 else pd.DataFrame()
    else:
        # Use first row if none selected
        selected_row = filtered_df.iloc[0:1] if len(filtered_df) > 0 else pd.DataFrame()
    
    if len(selected_row) == 0:
        return None
    
    row = selected_row.iloc[0]
    
    # Get similar examples (simplified - would need full implementation)
    similar_examples = []
    if hasattr(st, 'session_state') and 'INDEX_MANAGER' in st.session_state:
        # This would use INDEX_MANAGER to find similar examples
        pass
    
    return {
        "selected_row": row.to_dict(),
        "similar_examples": similar_examples,
    }


def robot_array_section(selected_row_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get robot array plots - returns list of visualization dicts."""
    if selected_row_id is None:
        return []
    
    import streamlit as st
    df = _global_state.get("df")
    if df is None:
        return []
    
    # Find selected row
    if 'id' in df.columns:
        selected_row = df[df['id'] == selected_row_id]
    else:
        selected_row = df.iloc[0:1] if len(df) > 0 else pd.DataFrame()
    
    if len(selected_row) == 0:
        return []
    
    row = selected_row.iloc[0]
    all_vecs = _global_state.get("all_vecs", {})
    
    robot_array_visualizations = generate_robot_array_plot_visualizations(
        row,
        all_vecs,
        show_n=1000,
    )
    
    result = []
    for viz in robot_array_visualizations:
        fig = viz.get("figure")
        result.append({
            "title": viz.get("title", ""),
            "figure": fig.to_dict() if hasattr(fig, "to_dict") else {},
        })
    
    return result

