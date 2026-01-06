"""
Helper functions for building the Streamlit app. This file exists to abstract out the code for building the app into smaller, more manageable components
and leave the main `ares.app.webapp.py` file clean and readable, exclusively for display, metrics, and export functionality.
"""

import typing as t

import pandas as pd
#import streamlit as st

from ares.app.data_analysis import generate_automatic_visualizations
from ares.app.filter_helpers import (
    create_embedding_data_filter_display,
    select_row_from_df_user,
    structured_data_filters_display,
)
from ares.app.hero_display import show_hero_display
from ares.app.init_data import display_state_info, initialize_data
from ares.app.plot_primitives import show_dataframe
from ares.app.viz_helpers import (
    annotation_statistics,
    create_tabbed_visualizations,
    display_video_grid,
    generate_robot_array_plot_visualizations,
    generate_success_rate_visualizations,
    generate_time_series_visualizations,
    total_statistics,
)
from ares.databases.embedding_database import META_INDEX_NAMES


def load_data(tmp_dump_dir: str) -> pd.DataFrame:
    """Load data from session state or initialize if needed."""
    initialize_data(tmp_dump_dir)
    return st.session_state.df


def loading_data_section(title: str, tmp_dump_dir: str) -> pd.DataFrame:
    # st.set_page_config moved to main webapp.py
    # st.title(title) # Title is handled in sidebar or main area
    return load_data(tmp_dump_dir)


def state_info_section(df: pd.DataFrame) -> None:
    display_state_info()
    total_statistics(df)
    annotation_statistics(st.session_state.annotations_db)


def structured_data_filters_section(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, t.Any]]:
    # st.header(f"Data Filters") # Handled in sidebar
    structured_filtered_df, active_filters = structured_data_filters_display(
        df, debug=False
    )
    st.write(
        f"Selected {len(structured_filtered_df)} rows out of {len(df)} total via structured data filters"
    )
    if len(structured_filtered_df) == 0:
        st.warning("No data matches the structured filters!")
    return structured_filtered_df, active_filters


def embedding_data_filters_section(
    df: pd.DataFrame,
    structured_filtered_df: pd.DataFrame,
) -> pd.DataFrame:
    # st.subheader(f"Unstructured Data Filters") # Handled in sidebar
    embedding_figs = dict()
    embedding_filtered_dfs = []

    # Get filtered dataframes for each embedding
    for raw_data_key in META_INDEX_NAMES:
        if f"{raw_data_key}_reduced" not in st.session_state:
            continue
            
        st.write(f"**Filtering on {raw_data_key.replace('_', ' ').title()}**")
        filtered_df, cluster_fig = create_embedding_data_filter_display(
            df=df,  # Pass original df each time
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

        # Final filtered dataframe combines structured and embedding filters
        filtered_df = structured_filtered_df[
            structured_filtered_df["id"].isin(all_filtered_ids)
        ]
    else:
        filtered_df = structured_filtered_df
    return filtered_df, embedding_figs


def data_distributions_section(filtered_df: pd.DataFrame) -> list[dict]:
    """Show dataset distribution with only Environment and Trajectory tabs."""
    import plotly.graph_objects as go
    import numpy as np
    
    # Create placeholder visualizations
    placeholder_visualizations = []
    
    # Environment placeholder - scatter plot
    env_fig = go.Figure()
    # Generate random scatter points
    np.random.seed(42)
    
    # Combine all points into a single trace without labels or colors
    x_vals = np.concatenate([np.random.randn(20) + np.random.uniform(-2, 2) for _ in range(4)])
    y_vals = np.concatenate([np.random.randn(20) + np.random.uniform(-2, 2) for _ in range(4)])
    
    env_fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        marker=dict(size=8, opacity=0.7),
        showlegend=False
    ))
    
    env_fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='',
        height=400,
        showlegend=False
    )
    placeholder_visualizations.append({
        "title": "Environment",
        "figure": env_fig
    })
    
    # Trajectory placeholder - multiple irregular lines
    traj_fig = go.Figure()
    np.random.seed(42)
    
    # Generate multiple irregular trajectory lines
    num_trajectories = 15
    for i in range(num_trajectories):
        # Create irregular trajectory with varying lengths
        length = np.random.randint(20, 80)
        x_vals = np.arange(length)
        # Create irregular y values with some noise and trends
        y_vals = np.cumsum(np.random.randn(length) * 0.5) + np.sin(x_vals / 5) * 2 + np.random.randn(length) * 0.3
        
        traj_fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(width=1.5),
            opacity=0.6,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    traj_fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='',
        height=400
    )
    placeholder_visualizations.append({
        "title": "Trajectory",
        "figure": traj_fig
    })
    
    # Create tabs with the two visualizations
    create_tabbed_visualizations(
        placeholder_visualizations, 
        ["Environment", "Trajectory"]
    )
    
    return placeholder_visualizations


def success_rate_analytics_section(filtered_df: pd.DataFrame) -> list[dict]:
    # st.header("Success Estimate Analytics")
    success_visualizations = generate_success_rate_visualizations(filtered_df)
    create_tabbed_visualizations(
        success_visualizations, [viz["title"] for viz in success_visualizations]
    )
    return success_visualizations


def time_series_analytics_section(filtered_df: pd.DataFrame) -> list[dict]:
    # st.header("Time Series Trends")
    time_series_visualizations = generate_time_series_visualizations(
        filtered_df, time_column="ingestion_time"
    )
    create_tabbed_visualizations(
        time_series_visualizations,
        [viz["title"] for viz in time_series_visualizations],
    )
    return time_series_visualizations


def video_grid_section(filtered_df: pd.DataFrame) -> None:
    # show video cards of 5 random episodes in a horizontal layout
    # st.header("Rollout Examples")
    n_videos = 5
    if len(filtered_df) > 0:
        # Remove duplicates based on episode ID or filename to ensure unique episodes
        if 'id' in filtered_df.columns:
            # Deduplicate by episode ID
            unique_df = filtered_df.drop_duplicates(subset=['id'], keep='first')
        elif 'filename' in filtered_df.columns:
            # Fall back to filename if ID not available
            unique_df = filtered_df.drop_duplicates(subset=['filename'], keep='first')
        else:
            # If neither available, use all columns for deduplication
            unique_df = filtered_df.drop_duplicates(keep='first')
        
        # Select 5 random episodes from unique episodes
        if len(unique_df) >= n_videos:
            display_rows = unique_df.sample(n=n_videos, random_state=None)
        else:
            display_rows = unique_df
        # Load videos directly (not lazy) and don't show info
        display_video_grid(display_rows, lazy_load=False, show_info=False)


def plot_hero_section(df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.Series:
    # st.header("Rollout Display")
    # initialize or persist selected row in state
    select_row_from_df_user(filtered_df)
    selected_row = st.session_state.get("selected_row")

    if selected_row is not None:
        show_dataframe(
            pd.DataFrame([selected_row]), title="Selected Row", add_refresh_button=False
        )
        st.write(f"Selected row ID: {selected_row.id}")
        show_hero_display(
            df,  # compare selected row from filtered_df to all rows in df
            selected_row,
            st.session_state.all_vecs,
            index_manager=st.session_state.INDEX_MANAGER,
            lazy_load=False,
            retrieve_n_most_similar=10,
        )
    else:
        st.info("Please select a row to display details")
    return selected_row


def robot_array_section(selected_row: pd.Series) -> list[dict]:
    if st.button("Generate Robot Array Plots", key="robot_array_plots_button"):
        st.header("Robot Array Display")
        # Number of trajectories to display in plots
        robot_array_visualizations = generate_robot_array_plot_visualizations(
            selected_row,  # need row to select dataset/robot embodiment of trajectories
            st.session_state.all_vecs,
            show_n=1000,
        )
    else:
        st.write("No robot array plots generated")
        robot_array_visualizations = []
    return robot_array_visualizations
