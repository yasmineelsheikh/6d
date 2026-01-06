import io
import os
import random

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ares.utils.image_utils import get_video_frames, get_video_mp4


# Custom Color Palette
COLORS = [
    "#F63366",  # Primary Red
    "#FFFD80",  # Yellow
    "#00D4FF",  # Cyan
    "#FF4B4B",  # Red
    "#1C83E1",  # Blue
    "#803DFD",  # Purple
    "#00C0F2",  # Light Blue
]

def create_line_plot(
    df: pd.DataFrame,
    x: str,
    y: list[str],
    title: str,
    labels: dict[str, str],
    colors: list[str],
    y_format: str | None = None,
) -> plotly.graph_objects.Figure:
    # Use custom colors if not provided or if default
    if not colors or colors == ["#1f77b4"]:
        colors = COLORS

    fig = px.line(
        df,
        x=x,
        y=y,
        title=title,
        labels=labels,
        color_discrete_sequence=colors,
    )
    layout_args = {
        "yaxis_title": labels.get("value", "Value"),
        "showlegend": True,
        "legend_title_text": "",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "sans-serif", "color": "white"},
        "legend": {"font": {"color": "white"}},
    }
    if y_format:
        layout_args["yaxis_tickformat"] = y_format
    fig.update_layout(**layout_args)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font={"color": "white"}, tickfont={"color": "white"})
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font={"color": "white"}, tickfont={"color": "white"})
    return fig


def create_histogram(
    df: pd.DataFrame,
    x: str,
    title: str,
    labels: dict[str, str],
    color: str,
    nbins: int = 30,
) -> plotly.graph_objects.Figure:
    # Use primary color if default provided
    if color == "#2ecc71" or not color: # Assuming #2ecc71 was a default green used elsewhere
        color = COLORS[0]

    fig = px.histogram(
        df,
        x=x,
        nbins=nbins,
        title=title,
        labels=labels,
        color_discrete_sequence=[color],
        barmode="overlay",
        marginal="box",
    )
    fig.update_layout(
        xaxis_title=labels.get(x, x),
        yaxis_title="count",
        showlegend=False,
        bargap=0.1,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif", "color": "white"},
    )
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font={"color": "white"}, tickfont={"color": "white"})
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font={"color": "white"}, tickfont={"color": "white"})
    return fig


def create_bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    labels: dict[str, str],
    color: str,
) -> plotly.graph_objects.Figure:
    # Use primary color if default provided
    if color == "#2ecc71" or not color:
        color = COLORS[2] # Use a different color for bars

    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        labels=labels,
        color_discrete_sequence=[color],
    )
    
    # Set reasonable bar width (max 0.4 of category spacing - half of previous)
    # This prevents bars from being too wide when there are few categories
    num_categories = len(df[x].unique())
    if num_categories <= 5:
        # For few categories, set a fixed maximum bar width (half of previous)
        max_bar_width = 0.3
    else:
        # For many categories, use default spacing (half of previous)
        max_bar_width = 0.4
    
    fig.update_traces(width=max_bar_width)
    
    fig.update_layout(
        xaxis_title=labels.get(x, x),
        yaxis_title=labels.get(y, y),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif", "color": "white"},
        bargap=0.2,  # Add gap between bars
    )
    fig.update_xaxes(showgrid=False, title_font={"color": "white"}, tickfont={"color": "white"})
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font={"color": "white"}, tickfont={"color": "white"})
    return fig


def create_robot_array_plot(
    robot_array: np.ndarray,
    title_base: str,
    highlight_idx: int | None = None,
    show_n: int | None = None,
    scores: np.ndarray | None = None,
    colorscale: str = "RdYlGn",
    ids: list[str] | None = None,
) -> plotly.graph_objects.Figure:
    assert (
        robot_array.ndim == 3
    ), f"robot_array must have 3 dimensions; received shape: {robot_array.shape}"

    if show_n is not None:
        indices = np.arange(robot_array.shape[0])
        sampled_indices = np.random.choice(
            indices, min(show_n, len(indices), 1000), replace=False
        )

        if highlight_idx is not None:
            sampled_indices = np.unique(np.append(sampled_indices, highlight_idx))
            highlight_idx = np.where(sampled_indices == highlight_idx)[0][0]

        robot_array = robot_array[sampled_indices]
        if scores is not None:
            scores = scores[sampled_indices]
        if ids is not None:
            ids = [
                ids[i] for i in sampled_indices
            ]  # Update ids to match sampled indices

    # limit number of timesteps to 100 per trace
    if robot_array.shape[1] > 100:
        step = robot_array.shape[1] // 100
        robot_array = robot_array[:, ::step, :]
        if scores is not None:
            scores = scores[::step]

    # Create subplots - one for each dimension
    n_dims = robot_array.shape[2]

    fig = make_subplots(
        rows=n_dims,
        cols=1,
        subplot_titles=[f"Dimension {i+1}" for i in range(n_dims)],
        shared_xaxes=False,
        vertical_spacing=0.02,  # Reduce spacing between plots
        row_heights=[1 / n_dims] * n_dims,  # Ensure equal height distribution
    )
    fig.update_layout(
        title=title_base,
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )

    for dim in range(n_dims):
        dim_traces = []  # Create a new list for this dimension's traces
        if scores is not None:
            for i in range(len(robot_array)):
                if highlight_idx is not None and i == highlight_idx:
                    continue
                color = px.colors.sample_colorscale(colorscale, float(scores[i]))[0]
                dim_traces.append(
                    go.Scatter(
                        x=list(range(robot_array.shape[1])),
                        y=robot_array[i, :, dim],
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.5,
                        name=f"Score: {scores[i]:.2f}",
                        showlegend=i < 5
                        and dim == 0,  # Only show legend for first dimension
                        hovertemplate=f"Score: {scores[i]:.2f}<br>Value: %{{y}}<extra></extra>",
                    ),
                )
        else:
            mask = np.ones(robot_array.shape[0], dtype=bool)
            if highlight_idx is not None:
                mask[highlight_idx] = False

            x = np.tile(np.arange(robot_array.shape[1]), mask.sum())
            y = robot_array[mask, :, dim].flatten()
            masked_ids = (
                np.arange(robot_array.shape[0])[mask]
                if ids is None
                else [_id for i, _id in enumerate(ids) if mask[i]]
            )
            traj_ids = np.repeat(masked_ids, robot_array.shape[1])

            dim_traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=COLORS[4], width=1), # Use blue from palette
                    opacity=0.3,
                    name="Other Trajectories",
                    legendgroup="other",
                    showlegend=dim == 0,  # Only show legend for first dimension
                    hovertemplate="Trajectory %{customdata}<br>Value: %{y}<extra></extra>",
                    customdata=traj_ids,
                ),
            )

        if highlight_idx is not None and highlight_idx < robot_array.shape[0]:
            name = f"Trajectory {highlight_idx}" if ids is None else ids[highlight_idx]
            dim_traces.append(
                go.Scatter(
                    x=list(range(robot_array.shape[1])),
                    y=robot_array[highlight_idx, :, dim],
                    mode="lines",
                    name=name,
                    line=dict(color=COLORS[0], width=3), # Use primary red
                    opacity=1.0,
                    showlegend=dim == 0,  # Only show legend for first dimension
                    hovertemplate="Trajectory %{customdata}<br>Value: %{y}<extra></extra>",
                    customdata=[ids[highlight_idx]] if ids is not None else [name],
                ),
            )

        # Add traces for this dimension to its subplot
        fig.add_traces(dim_traces, rows=dim + 1, cols=1)

    # Update layout
    fig.update_layout(
        height=min(max(800, 300 * n_dims), 8000),
        yaxis_title="Value",
        hovermode="closest",
    )

    # Update x-axis titles: hide all except the bottom subplot
    for i in range(n_dims):
        if i < n_dims - 1:
            fig.update_xaxes(title_text="", row=i + 1, col=1)
        else:
            # Only show x-axis title for the bottom subplot
            fig.update_xaxes(title_text="Relative Timestep", row=i + 1, col=1)

    # Update y-axis properties for each subplot to maintain aspect ratio
    for i in range(n_dims):
        fig.update_yaxes(row=i + 1, col=1, automargin=True, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_xaxes(row=i + 1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig


def display_video_card(
    row: pd.Series,
    lazy_load: bool = False,
    key: str = "",
    extra_display_keys: list[str] | None = None,
    show_info: bool = True,
) -> None:
    if not pd.isna(row["path"]):
        try:
            dataset_filename, fname = (
                row["dataset_filename"],
                row["filename"],
            )
            # Get dataset path from row or session state
            dataset_path = None
            if "path" in row and pd.notna(row["path"]):
                dataset_path = row["path"]
            elif "dataset_path" in st.session_state:
                dataset_path = st.session_state.dataset_path
            
            if not lazy_load:
                st.video(get_video_mp4(dataset_filename, fname, dataset_path=dataset_path))
            else:
                # show placeholder image (along the same path), then button to load and play video
                frame = get_video_frames(dataset_filename, fname, n_frames=1, dataset_path=dataset_path)[0]
                st.image(frame)
                this_key = f"video_button_{row['id']}_{key}"
                persist_key = f"video_button_persist_{row['id']}_{key}"

                # handle persisting state for button
                if persist_key not in st.session_state:
                    st.session_state[persist_key] = False
                if st.button("Load Video", key=this_key):
                    st.session_state[persist_key] = True
                if st.session_state[persist_key]:
                    st.video(get_video_mp4(dataset_filename, fname, dataset_path=dataset_path))

            # Only show info if show_info is True
            if show_info:
                st.write(f"**{row['id']}**")
                task = row["task_language_instruction"]
                st.write(f"Task: {task if task else '(No task recorded)' }")
                st.write(f"Dataset: {row['dataset_formalname']}")
                if extra_display_keys:
                    with st.expander("Extra Info", expanded=False):
                        for key in extra_display_keys:
                            st.write(f"**{key.replace('_', ' ').title()}**: {row[key]}")
        except Exception as e:
            st.warning(f"Error loading video for {row['id']}: {e}")
    else:
        st.warning(f"Invalid video path for {row['id'], row['video_path']}")


def show_dataframe(
    df: pd.DataFrame,
    title: str,
    show_columns: list[str] | None = None,
    hide_columns: list[str] | None = None,
    add_refresh_button: bool = True,
) -> None:
    """Helper function to display DataFrames with consistent styling.

    Args:
        df: DataFrame to display
        title: Subheader title for the table
        show_columns: List of column names to show (exclusive with hide_columns)
        hide_columns: List of column names to hide (exclusive with show_columns)
    """
    if show_columns and hide_columns:
        raise ValueError("Cannot specify both show_columns and hide_columns")

    st.subheader(title)

    # Add a button to refresh the sample
    if add_refresh_button:
        st.button(
            "Get New Random Sample", key=f"refresh_sample_{title}"
        )  # Button press triggers streamlit rerun, triggers new random sample

    # Create copy and filter columns
    display_df = df.copy()
    if show_columns:
        display_df = display_df[show_columns]
    elif hide_columns:
        display_df = display_df.drop(hide_columns, axis=1)

    # Auto-generate column configs based on data types
    column_config = {}
    for col in display_df.columns:
        # Convert UUID columns to strings for pyarrow
        if col == "id":
            display_df[col] = display_df[col].astype(str)
            column_config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title()
            )
        elif pd.api.types.is_datetime64_any_dtype(display_df[col]):
            column_config[col] = st.column_config.DateColumn(
                col.replace("_", " ").title()
            )
        elif pd.api.types.is_numeric_dtype(display_df[col]):
            if "rate" in col.lower() or "percent" in col.lower():
                column_config[col] = st.column_config.NumberColumn(
                    col.replace("_", " ").title(), format="%.2%"
                )
            else:
                column_config[col] = st.column_config.NumberColumn(
                    col.replace("_", " ").title(), format="%g"
                )
        else:
            column_config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title()
            )

    st.dataframe(
        display_df,
        column_config=column_config,
        hide_index=True,
    )
