"""
Simple helpers for inferring data types and generating visualizations. For example, a column of data containing a small number of unique strings can 
be inferred to be a bar chart, while a column of data containing a large number of unique floats can be inferred to be a histogram. We also choose
to *not* display certain columns that have poor visualizations, such as a column of data containing a large number of unique strings such as IDs or file paths.
"""

import typing as t

import pandas as pd

from ares.app.plot_primitives import create_bar_plot, create_histogram
from ares.constants import IGNORE_COLS


def infer_visualization_type(
    column_name: str,
    data: pd.DataFrame,
    ignore_cols: list | None = None,
    max_str_length: int = 500,
) -> dict[str, t.Any]:
    """
    Heuristic solution for transforming a column of data into a visualization type,
    focusing on numeric ranges or category counts.
    """
    ignore_cols = ignore_cols or IGNORE_COLS

    dtype = str(data[column_name].dtype)
    nunique = data[column_name].nunique()

    result = {"viz_type": None, "dtype": dtype, "nunique": nunique}

    if column_name.lower() in ignore_cols:
        return result

    # Add special handling for boolean columns
    if pd.api.types.is_bool_dtype(data[column_name]):
        result["viz_type"] = "bar"
        return result

    if pd.api.types.is_string_dtype(data[column_name]):
        if data[column_name].str.len().max() > max_str_length:
            return result

    if pd.api.types.is_datetime64_any_dtype(data[column_name]):
        return result

    if pd.api.types.is_numeric_dtype(data[column_name]) or (
        dtype == "object"
        and len(data[column_name].dropna()) > 0
        and pd.to_numeric(data[column_name].dropna(), errors="coerce").notna().all()
    ):
        # check if lots of unique values or if it's a float between 0 and 1
        if nunique > 20 or (
            pd.api.types.is_float_dtype(data[column_name])
            and data[column_name].min() >= 0
            and data[column_name].max() <= 1
        ):
            result["viz_type"] = "histogram"
        else:
            result["viz_type"] = "bar"
        return result

    if pd.api.types.is_string_dtype(data[column_name]) or nunique < 20:
        result["viz_type"] = "bar"
        return result

    return result


def generate_automatic_visualizations(
    df: pd.DataFrame,
    time_column: str = "creation_time",
    ignore_cols: list[str] | None = None,
    max_x_bar_options: int = 100,
) -> list[dict]:
    """
    After inferring the 'type' of a column, we can create automatic visualizations.
    """
    ignore_cols = ignore_cols or IGNORE_COLS
    visualizations = []
    
    # Only include these specific columns for data distribution visualizations
    allowed_columns = [
        "environment_distractor_objects_estimate",
        "environment_lighting_estimate",
        "environment_surface_estimate",
    ]
    
    # Filter to only allowed columns that exist in the dataframe
    available_columns = [col for col in allowed_columns if col in df.columns]
    
    if not available_columns:
        return []

    # Pre-calculate visualization types for allowed columns only
    viz_infos = {
        col: infer_visualization_type(col, df)
        for col in sorted(available_columns)
        if col != time_column and col.lower() not in ignore_cols
    }

    # Group columns by visualization type
    histogram_cols = []
    bar_cols = []
    for col, info in viz_infos.items():
        if not info["nunique"] or (
            info["viz_type"] == "bar" and info["nunique"] > max_x_bar_options
        ):
            continue
        if info["viz_type"] == "histogram":
            histogram_cols.append(col)
        elif info["viz_type"] == "bar":
            bar_cols.append(col)

    # Custom title mapping for specific columns
    title_mapping = {
        "environment_distractor_objects_estimate": "Objects",
        "environment_lighting_estimate": "Lighting",
        "environment_surface_estimate": "Materials",
    }

    # Create histogram visualizations
    for col in histogram_cols:
        col_title = title_mapping.get(col, col.replace("_", " ").replace("-", " ").title())
        # Create histogram with percentage on y-axis
        fig = create_histogram(
            df,
            x=col,
            color="#1f77b4",
            title=col_title,
            labels={col: col_title, "count": "Percentage of episodes (%)"},
        )
        # Convert to percentages by normalizing
        fig.update_traces(histnorm='percent')
        visualizations.append(
            {
                "figure": fig,
                "title": col_title,
            }
        )

    # Create bar visualizations - handle each column separately
    for col in bar_cols:
        col_title = title_mapping.get(col, col.replace("_", " ").replace("-", " ").title())

        # Create aggregation consistently for both boolean and non-boolean columns
        if pd.api.types.is_bool_dtype(df[col]):
            value_counts = df[col].astype(str).value_counts()
            total = len(df)
        else:
            # Split comma-separated values and count each individual value
            # Handle NaN values by converting to empty string, then splitting
            split_values = df[col].astype(str).str.split(',').explode()
            # Strip whitespace from each split value
            split_values = split_values.str.strip()
            # Remove empty strings and 'nan' strings
            split_values = split_values[split_values != '']
            split_values = split_values[split_values.str.lower() != 'nan']
            value_counts = split_values.value_counts()
            # Total is the number of non-null values in the original column
            total = df[col].notna().sum()

        # Convert counts to percentages
        percentages = (value_counts / total * 100).round(2)
        
        agg_data = percentages.reset_index()
        agg_data.columns = [col, "percentage"]

        visualizations.append(
            {
                "figure": create_bar_plot(
                    agg_data,
                    x=col,
                    y="percentage",
                    color="#1f77b4",
                    title="",  # No title on graph, tab title is enough
                    labels={col: col_title, "percentage": "Percentage of episodes (%)"},
                ),
                "title": col_title,
            }
        )
    return visualizations
