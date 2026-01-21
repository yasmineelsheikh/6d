"""
Simple helpers for inferring data types and generating visualizations. For example, a column of data containing a small number of unique strings can 
be inferred to be a bar chart, while a column of data containing a large number of unique floats can be inferred to be a histogram. We also choose
to *not* display certain columns that have poor visualizations, such as a column of data containing a large number of unique strings such as IDs or file paths.
"""

import re
import typing as t

import pandas as pd

from ares.app.plot_primitives import create_bar_plot, create_histogram
from ares.constants import IGNORE_COLS


def normalize_object_name(obj: str) -> str:
    """
    Normalize object names by:
    1. Converting to lowercase and stripping whitespace
    2. Removing trailing punctuation (periods, commas, etc.)
    3. Removing plural forms (simple heuristic: remove trailing 's' if word length > 3)
    4. Handling common plural patterns
    """
    if not obj or not isinstance(obj, str):
        return ""
    
    obj = obj.strip().lower()
    # Remove trailing punctuation (periods, commas, semicolons, etc.)
    obj = obj.rstrip('.,;:!?')
    
    # Common plural-to-singular mappings
    plural_to_singular = {
        "boxes": "box",
        "containers": "container",
        "items": "item",
        "objects": "object",
        "tools": "tool",
        "utensils": "utensil",
        "plates": "plate",
        "cups": "cup",
        "bottles": "bottle",
        "books": "book",
        "pens": "pen",
        "keys": "key",
        "phones": "phone",
        "laptops": "laptop",
        "chairs": "chair",
        "tables": "table",
        "bags": "bag",
        "shoes": "shoe",
        "clothes": "clothing",
    }
    
    # Check if it's a known plural
    if obj in plural_to_singular:
        return plural_to_singular[obj]
    
    # Simple heuristic: remove trailing 's' if word is longer than 3 characters
    # and doesn't end with 'ss', 'us', 'is', 'es' (which are often not plurals)
    if len(obj) > 3 and obj.endswith('s') and not obj.endswith(('ss', 'us', 'is', 'es')):
        return obj[:-1]
    
    return obj


def is_valid_object_name(obj: str) -> bool:
    """
    Check if a string represents a valid object name (not a phrase or description).
    
    Valid objects:
    - Single words or short compound words (e.g., "storage box")
    - Not phrases with verbs, prepositions, or descriptive clauses
    
    Invalid examples:
    - "smaller items in the background"
    - "items that are"
    - "objects present"
    - "things in the"
    """
    if not obj or not isinstance(obj, str):
        return False
    
    obj = obj.strip().lower()
    
    # Filter out empty strings
    if not obj:
        return False
    
    # Filter out common non-object phrases
    invalid_phrases = [
        "smaller items",
        "items in",
        "in the",
        "in background",
        "in the background",
        "that are",
        "present in",
        "objects present",
        "things in",
        "items that",
        "objects that",
        "small items",
        "large items",
        "other items",
        "various items",
        "different items",
        "additional items",
        "background items",
        "items on",
        "items near",
        "items around",
    ]
    
    # Check if the string contains any invalid phrase
    for phrase in invalid_phrases:
        if phrase in obj:
            return False
    
    # Filter out strings that are too long (likely phrases, not object names)
    # Most object names are 1-3 words
    words = obj.split()
    if len(words) > 3:
        return False
    
    # Filter out strings that contain common phrase indicators
    phrase_indicators = ["the", "a", "an", "some", "many", "few", "several", "various"]
    if any(word in phrase_indicators for word in words):
        return False
    
    return True


def normalize_distractor_objects(objects_str: str) -> str:
    """
    Normalize a comma-separated list of distractor objects by:
    1. Splitting by comma
    2. Normalizing each object name (plural -> singular)
    3. Filtering out invalid object names (phrases, descriptions)
    4. Deduplicating
    5. Joining back into comma-separated string
    """
    if not objects_str or not isinstance(objects_str, str):
        return ""
    
    # Split by comma and process each object
    objects = [obj.strip() for obj in objects_str.split(',')]
    
    # Normalize and filter objects
    normalized_objects = []
    seen = set()
    
    for obj in objects:
        if not obj:
            continue
        
        # Normalize the object name
        normalized = normalize_object_name(obj)
        
        # Check if it's a valid object name
        if not is_valid_object_name(normalized):
            continue
        
        # Deduplicate
        if normalized not in seen:
            normalized_objects.append(normalized)
            seen.add(normalized)
    
    return ", ".join(normalized_objects)


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
    environment: str | None = None,
    selected_axes: list[str] | None = None,
) -> list[dict]:
    """
    After inferring the 'type' of a column, we can create automatic visualizations.
    
    Args:
        df: DataFrame to visualize
        time_column: Column name for time-based filtering
        ignore_cols: Columns to ignore
        max_x_bar_options: Maximum number of options for bar charts
        environment: "Indoor" or "Outdoor" mode
        selected_axes: List of axis names that should be shown (e.g., ["Objects", "Lighting", "Materials"])
    """
    import plotly.graph_objects as go
    
    ignore_cols = ignore_cols or IGNORE_COLS
    visualizations = []
    
    # Handle empty dataframe - still create visualizations for selected axes
    is_empty_df = df.empty or len(df) == 0
    
    # Define column mappings based on environment
    # Indoor mappings
    indoor_mapping = {
        "Objects": "environment_distractor_objects_estimate",
        "Lighting": "environment_lighting_estimate",
        "Materials": "environment_surface_estimate",
    }
    
    # Outdoor mappings
    outdoor_mapping = {
        "Objects": "environment_distractor_objects_estimate",
        "Lighting": "environment_outdoor_lighting_estimate",
        "Weather": "environment_weather_estimate",
        "Road Surface": "environment_road_surface_estimate",
    }
    
    # Normalize axis names: "Color/Material" -> "Materials"
    axis_name_normalization = {
        "Color/Material": "Materials",
    }
    
    # Determine which mapping to use
    if environment == "Indoor":
        axis_to_column = indoor_mapping
    elif environment == "Outdoor":
        axis_to_column = outdoor_mapping
    else:
        # Default to indoor mapping if environment not specified
        axis_to_column = indoor_mapping
    
    # If selected_axes is provided, use only those axes
    # If None, default to all axes for the environment
    # If empty list, show no axes (user explicitly unchecked all)
    if selected_axes is None:
        # Default: show all axes for the environment when not specified
        axes_to_show = list(axis_to_column.keys())
    elif len(selected_axes) == 0:
        # User explicitly unchecked all axes - show nothing
        axes_to_show = []
    else:
        # Normalize axis names and remove duplicates
        normalized_axes = []
        seen_columns = set()
        for axis in selected_axes:
            # Normalize "Color/Material" to "Materials"
            normalized_axis = axis_name_normalization.get(axis, axis)
            # Only add if it maps to a valid column and we haven't seen this column yet
            if normalized_axis in axis_to_column:
                column = axis_to_column[normalized_axis]
                if column not in seen_columns:
                    normalized_axes.append(normalized_axis)
                    seen_columns.add(column)
        axes_to_show = normalized_axes
    
    # Create visualizations for each selected axis
    for axis_name in axes_to_show:
        if axis_name not in axis_to_column:
            continue
            
        column_name = axis_to_column[axis_name]
        
        # Check if column exists in dataframe or if dataframe is empty
        if column_name not in df.columns or is_empty_df:
            # Create empty plot if column doesn't exist or dataframe is empty
            fig = go.Figure()
            fig.update_layout(
                xaxis_title=axis_name,
                yaxis_title="Percentage of episodes (%)",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "sans-serif", "color": "white"},
                height=400,
            )
            fig.update_xaxes(showgrid=False, title_font={"color": "white"}, tickfont={"color": "white"})
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font={"color": "white"}, tickfont={"color": "white"})
            visualizations.append({
                "figure": fig,
                "title": axis_name,
            })
            continue
        
        # Infer visualization type
        viz_info = infer_visualization_type(column_name, df, ignore_cols)
        
        # Skip if column is in ignore list or is time column
        if column_name.lower() in ignore_cols or column_name == time_column:
            continue
        
        # Determine visualization type
        viz_type = viz_info["viz_type"]
        nunique = viz_info["nunique"]
        
        # Skip if too many unique values for bar chart
        if viz_type == "bar" and nunique > max_x_bar_options:
            # Still create empty plot to show the tab
            fig = go.Figure()
            fig.update_layout(
                xaxis_title=axis_name,
                yaxis_title="Percentage of episodes (%)",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "sans-serif", "color": "white"},
                height=400,
            )
            fig.update_xaxes(showgrid=False, title_font={"color": "white"}, tickfont={"color": "white"})
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font={"color": "white"}, tickfont={"color": "white"})
            visualizations.append({
                "figure": fig,
                "title": axis_name,
            })
            continue
        
        # Create histogram visualizations
        if viz_type == "histogram":
            fig = create_histogram(
                df,
                x=column_name,
                color="#1f77b4",
                title="",
                labels={column_name: axis_name, "count": "Percentage of episodes (%)"},
            )
            fig.update_traces(histnorm='percent')
            visualizations.append({
                "figure": fig,
                "title": axis_name,
            })
        
        # Create bar visualizations
        elif viz_type == "bar":
            # Create aggregation consistently for both boolean and non-boolean columns
            if pd.api.types.is_bool_dtype(df[column_name]):
                value_counts = df[column_name].astype(str).value_counts()
                total = len(df)
            else:
                # Split comma-separated values and count each individual value
                # Start from non-null values only
                non_null = df[column_name].dropna().astype(str)
                
                # Special handling for distractor_objects_estimate: normalize objects
                if column_name == "environment_distractor_objects_estimate":
                    # Normalize the entire string first, then split
                    normalized_series = non_null.apply(normalize_distractor_objects)
                    split_values = normalized_series.str.split(',').explode()
                else:
                    split_values = non_null.str.split(',').explode()
                
                # Normalize and clean individual tokens
                split_values = (
                    split_values
                    .astype(str)
                    .str.strip()
                    # Drop empty strings and explicit NaN/None-like markers
                    .loc[~split_values.str.fullmatch(r"", na=True)]
                )
                
                # For distractor_objects_estimate, apply additional normalization to individual tokens
                if column_name == "environment_distractor_objects_estimate":
                    # Normalize each token (plural -> singular) and filter invalid objects
                    normalized_tokens = []
                    for token in split_values:
                        normalized = normalize_object_name(token)
                        if is_valid_object_name(normalized):
                            normalized_tokens.append(normalized)
                    split_values = pd.Series(normalized_tokens, dtype=str)
                else:
                    # Remove common non-informative placeholders (None/none/NULL/nan/unknown)
                    invalid_tokens = {"none", "null", "nan", "na", "n/a", "unknown"}
                    split_values = split_values[~split_values.str.lower().isin(invalid_tokens)]
                
                if split_values.empty:
                    # Create empty plot if no valid values remain
                    fig = go.Figure()
                    fig.update_layout(
                        xaxis_title=axis_name,
                        yaxis_title="Percentage of episodes (%)",
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"family": "sans-serif", "color": "white"},
                        height=400,
                    )
                    fig.update_xaxes(showgrid=False, title_font={"color": "white"}, tickfont={"color": "white"})
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font={"color": "white"}, tickfont={"color": "white"})
                    visualizations.append({
                        "figure": fig,
                        "title": axis_name,
                    })
                    continue

                value_counts = split_values.value_counts()
                # Total is the number of valid (non-placeholder) values
                total = len(split_values)

            # Convert counts to percentages
            percentages = (value_counts / total * 100).round(2)
            
            agg_data = percentages.reset_index()
            agg_data.columns = [column_name, "percentage"]

            visualizations.append({
                "figure": create_bar_plot(
                    agg_data,
                    x=column_name,
                    y="percentage",
                    color="#1f77b4",
                    title="",
                    labels={column_name: axis_name, "percentage": "Percentage of episodes (%)"},
                ),
                "title": axis_name,
            })
        
        else:
            # Unknown visualization type - create empty plot
            fig = go.Figure()
            fig.update_layout(
                xaxis_title=axis_name,
                yaxis_title="Percentage of episodes (%)",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "sans-serif", "color": "white"},
                height=400,
            )
            fig.update_xaxes(showgrid=False, title_font={"color": "white"}, tickfont={"color": "white"})
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font={"color": "white"}, tickfont={"color": "white"})
            visualizations.append({
                "figure": fig,
                "title": axis_name,
            })
    
    return visualizations
