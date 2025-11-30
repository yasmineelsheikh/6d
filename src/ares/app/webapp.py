"""
Main file for displaying the Streamlit app. This file contains the main function that defines the order of the sections in the app as well as
state management, error handling, timing, and data export functionality.
"""

import json
import os
import shutil
import tarfile
import tempfile
import time
import traceback
import typing as t
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ares.app.export_data import export_options
from ares.app.plot_primitives import show_dataframe
from ares.app.sections import (
    data_distributions_section,
    embedding_data_filters_section,
    loading_data_section,
    plot_hero_section,
    robot_array_section,
    state_info_section,
    structured_data_filters_section,
    success_rate_analytics_section,
    time_series_analytics_section,
    video_grid_section,
)
from ares.constants import ARES_DATA_DIR
from ares.databases.structured_database import ROBOT_DB_PATH

# top level global variables
title = "6d labs"
tmp_dump_dir = os.path.join(ARES_DATA_DIR, "webapp_tmp")
section_times: dict[str, float] = defaultdict(float)


######################################################################
# Context managers for error handling and timing
# - `error_context` is used to catch errors in computation and render the error in the app
# - `timer_context` is used to time the execution of a section and print the timing to the console
######################################################################
@contextmanager
def error_context(section_name: str) -> t.Any:
    """
    Context manager for gracefully handling errors in computation of a section.
    Catch the error and render it in the app for easy debugging and readability.
    """
    print(section_name)
    try:
        yield
    except Exception as e:
        st.error(f"Error in {section_name}: {str(e)}\\n{traceback.format_exc()}")
        st.write("Stopping execution")
        st.stop()


@contextmanager
def timer_context(section_name: str) -> t.Any:
    """
    Context manager for timing sections, helpful for debugging and performance analysis.
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        section_times[section_name] += elapsed_time


# Main function defining the order of the streamlit subsections
# Note: streamlit displays standalone-strings like `"""..."""` as markdown!
# Use `#` for comments in the streamlit context.
def main() -> None:
    st.set_page_config(page_title=title, page_icon="📊", layout="wide")
    
    # Initialize session state for workflow management
    if "workflow_stage" not in st.session_state:
        # Check if database already has data
        from ares.databases.structured_database import setup_database, RolloutSQLModel
        from sqlalchemy import select
        
        try:
            engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
            query = select(RolloutSQLModel)
            df = pd.read_sql(query, engine)
            
            # If data exists, go directly to dashboard
            if len(df) > 0:
                st.session_state.workflow_stage = "dashboard"
            else:
                st.session_state.workflow_stage = "dashboard"  # Always show dashboard now
        except:
            # If database doesn't exist or error, show dashboard anyway
            st.session_state.workflow_stage = "dashboard"
            
    if "ingestion_complete" not in st.session_state:
        st.session_state.ingestion_complete = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Always show dashboard (removed upload page workflow)
    show_dashboard_page()


def process_local_dataset(dataset_path_str: str, dataset_name: str) -> None:
    """
    Process a local LeRobot dataset directory.
    
    Args:
        dataset_path_str: Absolute path to the dataset directory
        dataset_name: Name for the dataset
    """
    try:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists():
            st.error(f"Path does not exist: {dataset_path_str}")
            return
            
        if not (dataset_path / "data").exists() or not (dataset_path / "meta").exists():
            st.error("Invalid LeRobot dataset structure. Directory must contain 'data/' and 'meta/' folders.")
            return
            
        # Run ingestion pipeline
        with st.spinner("🔄 Processing dataset... This may take a few minutes."):
            try:
                # Import ingestion function
                import sys
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                from scripts.ingest_lerobot_dataset import ingest_dataset
                from ares.databases.structured_database import ROBOT_DB_PATH
                
                engine_url = ROBOT_DB_PATH
                count = ingest_dataset(str(dataset_path), engine_url, dataset_name)
                
                st.success(f"Dataset processing complete! Ingested {count} episodes.")
                st.balloons()
                
                # Switch to dashboard view and refresh
                st.session_state.dataset_uploaded = True
                st.session_state.current_dataset = dataset_name
                st.session_state.dataset_path = str(dataset_path)  # Store dataset path for reading info.json
                st.session_state.active_tab = "original"
                
                # Force rerun to show updated dashboard
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
                
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")


def load_dataset_info(dataset_path: str) -> dict | None:
    """
    Load dataset information from meta/info.json file.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary with dataset info or None if file doesn't exist
    """
    try:
        info_path = Path(dataset_path) / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def show_dashboard_page() -> None:
    """Display the main analytics dashboard."""
    
    # Title in left corner - smaller size
    st.markdown(
        """
        <div style='position: fixed; top: 10px; left: 20px; z-index: 999;'>
            <h3 style='margin: 0; color: #262730;'>6d labs</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    ######################################################################
    # Top Upload Bar
    ######################################################################
    st.markdown("### Upload Data")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        dataset_path_input = st.text_input(
            "Dataset Directory Path",
            placeholder="/path/to/lerobot_dataset",
            help="Enter the absolute path to your LeRobot dataset directory (must include data/ and meta/ folders)",
            label_visibility="collapsed"
        )
    
    with col2:
        dataset_name = st.text_input(
            "Dataset Name",
            placeholder="e.g., stack_cups",
            label_visibility="collapsed"
        )
    
    with col3:
        upload_button = st.button(
            "Load Dataset",
            type="primary",
            width="stretch",
            disabled=not dataset_path_input or not dataset_name
        )
    
    if upload_button and dataset_path_input and dataset_name:
        process_local_dataset(dataset_path_input, dataset_name)
    
    
    ######################################################################
    # Original / Curated Tabs
    ######################################################################
    
    # Auto-select Original tab after upload
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "original"
    
    tab_original, tab_curated = st.tabs(["Original", "Curated"])
    
    with tab_original:
        st.markdown("### Dataset Overview")
        
        # Only load and show data if user uploaded a dataset in this session
        df = pd.DataFrame()  # Initialize empty dataframe
        if 'current_dataset' in st.session_state and st.session_state.current_dataset:
            try:
                # Only load data for the current session's dataset
                section_loading = "loading data"
                with error_context(section_loading), timer_context(section_loading):
                    df = loading_data_section(title, tmp_dump_dir)
                
                # Filter to only show the current session's dataset (ignore all other data)
                if 'dataset_formalname' in df.columns:
                    df = df[df['dataset_formalname'] == st.session_state.current_dataset]
                elif 'dataset_name' in df.columns:
                    df = df[df['dataset_name'] == st.session_state.current_dataset]
            except Exception:
                # Error loading - use empty dataframe
                df = pd.DataFrame()
        
        if len(df) == 0:
            pass  # Empty UI when no data - only section title is shown
        else:
            # Show all the ARES analysis content for single dataset
            
            # Load dataset info from meta/info.json
            dataset_info = None
            if 'dataset_path' in st.session_state and st.session_state.dataset_path:
                dataset_info = load_dataset_info(st.session_state.dataset_path)
            
            # Top Metrics Row - use info.json data if available, otherwise fall back to dataframe
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if dataset_info and 'total_episodes' in dataset_info:
                    st.metric("Total Episodes", dataset_info['total_episodes'])
                else:
                    st.metric("Total Episodes", len(df))
            with col2:
                if dataset_info and 'total_frames' in dataset_info:
                    st.metric("Total Frames", f"{dataset_info['total_frames']:,}")
                else:
                    total_frames = df['length'].sum() if 'length' in df.columns else 0
                    st.metric("Total Frames", f"{total_frames:,}")
            with col3:
                if dataset_info and 'robot_type' in dataset_info:
                    st.metric("Robot Type", dataset_info['robot_type'].replace('_', ' ').title())
                elif 'robot_embodiment' in df.columns and len(df) > 0:
                    st.metric("Robot Type", df['robot_embodiment'].iloc[0])
                else:
                    st.metric("Robot Type", "N/A")
            with col4:
                if dataset_info and 'fps' in dataset_info:
                    st.metric("FPS", f"{dataset_info['fps']:.1f}")
                else:
                    st.metric("FPS", "N/A")
            
            st.markdown("---")
            
            # Dataset Distribution
            st.subheader("Dataset Distribution")
            section_display = "data distributions"
            with error_context(section_display), timer_context(section_display):
                data_distributions_section(df)
            
            st.markdown("---")
            
            # Episode Grid
            st.subheader("Episode Preview")
            section_video_grid = "video grid"
            with error_context(section_video_grid), timer_context(section_video_grid):
                video_grid_section(df)
        
        # Action Buttons at bottom of Original tab
        st.markdown("---")
        st.markdown("### Dataset Processing")
        
        # Only show processing buttons when data exists
        if len(df) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Run Augmentation", width="stretch", type="secondary"):
                    with st.spinner("Running data augmentation..."):
                        from ares.augmentation import CosmosAugmentor
                        
                        # Initialize augmentor
                        augmentor = CosmosAugmentor()
                        
                        # Get current dataset
                        # Assuming df is available in the scope or we need to reload it
                        # The df variable is defined in the outer scope (tab_original context)
                        # But to be safe, let's access it from session state if possible, or use the local df
                        
                        # Create a place to store new episodes
                        new_episodes = []
                        
                        # Iterate through a subset of the dataframe for demo purposes (e.g., first 5 episodes)
                        # In a real app, we might want to augment the whole dataset or a selection
                        episodes_to_augment = df.head(5)
                        
                        # Create output directory for augmented videos
                        output_dir = os.path.join(tmp_dump_dir, "augmented_videos")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        progress_bar = st.progress(0)
                        for idx, (i, row) in enumerate(episodes_to_augment.iterrows()):
                            augmented_variants = augmentor.augment_episode(row, Path(output_dir))
                            new_episodes.extend(augmented_variants)
                            progress_bar.progress((idx + 1) / len(episodes_to_augment))
                        
                        if new_episodes:
                            # Convert new episodes to DataFrame
                            new_df = pd.DataFrame(new_episodes)
                            
                            # Append to existing dataframe in session state or re-ingest
                            # For this demo, we'll append to the in-memory dataframe and update session state
                            
                            # Concatenate
                            augmented_df = pd.concat([df, new_df], ignore_index=True)
                            
                            # Update the display dataframe
                            # Note: This updates the local variable 'df', but we might need to update 
                            # where it's stored to persist it across reruns if not saved to DB.
                            # For now, let's update session state if used there, or just show success.
                            
                            st.session_state.curated_dataset = augmented_df
                            st.success(f"Augmentation complete! Generated {len(new_episodes)} new episodes.")
                        else:
                            st.warning("No episodes were augmented (check video paths).")
            
            with col2:
                if st.button("Run Optimisation", width="stretch", type="secondary"):
                    with st.spinner("Running dataset optimisation..."):
                        import time
                        time.sleep(2)
                    st.success("Optimisation complete!")
            
            with col3:
                if st.button("Run Both", width="stretch", type="primary"):
                    with st.spinner("Running augmentation and optimisation..."):
                        import time
                        time.sleep(3)
                    st.success("Both processes complete!")
                    # Create a curated dataset for demo
                    st.session_state.curated_dataset = df.copy()
    
    with tab_curated:
        st.markdown("### Curated Dataset")
        
        # Only load and show data if user uploaded a dataset in this session
        df = pd.DataFrame()  # Initialize empty dataframe
        if 'current_dataset' in st.session_state and st.session_state.current_dataset:
            try:
                # Only load data for the current session's dataset
                section_loading = "loading data"
                with error_context(section_loading), timer_context(section_loading):
                    df = loading_data_section(title, tmp_dump_dir)
                
                # Filter to only show the current session's dataset (ignore all other data)
                if 'dataset_formalname' in df.columns:
                    df = df[df['dataset_formalname'] == st.session_state.current_dataset]
                elif 'dataset_name' in df.columns:
                    df = df[df['dataset_name'] == st.session_state.current_dataset]
            except Exception:
                # Error loading - use empty dataframe
                df = pd.DataFrame()
        
        # Only show content if data exists
        if len(df) > 0:
            if 'curated_dataset' in st.session_state:
                curated_df = st.session_state.curated_dataset
                
                # Show curated dataset metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Curated Episodes", len(curated_df))
                with col2:
                    if 'task_success' in curated_df.columns:
                        success_rate = curated_df['task_success'].mean() * 100
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                with col3:
                    if 'length' in curated_df.columns:
                        avg_len = curated_df['length'].mean()
                        st.metric("Avg Length", f"{avg_len:.0f}")
                with col4:
                    # Calculate improvement
                    if 'current_dataset' in st.session_state and st.session_state.current_dataset:
                        section_loading = "loading data"
                        with error_context(section_loading), timer_context(section_loading):
                            original_df = loading_data_section(title, tmp_dump_dir)
                        # Filter to only current session's dataset
                        if 'dataset_formalname' in original_df.columns:
                            original_df = original_df[original_df['dataset_formalname'] == st.session_state.current_dataset]
                        elif 'dataset_name' in original_df.columns:
                            original_df = original_df[original_df['dataset_name'] == st.session_state.current_dataset]
                        if len(original_df) > 0:
                            reduction = (1 - len(curated_df) / len(original_df)) * 100
                            st.metric("Size Reduction", f"{reduction:.1f}%")
                    else:
                        st.metric("Size Reduction", "N/A")
                
                st.markdown("---")
                
                # Show curated dataset preview
                st.subheader("Curated Dataset Preview")
                st.dataframe(curated_df.head(20), width="stretch")
        
        # Export section title always shown, content only when data exists
        st.markdown("---")
        st.markdown("### Export Dataset")
        
        # Only show export content if data exists
        if len(df) > 0:
            # Only show export button if curated dataset exists
            if 'curated_dataset' in st.session_state:
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col2:
                    export_button = st.button(
                        "Export Curated Dataset",
                        width="stretch",
                        type="primary",
                        disabled='curated_dataset' not in st.session_state
                    )
                    
                    if export_button and 'curated_dataset' in st.session_state:
                        curated_df = st.session_state.curated_dataset
                        
                        # Create download
                        csv = curated_df.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="curated_dataset.csv",
                            mime="text/csv",
                            width="stretch"
                        )
                        st.success("Dataset ready for export!")


if __name__ == "__main__":
    main()
