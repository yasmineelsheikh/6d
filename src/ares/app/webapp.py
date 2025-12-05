"""
Main file for displaying the Streamlit app. This file contains the main function that defines the order of the sections in the app as well as
state management, error handling, timing, and data export functionality.
"""

import json
import os
import shutil
import boto3
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
import requests

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

# ---------- Pod API Config ----------
POD_API_URL = "https://g6hxoyusgab5l4-8000.proxy.runpod.net/run-json"  # Replace <pod-ip> with your pod's IP address

S3_BUCKET = "6d-temp-storage"
S3_REGION = "us-west-2"
s3 = boto3.client("s3", region_name=S3_REGION)

def upload_folder_to_s3(local_folder_path, bucket_name, s3_prefix='') -> str:
    """
    Uploads a local folder and its contents to an S3 bucket.

    Args:
        local_folder_path (str): The path to the local folder to upload.
        bucket_name (str): The name of the S3 bucket.
        s3_prefix (str): The S3 prefix (folder) to upload into. 
                         Defaults to an empty string for the bucket root.
    """
    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            
            # Construct the S3 key, maintaining the folder structure
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # Ensure forward slashes for S3 keys

            print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
            try:
                s3.upload_file(local_file_path, bucket_name, s3_key)
            except Exception as e:
                print(f"Error uploading {local_file_path}: {e}")

######################################################################
# Context managers for error handling and timing
######################################################################
@contextmanager
def error_context(section_name: str) -> t.Any:
    print(section_name)
    try:
        yield
    except Exception as e:
        st.error(f"Error in {section_name}: {str(e)}")
        st.stop()


@contextmanager
def timer_context(section_name: str) -> t.Any:
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        section_times[section_name] += elapsed_time


# ---------- Helper function to call pod ----------
def run_cosmos_on_pod(json_file_path: str) -> str:
    """
    Send JSON file to the pod and return the S3 URL of the output video.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    response = requests.post(POD_API_URL, json={"json_file_content": data})
    response.raise_for_status()
    
    s3_url = response.json().get("s3_url")
    if not s3_url:
        raise RuntimeError("Pod did not return an S3 URL")
    
    return s3_url


# ---------- Main Streamlit App ----------
def main() -> None:
    st.set_page_config(page_title=title, page_icon="📊", layout="wide")
    
    # Initialize session state for workflow management
    if "workflow_stage" not in st.session_state:
        from ares.databases.structured_database import setup_database, RolloutSQLModel
        from sqlalchemy import select
        
        try:
            engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
            query = select(RolloutSQLModel)
            df = pd.read_sql(query, engine)
            if len(df) > 0:
                st.session_state.workflow_stage = "dashboard"
            else:
                st.session_state.workflow_stage = "dashboard"
        except:
            st.session_state.workflow_stage = "dashboard"
            
    if "ingestion_complete" not in st.session_state:
        st.session_state.ingestion_complete = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Always show dashboard
    show_dashboard_page()


def process_local_dataset(dataset_path_str: str, dataset_name: str) -> None:
    try:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists():
            st.error(f"Path does not exist: {dataset_path_str}")
            return
            
        if not (dataset_path / "data").exists() or not (dataset_path / "meta").exists():
            st.error("Invalid LeRobot dataset structure. Directory must contain 'data/' and 'meta/' folders.")
            return
            
        with st.spinner("Processing dataset... This may take a few minutes."):
            try:
                import sys
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                from scripts.ingest_lerobot_dataset import ingest_dataset
                from ares.databases.structured_database import ROBOT_DB_PATH
                
                engine_url = ROBOT_DB_PATH
                count = ingest_dataset(str(dataset_path), engine_url, dataset_name)
                
                
                st.session_state.dataset_uploaded = True
                st.session_state.current_dataset = dataset_name
                st.session_state.dataset_path = str(dataset_path)
                st.session_state.active_tab = "original"
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")
                
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")


def load_dataset_info(dataset_path: str) -> dict | None:
    try:
        info_path = Path(dataset_path) / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def show_dashboard_page() -> None:
    st.markdown(
        """
        <div style='position: fixed; top: 10px; left: 20px; z-index: 999;'>
            <h3 style='margin: 0; color: #262730;'>6d labs</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
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
    
    # Original / Curated Tabs
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "original"
    
    tab_original, tab_curated = st.tabs(["Original", "Curated"])
    
    with tab_original:
        st.markdown("### Dataset Overview")
        
        df = pd.DataFrame()
        if 'current_dataset' in st.session_state and st.session_state.current_dataset:
            try:
                section_loading = "loading data"
                with error_context(section_loading), timer_context(section_loading):
                    df = loading_data_section(title, tmp_dump_dir)
                if 'dataset_formalname' in df.columns:
                    df = df[df['dataset_formalname'] == st.session_state.current_dataset]
                elif 'dataset_name' in df.columns:
                    df = df[df['dataset_name'] == st.session_state.current_dataset]
            except Exception:
                df = pd.DataFrame()
        
        if len(df) > 0:
            dataset_info = None
            if 'dataset_path' in st.session_state and st.session_state.dataset_path:
                dataset_info = load_dataset_info(st.session_state.dataset_path)
            
            col1, col2 = st.columns(2)
            with col1:
                if dataset_info and 'total_episodes' in dataset_info:
                    st.metric("Total Episodes", dataset_info['total_episodes'])
                else:
                    st.metric("Total Episodes", len(df))
            with col2:
                if dataset_info and 'robot_type' in dataset_info:
                    st.metric("Robot Type", dataset_info['robot_type'].replace('_', ' ').title())
                elif 'robot_embodiment' in df.columns and len(df) > 0:
                    st.metric("Robot Type", df['robot_embodiment'].iloc[0])
                else:
                    st.metric("Robot Type", "N/A")
            
            st.markdown("---")
            
            st.subheader("Dataset Distribution")
            section_display = "data distributions"
            with error_context(section_display), timer_context(section_display):
                data_distributions_section(df)
            
            st.markdown("---")
            
            st.subheader("Episode Preview")
            section_video_grid = "video grid"
            with error_context(section_video_grid), timer_context(section_video_grid):
                video_grid_section(df)
        
        st.markdown("---")
        st.markdown("### Dataset Processing")
        
        if len(df) > 0:
            tab_augmentation, tab_optimisation = st.tabs(["Augmentation", "Optimisation"])
            
            with tab_augmentation:
                text_input = st.text_input("Enter description")
                st.markdown("---")
                
            if st.button("Execute", key="execute_augmentation", type="primary", use_container_width=True):
                if text_input and st.session_state.get("current_dataset") and st.session_state.get("dataset_path"):
                    with st.spinner("Running data augmentation..."):
                        import tempfile
                        import json
                        import os
                        import sys
                        
                        # Add project root to path for imports
                        current_file = Path(__file__).resolve()
                        project_root = current_file.parent.parent.parent.parent
                        if str(project_root) not in sys.path:
                            sys.path.insert(0, str(project_root))
                        
                        from scripts.get_prompt import get_prompt
                        
                        dataset_name = st.session_state.get("current_dataset", "unknown")
                        dataset_path = st.session_state.get("dataset_path", "unknown")
                        try:
                            upload_folder_to_s3(dataset_path, S3_BUCKET, f"input_data/{dataset_name}")

                            prompt_path = get_prompt(text_input)
                            s3.upload_file(prompt_path, S3_BUCKET, f"input_data/{dataset_name}/prompt.txt")

                            # Call pod API
                            st.info("Starting Cosmos inference on pod...")
                            response = requests.post(
                                POD_API_URL,
                                json={"dataset_name": dataset_name},
                                timeout=3600  # 1 hour timeout for long-running inference
                            )
                            response.raise_for_status()
                            
                            result = response.json()
                            s3_output_url = result.get("s3_url")
                            
                            if not s3_output_url:
                                raise RuntimeError("Pod did not return an S3 URL")
                            
                            st.success("Cosmos processing complete!")
                            st.info(f"Output video uploaded to S3:\n{s3_output_url}")
                            
                            # Populate curated dataset with original dataset
                            if 'current_dataset' in st.session_state:
                                try:
                                    section_loading = "loading data"
                                    with error_context(section_loading), timer_context(section_loading):
                                        curated_df = loading_data_section(title, tmp_dump_dir)
                                    if 'dataset_formalname' in curated_df.columns:
                                        curated_df = curated_df[curated_df['dataset_formalname'] == st.session_state.current_dataset]
                                    elif 'dataset_name' in curated_df.columns:
                                        curated_df = curated_df[curated_df['dataset_name'] == st.session_state.current_dataset]
                                    st.session_state.curated_dataset = curated_df.copy()
                                    st.rerun()  # Refresh to show curated section
                                except Exception as e:
                                    st.warning(f"Could not load dataset for curated view: {e}")

                        except Exception as e:
                            st.error(f"Error running augmentation: {str(e)}")
                else:
                    st.warning("Please fill in all fields before executing.")
            
            with tab_optimisation:
                st.info("Configure optimisation parameters here.")
                st.markdown("---")
                if st.button("Execute", key="execute_optimisation", type="primary", use_container_width=True):
                    with st.spinner("Running dataset optimisation..."):
                        import time
                        time.sleep(2)
                    st.success("Optimisation complete!")
    
    with tab_curated:
        st.markdown("### Curated Dataset")
        
        if 'curated_dataset' in st.session_state and len(st.session_state.curated_dataset) > 0:
            curated_df = st.session_state.curated_dataset
            
            # Show same metrics as original
            dataset_info = None
            if 'dataset_path' in st.session_state and st.session_state.dataset_path:
                dataset_info = load_dataset_info(st.session_state.dataset_path)
            
            col1, col2 = st.columns(2)
            with col1:
                if dataset_info and 'total_episodes' in dataset_info:
                    st.metric("Total Episodes", dataset_info['total_episodes'])
                else:
                    st.metric("Total Episodes", len(curated_df))
            with col2:
                if dataset_info and 'robot_type' in dataset_info:
                    st.metric("Robot Type", dataset_info['robot_type'].replace('_', ' ').title())
                elif 'robot_embodiment' in curated_df.columns and len(curated_df) > 0:
                    st.metric("Robot Type", curated_df['robot_embodiment'].iloc[0])
                else:
                    st.metric("Robot Type", "N/A")
            
            st.markdown("---")
            
            st.subheader("Dataset Distribution")
            section_display = "data distributions"
            with error_context(section_display), timer_context(section_display):
                data_distributions_section(curated_df)
            
            st.markdown("---")
            
            st.subheader("Episode Preview")
            section_video_grid = "video grid"
            with error_context(section_video_grid), timer_context(section_video_grid):
                video_grid_section(curated_df)
        else:
            st.info("No curated dataset available. Run augmentation to generate curated data.")
        
        st.markdown("---")
        st.markdown("### Export Dataset")
        if 'curated_dataset' in st.session_state and len(st.session_state.curated_dataset) > 0:
            curated_df = st.session_state.curated_dataset
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                csv = curated_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="curated_dataset.csv",
                    mime="text/csv",
                    width="stretch"
                )


if __name__ == "__main__":
    main()
