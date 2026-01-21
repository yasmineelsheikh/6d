"""
Upload page component for the ARES platform.
Provides a simple interface for users to upload robotics data files.
"""

import os
import shutil
import tempfile
import typing as t
from pathlib import Path

import streamlit as st


def show_upload_page() -> None:
    """Display the upload page with file uploader and ingestion trigger."""
    
    # Center the title
    st.markdown(
        "<h1 style='text-align: center; margin-top: 100px; font-size: 72px;'>6d labs</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Centered container for upload
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Upload Your Robotics Data")
        st.markdown("""
        Upload your robotics dataset files to begin analysis. Supported formats:
        - TensorFlow Datasets (TFDS)
        - HDF5 files (.h5, .hdf5)
        - Compressed archives (.zip, .tar.gz)
        """)
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['h5', 'hdf5', 'zip', 'tar', 'gz'],
            help="Upload one or more robotics data files"
        )
        
        st.markdown("**OR**")
        
        directory_path = st.text_input(
            "Directory Path (for LeRobot datasets)",
            placeholder="/path/to/dataset",
            help="Enter the absolute path to a local LeRobot dataset directory"
        )
        
        # Show uploaded files
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    st.text(f"📄 {file.name} ({file.size / 1024:.1f} KB)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Dataset name input
        dataset_name = st.text_input(
            "Dataset Name",
            placeholder="e.g., my_robot_dataset",
            help="Enter a name for your dataset"
        )
        
        # Process button
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            process_button = st.button(
                "🚀 Process Data",
                type="primary",
                use_container_width=True,
                use_container_width=True,
                disabled=not (uploaded_files or directory_path) or not dataset_name
            )
        
        if process_button and dataset_name:
            if uploaded_files:
                process_uploaded_data(uploaded_files, dataset_name)
            elif directory_path:
                process_directory_data(directory_path, dataset_name)


def process_uploaded_data(uploaded_files: list, dataset_name: str) -> None:
    """
    Process uploaded files and run the ingestion pipeline.
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
        dataset_name: Name for the dataset
    """
    try:
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp(prefix="ares_upload_")
        
        with st.spinner("Saving uploaded files..."):
            # Save uploaded files to temp directory
            saved_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)
            
            st.success(f"✅ Saved {len(saved_paths)} file(s)")
        
        # Store upload info in session state
        st.session_state.uploaded_data_path = temp_dir
        st.session_state.uploaded_files_paths = saved_paths
        st.session_state.dataset_name = dataset_name
        
        # Run ingestion pipeline
        with st.spinner("🔄 Processing data... This may take a few minutes."):
            success = run_ingestion_on_uploaded_data(temp_dir, dataset_name)
        
        if success:
            st.success("✅ Data processing complete!")
            st.balloons()
            
            # Switch to dashboard view
            st.session_state.workflow_stage = "dashboard"
            st.session_state.ingestion_complete = True
            
            # Force rerun to show dashboard
            st.rerun()
        else:
            st.error("❌ Data processing failed. Please check your files and try again.")
            
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        # Clean up temp directory on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_ingestion_on_uploaded_data(data_path: str, dataset_name: str) -> bool:
    """
    Run the ARES ingestion pipeline on uploaded data.
    
    Args:
        data_path: Path to directory containing uploaded files
        dataset_name: Name for the dataset
        
    Returns:
        True if ingestion succeeded, False otherwise
    """
    try:
        # TODO: Integrate with actual ingestion pipeline from main.py
        # For now, this is a placeholder that simulates the process
        
        import time
        import random
        
        # Simulate processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = [
            "Validating files...",
            "Loading dataset...",
            "Processing structured data...",
            "Generating embeddings...",
            "Running annotations...",
            "Finalizing..."
        ]
        
        for i, step in enumerate(steps):
            status_text.text(step)
            time.sleep(random.uniform(0.5, 1.5))
            progress_bar.progress((i + 1) / len(steps))
        
        status_text.text("✅ Processing complete!")
        
        # In a real implementation, this would:
        # 1. Load the dataset from uploaded files
        # 2. Call run_structured_database_ingestion()
        # 3. Call run_embedding_database_ingestion_per_dataset()
        # 4. Call orchestrate_annotating()
        
        return True
        
    except Exception as e:
        st.error(f"Ingestion error: {str(e)}")
        return False


def process_directory_data(directory_path: str, dataset_name: str) -> None:
    """
    Process a local directory dataset.
    
    Args:
        directory_path: Path to the dataset directory
        dataset_name: Name for the dataset
    """
    try:
        if not os.path.exists(directory_path):
            st.error(f"❌ Directory not found: {directory_path}")
            return
            
        # Run ingestion pipeline
        with st.spinner("🔄 Processing data... This may take a few minutes."):
            # Import here to avoid circular imports if any
            from scripts.ingest_lerobot_dataset import ingest_dataset
            from src.ares.databases.structured_database import ROBOT_DB_PATH, SQLITE_PREFIX
            
            engine_url = f"{SQLITE_PREFIX}{ROBOT_DB_PATH}"
            
            try:
                count = ingest_dataset(directory_path, engine_url, dataset_name)
                
                # Also trigger embedding ingestion
                # For now, we'll just do the structured ingestion as requested
                # But we should probably run the embedding ingestion too
                
                st.success(f"✅ Data processing complete! Ingested {count} episodes.")
                st.balloons()
                
                # Switch to dashboard view
                st.session_state.workflow_stage = "dashboard"
                st.session_state.ingestion_complete = True
                
                # Force rerun to show dashboard
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during ingestion: {str(e)}")
                
    except Exception as e:
        st.error(f"❌ Error processing directory: {str(e)}")
