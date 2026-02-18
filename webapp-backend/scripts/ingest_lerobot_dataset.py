"""
Script to ingest LeRobot datasets into the ARES database.
Reads parquet files from the LeRobot format and converts them to ARES Rollout objects.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

# Database imports removed - step 3c (database record creation) is skipped
# from ares.configs.base import Environment, Robot, Rollout, Task, Trajectory
# from ares.databases.structured_database import (
#     RolloutSQLModel,
#     add_rollout,
#     setup_database,
# )


def load_metadata(dataset_path: Path) -> tuple[dict, pd.DataFrame]:
    """Load dataset metadata and task descriptions.
    
    If meta folder doesn't exist, generates minimal metadata by counting video files.
    """
    info_path = dataset_path / "meta/info.json"
    tasks_path = dataset_path / "meta/tasks.parquet"
    
    # Check if meta folder exists
    meta_folder = dataset_path / "meta"
    if not meta_folder.exists():
        print(f"No meta folder found at {meta_folder}. Generating minimal metadata from video files...")
        
        # Count video files
        videos_folder = dataset_path / "videos"
        video_count = 0
        
        if videos_folder.exists():
            # Count video files (episode_*.mp4 or any .mp4 files)
            video_files = list(videos_folder.glob("*.mp4"))
            video_count = len(video_files)
            print(f"Found {video_count} video files in {videos_folder}")
        else:
            print(f"No videos folder found at {videos_folder}")
        
        # Generate minimal info dict
        info = {
            "codebase_version": "unknown",
            "robot_type": "unknown",
            "total_episodes": video_count,
            "total_frames": 0,
            "total_tasks": 1,
            "total_videos": video_count,
            "fps": 30,
            "video_codec": "unknown",
            "generated_from_videos": True  # Flag to indicate this was generated
        }
        
        # Generate minimal tasks dataframe with one default task
        tasks_df = pd.DataFrame({
            "task_index": [0],
            "task": ["default_task"]
        })
        
        return info, tasks_df

    # Original logic when meta folder exists
    with open(info_path, "r") as f:
        info = json.load(f)

    tasks_df = pd.read_parquet(tasks_path)
    return info, tasks_df


def copy_episode_videos(dataset_path: Path, dataset_name: str):
    """Copy episode video clips from dataset's videos directory to ARES_VIDEO_DIR.
    
    Assumes episode clips are already present in {dataset_path}/videos/ as episode_{idx}.mp4
    and copies them to ARES_VIDEO_DIR/{dataset_name}/episode_{idx}.mp4
    """
    try:
        from ares.constants import ARES_VIDEO_DIR
        import shutil
    except ImportError:
        print("Required imports not available. Skipping video copying.")
        return
    
    videos_source_dir = dataset_path / "videos"
    if not videos_source_dir.exists():
        print(f"Videos directory not found: {videos_source_dir}")
        return
    
    # Create output directory
    output_dir = Path(ARES_VIDEO_DIR) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying episode videos from {videos_source_dir} to {output_dir}...")
    
    # Find all episode video files (episode_*.mp4)
    episode_videos = sorted(videos_source_dir.glob("episode_*.mp4"))
    
    if not episode_videos:
        print(f"No episode videos found in {videos_source_dir}")
        return
    
    copied_count = 0
    for video_file in episode_videos:
        # Extract episode index from filename (e.g., episode_0.mp4 -> 0)
        episode_name = video_file.stem  # episode_0
        output_path = output_dir / video_file.name  # episode_0.mp4
        
        # Skip if already exists
        if output_path.exists():
            continue
        
        try:
            # Copy video file
            shutil.copy2(video_file, output_path)
            copied_count += 1
            print(f"Copied {video_file.name} to {output_path}")
        except Exception as e:
            print(f"Error copying {video_file.name}: {e}")
    
    print(f"Copied {copied_count} episode videos to {output_dir}")


def process_chunk(
    chunk_path: Path,
    dataset_path: Path,
    info: dict,
    tasks_df: pd.DataFrame,
    engine,
    dataset_formalname: str,
) -> int:
    """Process a single chunk of parquet files.
    
    NOTE: Step 3c (database record creation) is skipped.
    This function only counts episodes and processes videos.
    No database records (Robot, Environment, Task, Trajectory, Rollout) are created.
    """
    count = 0
    for file_path in chunk_path.glob("*.parquet"):
        df = pd.read_parquet(file_path)
        
        # Group by episode_index
        for episode_idx, episode_df in tqdm(df.groupby("episode_index"), desc=f"Processing {file_path.name}"):
            # Sort by frame_index to ensure order
            episode_df = episode_df.sort_values("frame_index")
            
            # Step 3c SKIPPED: Database record creation
            # The following would normally be created but are skipped:
            # - Robot object
            # - Environment object  
            # - Task object
            # - Trajectory object
            # - Rollout object (with add_rollout to database)
            
            # Just count episodes for reporting
            count += 1
            
    return count


def ingest_dataset(data_dir: str, engine_url: str, dataset_name: str) -> int:
    """Ingest LeRobot dataset into ARES.
    
    NOTE: Step 3c (database record creation) is skipped.
    Copies episode videos from dataset's videos directory and counts episodes.
    No database records are created.
    
    Assumes episode clips are already present in {data_dir}/videos/ as episode_{idx}.mp4
    
    Returns:
        int: Number of episodes processed (not ingested into database)
    """
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
    # Step 3c SKIPPED: Database setup skipped since we're not creating records
    engine = None
    
    print(f"Loading metadata from {dataset_path}...")
    info, tasks_df = load_metadata(dataset_path)
    print(f"Found {len(tasks_df)} tasks.")
    
    data_path = dataset_path / "data"
    total_episodes = 0
    
    # Copy episode videos from dataset's videos directory to ARES_VIDEO_DIR
    copy_episode_videos(dataset_path, dataset_name)
    
    # Check if data folder exists
    if data_path.exists():
        # Iterate over chunks
        for chunk_path in data_path.glob("chunk-*"):
            if chunk_path.is_dir():
                print(f"Processing chunk: {chunk_path.name}")
                count = process_chunk(chunk_path, dataset_path, info, tasks_df, engine, dataset_name)
                total_episodes += count
    else:
        # No data folder - use episode count from metadata (generated from video files)
        print(f"No data folder found at {data_path}. Using episode count from metadata.")
        total_episodes = info.get("total_episodes", 0)
        print(f"Episode count from metadata: {total_episodes}")
    
    print(f"Ingestion complete. Total episodes processed: {total_episodes} (videos copied, database records skipped)")
    return total_episodes


@click.command()
@click.option("--data-dir", required=True, help="Path to the LeRobot dataset directory")
@click.option("--engine-url", required=True, help="SQLAlchemy database URL")
@click.option("--dataset-name", default="lerobot_dataset", help="Formal name for the dataset in ARES")
def main(data_dir: str, engine_url: str, dataset_name: str):
    ingest_dataset(data_dir, engine_url, dataset_name)


if __name__ == "__main__":
    main()
