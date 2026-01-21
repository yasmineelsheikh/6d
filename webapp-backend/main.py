"""
FastAPI backend for the Ares platform web application.
Provides REST API endpoints for all data processing and visualization functionality.
"""

import json
import os
import sys
import tempfile
import importlib.util
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
import traceback

import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import requests
from dotenv import load_dotenv

load_dotenv()

# Add project root and src/ to path so we can import the ares package
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

for path in (project_root, src_dir):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Import auth utilities
from auth import (
    get_db, create_user, get_user_by_email,
    verify_password, create_access_token, decode_access_token, User
)

app = FastAPI(title="Ares Platform API", version="1.0.0")

# Security
security = HTTPBearer()

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Backend is running"}

# Tasks API Endpoints
class TaskRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    dataset_path: Optional[str] = ""
    prompt: Optional[str] = ""
    priority: Optional[str] = "medium"

class TaskResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    dataset_path: Optional[str] = ""
    prompt: Optional[str] = ""
    priority: Optional[str] = "medium"
    created_at: str

# In-memory task storage (in production, use a database)
tasks_storage: Dict[str, Dict[str, Any]] = {}

@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(task: TaskRequest):
    """Create a new task."""
    try:
        import uuid
        from datetime import datetime
        
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "name": task.name,
            "description": task.description or "",
            "dataset_path": task.dataset_path or "",
            "prompt": task.prompt or "",
            "priority": task.priority or "medium",
            "created_at": datetime.now().isoformat(),
        }
        
        tasks_storage[task_id] = task_data
        
        return TaskResponse(**task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")

@app.get("/api/tasks", response_model=List[TaskResponse])
async def get_tasks():
    """Get all tasks."""
    try:
        return [TaskResponse(**task) for task in tasks_storage.values()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tasks: {str(e)}")

@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get a specific task by ID."""
    try:
        if task_id not in tasks_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskResponse(**tasks_storage[task_id])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching task: {str(e)}")

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    try:
        if task_id not in tasks_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        del tasks_storage[task_id]
        return {"success": True, "message": "Task deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting task: {str(e)}")

# Settings API Endpoints
class SettingsRequest(BaseModel):
    api_endpoint: Optional[str] = None
    default_dataset_path: Optional[str] = None
    auto_save: Optional[bool] = None
    theme: Optional[str] = None
    max_episodes_display: Optional[int] = None
    vlm_provider: Optional[str] = None
    vlm_model: Optional[str] = None

class SettingsResponse(BaseModel):
    api_endpoint: Optional[str] = None
    default_dataset_path: Optional[str] = None
    auto_save: Optional[bool] = None
    theme: Optional[str] = None
    max_episodes_display: Optional[int] = None
    vlm_provider: Optional[str] = None
    vlm_model: Optional[str] = None

# In-memory settings storage (in production, use a database or config file)
settings_storage: Dict[str, Any] = {}

@app.post("/api/settings", response_model=SettingsResponse)
async def save_settings(settings: SettingsRequest):
    """Save application settings."""
    try:
        # Update settings storage with provided values
        settings_dict = settings.dict(exclude_unset=True)
        settings_storage.update(settings_dict)
        
        return SettingsResponse(**settings_storage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving settings: {str(e)}")

@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings():
    """Get current application settings."""
    try:
        return SettingsResponse(**settings_storage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching settings: {str(e)}")

# Authentication Models
class RegisterRequest(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user."""
    token = credentials.credentials
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        # Validate password length (bcrypt limit is 72 bytes)
        password_bytes = len(request.password.encode('utf-8'))
        if password_bytes > 72:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password is too long. Maximum length is 72 characters."
            )
        
        if len(request.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long."
            )
        
        # Check if user already exists
        if get_user_by_email(db, request.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user
        user = create_user(db, request.email, request.first_name, request.last_name, request.password)
        
        # Create access token
        access_token = create_access_token(data={"sub": user.id})
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                created_at=user.created_at.isoformat() if user.created_at else ""
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Registration error: {error_detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering user: {str(e)}"
        )

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Login and get access token."""
    try:
        user = get_user_by_email(db, request.email)
        if not user or not verify_password(request.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token = create_access_token(data={"sub": user.id})
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                created_at=user.created_at.isoformat() if user.created_at else ""
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error logging in: {str(e)}"
        )

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        created_at=current_user.created_at.isoformat() if current_user.created_at else ""
    )

@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint to verify backend is working."""
    return {"status": "ok", "message": "Backend API is working"}

@app.post("/api/database/clear")
async def clear_database():
    """Clear/reset the entire rollout database. Called when user starts a new task."""
    try:
        from ares.databases.structured_database import ROBOT_DB_PATH, RolloutSQLModel, setup_database
        from sqlalchemy import text

        # Clear the rollout table
        engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM rollout"))

        # Clear in-memory dataset paths
        dataset_paths.clear()
        ingestion_status.clear()  # Clear ingestion status tracking

        # Clear ARES global state cache so fresh data is loaded after ingestion
        try:
            from ares_api import _global_state
            _global_state.clear()
            print("ARES global state cache cleared")
        except Exception as cache_error:
            print(f"Warning: Could not clear ARES cache: {cache_error}")

        print("Database (rollout table) cleared and dataset paths reset")
        return {
            "success": True,
            "message": "Database cleared",
        }
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error clearing database: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

# CORS middleware - allow all origins for development
frontend_origins = [
    "https://6d-8jlk-git-main-yasmines-projects-3b23a4db.vercel.app/",
    "https://6d-8jlk.vercel.app/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configuration
POD_API_URL = "https://g6hxoyusgab5l4-8000.proxy.runpod.net/run-json"
S3_BUCKET = "6d-temp-storage"
S3_REGION = "us-west-2"
s3 = boto3.client("s3", region_name=S3_REGION)

# Supabase Storage configuration (for datasets and prompts).
# We talk directly to the Supabase Storage REST API using requests,
# instead of relying on the supabase-py client (which can be brittle
# in serverless environments).
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "datasets")
# Convenience flag: whether Supabase Storage is usable
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)

if not SUPABASE_ENABLED:
    print(
        "Warning: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set. "
        "Supabase Storage operations will not be available."
    )

# Use /tmp on Vercel/serverless, otherwise use home directory for local dev
if os.getenv("VERCEL") == "1" or os.getenv("LAMBDA_TASK_ROOT"):
    tmp_dump_dir = "/tmp/ares_webapp_tmp"
else:
    tmp_dump_dir = os.path.join(os.path.expanduser("~"), ".ares_webapp_tmp")

os.makedirs(tmp_dump_dir, exist_ok=True)

# Store dataset paths in memory (dataset_name -> dataset_path)
dataset_paths: Dict[str, str] = {}
# Track ingestion status per dataset: "in_progress", "complete", or None (not started)
ingestion_status: Dict[str, str] = {}

# Request/Response models
class DatasetLoadRequest(BaseModel):
    dataset_path: str
    dataset_name: str
    is_s3: Optional[bool] = False

class AugmentationRequest(BaseModel):
    dataset_name: str
    prompt: str
    task_description: Optional[str] = ""
    axes: Optional[List[str]] = None

class OptimizationRequest(BaseModel):
    dataset_name: str

class TestingUploadRequest(BaseModel):
    dataset_name: str
    test_directory: str

class DatasetInfo(BaseModel):
    total_episodes: int
    robot_type: str
    dataset_name: str

# Helper functions
def _ensure_supabase_storage_config() -> None:
    """Validate that Supabase Storage is configured."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase Storage is not configured. "
                   "Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
        )


def upload_file_to_supabase(file_path: str, file_content: bytes, bucket: str = "datasets") -> str:
    """Upload a file to Supabase Storage via REST API."""
    _ensure_supabase_storage_config()

    bucket_name = bucket or SUPABASE_STORAGE_BUCKET
    base_url = SUPABASE_URL.rstrip("/")
    # Ensure no leading slash in file_path
    object_path = file_path.lstrip("/")
    url = f"{base_url}/storage/v1/object/{bucket_name}/{object_path}"

    headers = {
        # Supabase Storage requires both Authorization and apikey headers
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": "application/octet-stream",
    }
    params = {"upsert": "true"}

    try:
        resp = requests.post(url, params=params, data=file_content, headers=headers, timeout=60)
        if not resp.ok:
            raise HTTPException(
                status_code=500,
                detail=f"Supabase Storage upload failed ({resp.status_code}): {resp.text}",
            )
        return object_path
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading to Supabase Storage: {str(e)}")


def download_file_from_supabase(file_path: str, bucket: str = "datasets") -> bytes:
    """Download a file from Supabase Storage via REST API."""
    _ensure_supabase_storage_config()

    bucket_name = bucket or SUPABASE_STORAGE_BUCKET
    base_url = SUPABASE_URL.rstrip("/")
    object_path = file_path.lstrip("/")
    url = f"{base_url}/storage/v1/object/{bucket_name}/{object_path}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
    }

    try:
        resp = requests.get(url, headers=headers, timeout=60)
        if not resp.ok:
            raise HTTPException(
                status_code=500,
                detail=f"Supabase Storage download failed ({resp.status_code}): {resp.text}",
            )
        return resp.content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading from Supabase Storage: {str(e)}")


def list_files_in_supabase(folder_path: str, bucket: str = "datasets") -> List[Dict[str, Any]]:
    """List all files in a Supabase Storage folder via REST API."""
    _ensure_supabase_storage_config()

    bucket_name = bucket or SUPABASE_STORAGE_BUCKET
    base_url = SUPABASE_URL.rstrip("/")
    url = f"{base_url}/storage/v1/object/list/{bucket_name}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": "application/json",
    }
    # Supabase Storage list endpoint expects a JSON body
    body = {
        "prefix": folder_path.rstrip("/") if folder_path else "",
        "limit": 1000,
    }

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=60)
        if not resp.ok:
            raise HTTPException(
                status_code=500,
                detail=f"Supabase Storage list failed ({resp.status_code}): {resp.text}",
            )
        return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing Supabase Storage files: {str(e)}")


def upload_folder_to_s3(local_folder_path: str, bucket_name: str, s3_prefix: str = "") -> None:
    """Upload all files in a folder to S3."""
    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            try:
                s3.upload_file(local_file_path, bucket_name, s3_key)
            except Exception as e:
                print(f"Error uploading {local_file_path}: {e}")

def load_dataset_info(dataset_path: str) -> Optional[Dict[str, Any]]:
    """Load dataset info from meta/info.json."""
    try:
        info_path = Path(dataset_path) / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def count_episodes_from_parquet(dataset_path: str) -> int:
    """Count total episodes from parquet files in the dataset."""
    try:
        data_path = Path(dataset_path) / "data"
        if not data_path.exists():
            return 0
        
        all_episode_indices = set()
        for chunk_path in data_path.glob("chunk-*"):
            if chunk_path.is_dir():
                for parquet_file in chunk_path.glob("*.parquet"):
                    try:
                        df = pd.read_parquet(parquet_file)
                        if 'episode_index' in df.columns:
                            # Get unique episode indices from this file
                            unique_episodes = df['episode_index'].unique()
                            all_episode_indices.update(unique_episodes)
                    except Exception as e:
                        print(f"Error reading {parquet_file}: {e}")
                        continue
        
        return len(all_episode_indices)
    except Exception as e:
        print(f"Error counting episodes: {e}")
        return 0

def run_ingestion_for_dataset(dataset_name: str, dataset_path: str, task: str = None):
    """
    Run the full ingestion pipeline for a dataset.
    This populates robot_data.db with rollouts from the dataset's own videos directory.
    """
    # Mark ingestion as in progress
    ingestion_status[dataset_name] = "in_progress"
    print(f"[INFO] Ingestion started for dataset: {dataset_name}")
    
    try:
        # Import necessary modules
        from ares.databases.structured_database import ROBOT_DB_PATH, RolloutSQLModel, setup_database
        from ares.models.shortcuts import get_nomic_embedder
        from ares.utils.image_utils import split_video_to_frames
        from sqlalchemy import text

        # Dynamically import run_ingestion_pipeline from the project root main.py
        main_path = project_root / "main.py"
        spec = importlib.util.spec_from_file_location("ares_ingestion_main", main_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load main.py from {main_path}")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        run_ingestion_pipeline = getattr(main_module, "run_ingestion_pipeline")
        from tqdm import tqdm

        # Initialize VLM, engine, and embedder
        vlm_name = "gpt-4o"
        #vlm_name = "gpt-4o-mini"
        #vlm_name = "claude-3-5-sonnet"
        engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
        
        # Clear the database for this dataset to ensure clean ingestion
        print(f"Clearing existing rollouts for dataset '{dataset_name}' from database...")
        with engine.begin() as conn:
            # Delete rollouts for this specific dataset
            conn.execute(
                text("DELETE FROM rollout WHERE dataset_filename = :dataset_name"),
                {"dataset_name": dataset_name}
            )
        print(f"Database cleared for dataset '{dataset_name}'")
        
        embedder = get_nomic_embedder()

        # Use dataset_name as task if not provided
        if task is None:
            task = dataset_name

        dataset_path_obj = Path(dataset_path)

        # Load dataset info from meta/info.json if available
        dataset_info_dict = load_dataset_info(dataset_path)

        # Create full_dataset_info dict
        dataset_formalname = dataset_info_dict.get("dataset_formalname", dataset_name) if dataset_info_dict else dataset_name
        split_name = dataset_info_dict.get("split", "test") if dataset_info_dict else "test"
        full_dataset_info = {
            "Dataset": dataset_formalname,
            "Dataset Filename": dataset_name,
            "Dataset Formalname": dataset_formalname,
            "Split": split_name,
            "Robot": dataset_info_dict.get("robot_type") if dataset_info_dict else None,
            "Robot Morphology": None,
            "Gripper": None,
            "Action Space": None,
            "# RGB Cams": None,
            "# Depth Cams": None,
            "# Wrist Cams": None,
            "Language Annotations": "Natural",
            "Data Collect Method": "Expert Policy",
            "Scene Type": None,
            "Citation": "year={2024}",
        }

        # Define prep_for_oxe_episode function
        def prep_for_oxe_episode(task: str, episode_filename: str):
            """Force the videos and task information into the OpenXEmbodimentEpisode format."""
            filename = episode_filename
            try:
                frames = split_video_to_frames(filename)
            except Exception as e:
                print(f"Error getting video frames for {filename}: {e}")
                return None
            metadata = {"file_path": filename}
            steps = []
            for i, frame in enumerate(frames):
                # Observation should contain the image, not at the step level
                observation = {"image": frame}
                steps.append({
                    "action": None,
                    "is_first": i == 0,
                    "is_last": i == len(frames) - 1,
                    "is_terminal": False,
                    "language_embedding": None,
                    "language_instruction": task,
                    "observation": observation,
                })
            return {"episode_metadata": metadata, "steps": steps}

        # Create DemoIngestion class that iterates through videos in the dataset's own videos directory
        class DemoIngestion:
            def __init__(self, task: str, videos_dir: Path):
                self.task = task
                self._episodes = []

                if not videos_dir.exists():
                    print(f"Warning: Video directory {videos_dir} does not exist")
                    return

                # Find all episode video files in the directory
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
                video_files = sorted([
                    f for f in videos_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in video_extensions
                ])

                # Iterate through actual video files
                for video_path in tqdm(video_files, desc=f"Loading episodes from {videos_dir}"):
                    episode = prep_for_oxe_episode(task, str(video_path))
                    if episode is not None:
                        self._episodes.append(episode)
                self._index = 0

            def __iter__(self):
                self._index = 0
                return self

            def __next__(self):
                if self._index >= len(self._episodes):
                    raise StopIteration
                episode = self._episodes[self._index]
                self._index += 1
                return episode

            def __len__(self):
                return len(self._episodes)

        videos_dir = dataset_path_obj / "videos"
        ds = DemoIngestion(task, videos_dir)

        if len(ds) == 0:
            print(f"Warning: No episodes found for dataset {dataset_name} in {videos_dir}")
            return

        # Run the ingestion pipeline
        print(f"Running ingestion pipeline for {dataset_name} from {videos_dir} ...")
        run_ingestion_pipeline(
            ds,
            full_dataset_info,
            dataset_formalname,
            vlm_name,
            engine,
            dataset_name,
            embedder,
            split_name,
        )
        print(f"Ingestion pipeline completed for {dataset_name}")
        
        # Mark ingestion as complete BEFORE generating distributions
        ingestion_status[dataset_name] = "complete"
        print(f"[INFO] Ingestion completed for dataset: {dataset_name}")
        
        # Invalidate ARES cache so next API call reloads fresh data
        try:
            from ares_api import _global_state, generate_and_cache_distributions
            # Clear the dataframe and related caches, but keep ENGINE/SESSION
            keys_to_clear = ["df", "INDEX_MANAGER", "all_vecs", "all_ids", "annotations_db", "annotation_db_stats"]
            # Also clear plot cache since we'll regenerate it
            plot_cache_keys = [key for key in _global_state.keys() if key.startswith("plot_cache_")]
            keys_to_clear.extend(plot_cache_keys)
            for key in keys_to_clear:
                if key in _global_state:
                    del _global_state[key]
            print(f"ARES cache invalidated after ingestion completion")
            
            # Generate plots for default Indoor environment after ingestion completes
            # This ensures plots are ready immediately, but they're always generated fresh (use_cache=False)
            print(f"[INGESTION] Generating distribution plots for {dataset_name} after ingestion...")
            try:
                from ares_api import get_dataframe, get_data_distributions
                df = get_dataframe()
                if not df.empty:
                    # Generate Indoor plots with default axes (fresh, no cache)
                    # Use "Color/Material" to match frontend naming, which will be normalized to "Materials"
                    indoor_axes = ["Objects", "Lighting", "Color/Material"]
                    print(f"[INGESTION] Proactive generation with axes: {indoor_axes}")
                    indoor_plots = get_data_distributions(
                        df, 
                        environment="Indoor", 
                        selected_axes=indoor_axes, 
                        use_cache=False
                    )
                    print(f"[INGESTION] Generated {len(indoor_plots)} Indoor plots after ingestion")
                else:
                    print("DataFrame is empty, skipping plot generation")
            except Exception as plot_error:
                print(f"Warning: Could not generate plots after ingestion: {plot_error}")
                traceback.print_exc()
        except Exception as cache_error:
            print(f"Warning: Could not invalidate ARES cache or generate plots: {cache_error}")
            traceback.print_exc()

    except Exception as e:
        # Mark ingestion as failed
        ingestion_status[dataset_name] = "failed"
        error_trace = traceback.format_exc()
        print(f"Error running ingestion pipeline for {dataset_name}: {error_trace}")
        raise

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Ares Platform API", "version": "1.0.0"}

def download_from_s3(s3_path: str, local_path: Path) -> Path:
    """Download a folder from S3 to a local path."""
    try:
        # Parse S3 path: s3://bucket-name/path/to/folder
        if not s3_path.startswith("s3://"):
            raise ValueError("S3 path must start with 's3://'")
        
        s3_path = s3_path[5:]  # Remove 's3://'
        parts = s3_path.split("/", 1)
        bucket_name = parts[0]
        s3_prefix = parts[1] if len(parts) > 1 else ""
        
        # Create local directory
        local_path.mkdir(parents=True, exist_ok=True)
        
        # List and download all objects with the prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
        
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                s3_key = obj['Key']
                # Get relative path from prefix
                if s3_prefix:
                    relative_path = s3_key[len(s3_prefix):].lstrip('/')
                else:
                    relative_path = s3_key
                
                if not relative_path:
                    continue
                
                local_file_path = local_path / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                s3.download_file(bucket_name, s3_key, str(local_file_path))
        
        return local_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading from S3: {str(e)}")

@app.post("/api/datasets/upload")
async def upload_dataset(
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(...),
    environment: str = Form(""),
    axes: str = Form("[]")
):
    """Upload a dataset folder from local files."""
    try:
        print(f"Received upload request for dataset: {dataset_name}")
        print(f"Number of files received: {len(files) if files else 0}")
        
        if not files or len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="No files were uploaded. Please select a folder to upload."
            )
        
        # Use Supabase Storage if available, otherwise fall back to local filesystem.
        use_supabase = SUPABASE_ENABLED
        upload_dir = None
        
        if not use_supabase:
            # Fallback to local filesystem
            if os.getenv("VERCEL") == "1" or os.getenv("LAMBDA_TASK_ROOT"):
                upload_dir = Path("/tmp") / "uploaded_datasets" / dataset_name
            else:
                upload_dir = project_root / "data" / "uploaded_datasets" / dataset_name
            upload_dir.mkdir(parents=True, exist_ok=True)
            print(f"Using local filesystem, upload directory: {upload_dir}")
        else:
            print(f"Using Supabase Storage for dataset: {dataset_name}")
        
        # Detect root folder name from first file's path
        # webkitRelativePath includes the root folder name (e.g., "my_dataset/data/file.parquet")
        # We need to strip it so files are saved directly under dataset_name
        root_folder_prefix = None
        first_file_path = None
        for file in files:
            if file.filename:
                first_file_path = file.filename.replace("\\", "/")
                # Extract root folder name (first part before first slash)
                if "/" in first_file_path:
                    root_folder_prefix = first_file_path.split("/")[0]
                    print(f"Detected root folder prefix: {root_folder_prefix}")
                break
        
        # Save all uploaded files preserving directory structure
        files_saved = 0
        for idx, file in enumerate(files):
            try:
                # Get the relative path from the file's path (if available)
                file_path = file.filename
                if not file_path:
                    print(f"File {idx} has no filename, skipping")
                    continue
                
                print(f"Processing file {idx + 1}/{len(files)}: {file_path}")
                
                # Normalize the path (handle both forward and backward slashes)
                file_path = file_path.replace("\\", "/")
                
                # Strip root folder prefix if it exists
                if root_folder_prefix and file_path.startswith(root_folder_prefix + "/"):
                    file_path = file_path[len(root_folder_prefix) + 1:]
                    print(f"Stripped root folder prefix, new path: {file_path}")
                
                # Read file content
                content = await file.read()
                
                if use_supabase:
                    # Upload to Supabase Storage
                    supabase_path = f"{dataset_name}/{file_path}"
                    upload_file_to_supabase(supabase_path, content, bucket="datasets")
                    print(f"Uploaded to Supabase Storage: {supabase_path}")
                else:
                    # Save to local filesystem
                    full_path = upload_dir / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(full_path, "wb") as f:
                        f.write(content)
                    print(f"Saved file: {full_path}")
                
                files_saved += 1
            except Exception as file_error:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Error saving file {file.filename if file.filename else 'unknown'}: {error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error saving file {file.filename if file.filename else 'unknown'}: {str(file_error)}"
                )
        
        if files_saved == 0:
            raise HTTPException(
                status_code=400,
                detail="No files were uploaded or saved."
            )
        
        # Validate dataset structure
        if not (upload_dir / "data").exists() or not (upload_dir / "meta").exists() or not (upload_dir / "videos").exists():
            raise HTTPException(
                status_code=400,
                detail="Invalid LeRobot dataset structure. Directory must contain 'data/', 'meta/', and 'videos/' folders."
            )
        
        # Run basic ingestion (no database creation)
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from scripts.ingest_lerobot_dataset import ingest_dataset
        count = ingest_dataset(str(upload_dir), None, dataset_name)
        
        # Store the dataset path for later use
        dataset_paths[dataset_name] = str(upload_dir)
        
        # Run the full ingestion pipeline to populate robot_data.db in the background.
        # This is best-effort for plots; core dataset info and previews do not depend on it.
        print(f"Starting background ingestion pipeline for uploaded dataset: {dataset_name}")
        try:
            loop = asyncio.get_running_loop()
            # Fire-and-forget: do not await, so upload can succeed regardless of ingestion outcome.
            loop.run_in_executor(None, run_ingestion_for_dataset, dataset_name, str(upload_dir), dataset_name)
        except Exception as ingestion_error:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error scheduling ingestion for {dataset_name}: {error_trace}")
        
        return {
            "success": True,
            "dataset_name": dataset_name,
            "dataset_path": str(upload_dir),
            "episodes_ingested": count
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error uploading dataset: {error_trace}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error uploading dataset: {str(e)}"}
        )

@app.post("/api/datasets/load")
async def load_dataset(request: DatasetLoadRequest):
    """Load a dataset from a local path or S3."""
    try:
        if request.is_s3:
            # Download from S3 to temporary directory
            temp_dir = project_root / "data" / "temp_datasets" / request.dataset_name
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_path = download_from_s3(request.dataset_path, temp_dir)
        else:
            dataset_path = Path(request.dataset_path)
            if not dataset_path.exists():
                raise HTTPException(status_code=400, detail=f"Path does not exist: {request.dataset_path}")
        
        if not (dataset_path / "data").exists() or not (dataset_path / "meta").exists() or not (dataset_path / "videos").exists():
            raise HTTPException(
                status_code=400,
                detail="Invalid LeRobot dataset structure. Directory must contain 'data/', 'meta/', and 'videos/' folders."
            )
        
        # Run basic ingestion (no database creation)
        # Ensure project root is in path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from scripts.ingest_lerobot_dataset import ingest_dataset
        # Using None for engine_url since database creation is skipped
        count = ingest_dataset(str(dataset_path), None, request.dataset_name)
        
        # Store the dataset path for later use
        dataset_paths[request.dataset_name] = str(dataset_path)
        
        # Run the full ingestion pipeline to populate robot_data.db in the background.
        # This is best-effort for plots; core dataset info and previews do not depend on it.
        print(f"Starting background ingestion pipeline for loaded dataset: {request.dataset_name}")
        try:
            loop = asyncio.get_running_loop()
            # Fire-and-forget: do not await, so load can succeed regardless of ingestion outcome.
            loop.run_in_executor(None, run_ingestion_for_dataset, request.dataset_name, str(dataset_path), request.dataset_name)
        except Exception as ingestion_error:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error scheduling ingestion for {request.dataset_name}: {error_trace}")
        
        return {
            "success": True,
            "dataset_name": request.dataset_name,
            "dataset_path": str(dataset_path),
            "episodes_ingested": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

@app.get("/api/datasets/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    """Get dataset information and metrics."""
    try:
        # Check if dataset path is stored
        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        dataset_path = dataset_paths[dataset_name]
        dataset_info = load_dataset_info(dataset_path)
        
        # Get episode count from info.json first, fallback to counting parquet files
        if dataset_info and 'total_episodes' in dataset_info:
            total_episodes = dataset_info['total_episodes']
        else:
            # Only count from parquet if not in info.json
            total_episodes = count_episodes_from_parquet(dataset_path)
        
        robot_type = "N/A"
        if dataset_info and 'robot_type' in dataset_info:
            robot_type = dataset_info['robot_type']
        
        return {
            "dataset_name": dataset_name,
            "total_episodes": total_episodes,
            "robot_type": robot_type
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset info: {str(e)}")

@app.get("/api/datasets/{dataset_name}/data")
async def get_dataset_data(dataset_name: str):
    """Get dataset data as JSON."""
    try:
        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        # Base dataset path
        dataset_path = dataset_paths[dataset_name]
        dataset_path_obj = Path(dataset_path)

        # Videos directory: each video file is treated as one episode
        videos_dir = dataset_path_obj / "videos"

        if not videos_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Videos directory not found for dataset '{dataset_name}'"
            )

        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
        video_files = sorted(
            f for f in videos_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        )

        episodes = []
        for idx, video_path in enumerate(video_files):
            episodes.append({
                "id": f"episode_{idx}",
                "episode_index": idx,
                "length": None,  # length (timesteps) not derived here
                "video_url": f"/api/datasets/{dataset_name}/videos/{idx}",
                "task_language_instruction": None,  # not currently mapped from parquet
            })

        return {
            "dataset_name": dataset_name,
            "data": episodes,
            "count": len(episodes)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset data: {str(e)}")

@app.get("/api/datasets/{dataset_name}/distributions")
async def get_dataset_distributions(dataset_name: str):
    """Get dataset distribution visualizations."""
    try:
        # Ares database access removed - return placeholder data
        # The actual visualization generation is done client-side
        return {
            "dataset_name": dataset_name,
            "distributions": {
                "environment": {
                    "type": "scatter",
                    "data": []
                },
                "trajectory": {
                    "type": "line",
                    "data": []
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting distributions: {str(e)}")

@app.post("/api/augmentation/run")
async def run_augmentation(request: AugmentationRequest):
    """Run Cosmos augmentation on a dataset."""
    try:
        import uuid
        from datetime import datetime
        from generate_prompt import generate_prompt_variations, generate_video_descriptions_with_vlm
        from ares.models.shortcuts import get_vlm
        
        # Get VLM instance (same as used in ingestion)
        # Using Gemini instead of OpenAI to avoid quota limits
        vlm_name = "gpt-4o"  # Changed from "gpt-4o" to match ingestion and avoid quota limits
        vlm = get_vlm(vlm_name)
        
        # Get video directory path from the dataset's own videos folder
        if request.dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_name}' not found")

        dataset_path = Path(dataset_paths[request.dataset_name])
        video_dir = dataset_path / "videos"

        if not video_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video directory not found for dataset '{request.dataset_name}'"
            )
        
        # Generate video descriptions using the same VLM as ingestion
        print(f"Generating video descriptions for dataset: {request.dataset_name} using {vlm_name}")
        descriptions = await generate_video_descriptions_with_vlm(
            video_dir,
            vlm,
            task_description=request.task_description if request.task_description else None
        )
        
        if not descriptions or all(not d for d in descriptions):
            raise HTTPException(
                status_code=400,
                detail="No video descriptions were generated. Please ensure videos exist in the dataset."
            )
        
        print(f"Generated {len(descriptions)} video descriptions")
        
        # Determine axes - use provided axes or default based on environment
        axes = request.axes if request.axes else ["Objects", "Lighting", "Color/Material"]
        
        # Generate prompt variations
        print(f"Generating prompt variations with axes: {axes}")
        variations = generate_prompt_variations(
            prompts=descriptions,
            axes=axes,
            task_description=request.task_description if request.task_description else None
        )
        
        print(f"Generated {len(variations)} prompt variations")
        
        # Create directory path for saving prompt variations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        prompt_folder_name = f"{timestamp}_{unique_id}"
        prompts_base_path = f"prompts/{request.dataset_name}/{prompt_folder_name}"
        
        # Use Supabase Storage if available, otherwise fall back to local filesystem
        use_supabase = supabase is not None
        prompts_dir = None
        
        if not use_supabase:
            # Fallback to local filesystem
            if os.getenv("VERCEL") == "1" or os.getenv("LAMBDA_TASK_ROOT"):
                prompts_dir = Path("/tmp") / "prompts" / request.dataset_name / prompt_folder_name
            else:
                prompts_dir = project_root / "data" / "prompts" / request.dataset_name / prompt_folder_name
            prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of video files in sorted order (same order as descriptions were generated)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
        video_files = sorted([
            f for f in video_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in video_extensions
        ])
        
        # Save each variation to a .txt file and build metadata
        saved_files = []
        metadata = []
        for idx, (axis_changed, variation_text) in enumerate(variations):
            filename = f"variation_{idx+1:04d}.txt"
            
            if use_supabase:
                # Upload to Supabase Storage
                supabase_file_path = f"{prompts_base_path}/{filename}"
                upload_file_to_supabase(supabase_file_path, variation_text.encode('utf-8'), bucket="datasets")
                saved_files.append(supabase_file_path)
                txt_file_path = supabase_file_path
                print(f"Uploaded variation {idx+1} to Supabase: {supabase_file_path}")
            else:
                # Save to local filesystem
                file_path = prompts_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(variation_text)
                saved_files.append(str(file_path.relative_to(project_root)))
                txt_file_path = f"prompts/{request.dataset_name}/{prompt_folder_name}/{filename}"
            
            # Get corresponding video path (same index as description)
            if idx < len(video_files):
                video_path = video_files[idx]
                # Store relative path from dataset root for S3 structure
                # Format: videos/episode_XX.mp4 (relative to dataset directory)
                video_relative_path = video_path.relative_to(dataset_path)
            else:
                # Fallback if mismatch (shouldn't happen, but handle gracefully)
                video_relative_path = video_files[0].relative_to(dataset_path) if video_files else ""
            
            metadata.append({
                "video_path": str(video_relative_path),
                "txt_file_path": txt_file_path,
                "axis": axis_changed
            })
            print(f"Saved variation {idx+1} (video: {video_relative_path}, axis: {axis_changed})")
        
        # Save metadata JSON file
        if use_supabase:
            metadata_file_path = f"{prompts_base_path}/metadata.json"
            metadata_json = json.dumps(metadata, indent=2)
            upload_file_to_supabase(metadata_file_path, metadata_json.encode('utf-8'), bucket="datasets")
            print(f"Uploaded metadata to Supabase: {metadata_file_path}")
        else:
            metadata_file = prompts_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {metadata_file}")
            
            # Upload entire folder to S3 (only if not using Supabase)
            s3_prefix = f"input_data/{request.dataset_name}/prompts/{timestamp}_{unique_id}"
            print(f"Uploading prompt variations folder to S3: {s3_prefix}")
            upload_folder_to_s3(str(prompts_dir), S3_BUCKET, s3_prefix)
        
        return {
            "success": True,
            "dataset_name": request.dataset_name,
            "descriptions_count": len(descriptions),
            "variations_count": len(variations),
            "saved_files": saved_files,
            "s3_prefix": s3_prefix,
            "message": f"Generated {len(variations)} prompt variations and uploaded to S3"
        }
                
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Augmentation error: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Error running augmentation: {str(e)}")

@app.post("/api/optimization/run")
async def run_optimization(request: OptimizationRequest):
    """Run optimization on a dataset."""
    try:
        # Check if dataset exists
        if request.dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_name}' not found")
        
        # Placeholder for optimization logic
        # In a real implementation, this would run optimization algorithms
        # on the dataset and return results
        
        return {
            "success": True,
            "result": f"Optimization completed for dataset '{request.dataset_name}'",
            "dataset_name": request.dataset_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running optimization: {str(e)}")

@app.get("/api/datasets/{dataset_name}/videos/{episode_index}")
async def get_episode_video(dataset_name: str, episode_index: int):
    """Serve episode video file."""
    try:
        if dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        # Serve from the dataset's own videos directory, using sorted video files
        dataset_path = Path(dataset_paths[dataset_name])
        videos_dir = dataset_path / "videos"

        if not videos_dir.exists():
            raise HTTPException(status_code=404, detail=f"Videos directory not found for dataset '{dataset_name}'")

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
        video_files = sorted(
            f for f in videos_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        )

        if episode_index < 0 or episode_index >= len(video_files):
            raise HTTPException(status_code=404, detail=f"Video not found for episode {episode_index}")

        video_path = video_files[episode_index]
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found for episode {episode_index}")
        
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=f"episode_{episode_index}.mp4"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving video: {str(e)}")

@app.post("/api/testing/upload")
async def upload_test_data(request: TestingUploadRequest):
    """Upload test data directory for a dataset."""
    try:
        if request.dataset_name not in dataset_paths:
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_name}' not found")
        
        test_dir = Path(request.test_directory)
        if not test_dir.exists():
            raise HTTPException(status_code=400, detail=f"Test directory does not exist: {request.test_directory}")
        
        if not test_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.test_directory}")
        
        # Placeholder: In a real implementation, you would process/upload the test data
        # For now, just validate and return success
        
        return {
            "success": True,
            "dataset_name": request.dataset_name,
            "test_directory": str(test_dir),
            "message": "Test data directory validated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading test data: {str(e)}")

@app.get("/api/datasets/{dataset_name}/export")
async def export_dataset(dataset_name: str, format: str = "csv"):
    """Export dataset in the specified format."""
    try:
        # Ares database access removed - return empty export
        # In a real implementation, you'd read from dataset files directly
        if format == "csv":
            csv_content = ""  # Empty CSV since no database
            return JSONResponse(
                content={"content": csv_content, "filename": f"{dataset_name}.csv"},
                media_type="application/json"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting dataset: {str(e)}")

# ARES Dashboard API Endpoints (replacing Streamlit functionality)

@app.get("/api/ares/state")
async def get_ares_state():
    """Get ARES state information."""
    try:
        from ares_api import get_state_info
        return get_state_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting state: {str(e)}")

@app.get("/api/ares/filters/metadata")
async def get_filter_metadata():
    """Get metadata for building filter UI (columns, types, ranges, unique values)."""
    try:
        from ares_api import get_dataframe
        from ares.app.data_analysis import infer_visualization_type
        from ares.constants import IGNORE_COLS
        import pandas as pd
        
        df = get_dataframe()
        metadata = []
        
        for col in df.columns:
            if col.lower() in IGNORE_COLS:
                continue
                
            viz_info = infer_visualization_type(col, df)
            if viz_info["viz_type"] is None:
                continue
            
            col_metadata = {
                "column": col,
                "viz_type": viz_info["viz_type"],
                "dtype": str(df[col].dtype),
                "nunique": int(viz_info["nunique"]),
            }
            
            # Add numeric range info
            if viz_info["viz_type"] in ["histogram", "bar"] and pd.api.types.is_numeric_dtype(df[col]):
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                valid_values = numeric_col.dropna()
                if len(valid_values) > 0:
                    min_val = valid_values.min()
                    max_val = valid_values.max()
                    # Handle infinity and NaN values - convert to None for JSON
                    if pd.notna(min_val) and not np.isinf(min_val):
                        col_metadata["min"] = float(min_val)
                    else:
                        col_metadata["min"] = None
                    if pd.notna(max_val) and not np.isinf(max_val):
                        col_metadata["max"] = float(max_val)
                    else:
                        col_metadata["max"] = None
                else:
                    col_metadata["min"] = None
                    col_metadata["max"] = None
                col_metadata["has_nan"] = bool(df[col].isna().any())
            
            # Add categorical options
            if viz_info["viz_type"] == "bar":
                unique_vals = df[col].unique()
                # Limit to 25 options for UI
                if len(unique_vals) <= 25:
                    try:
                        # Try to sort, but handle mixed types
                        sorted_vals = sorted([v for v in unique_vals if pd.notna(v)], key=lambda x: str(x))
                        col_metadata["options"] = [str(v) if pd.notna(v) else "(None)" for v in sorted_vals]
                        # Add None option if it exists
                        if df[col].isna().any():
                            col_metadata["options"].append("(None)")
                    except Exception as e:
                        print(f"[WARNING] Could not sort options for {col}: {e}")
                        col_metadata["options"] = [str(v) if pd.notna(v) else "(None)" for v in unique_vals[:25]]
                else:
                    col_metadata["options"] = None  # Too many options
            
            metadata.append(col_metadata)
        
        print(f"[DEBUG] Returning {len(metadata)} filterable columns")
        return {"columns": metadata}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting filter metadata: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.post("/api/ares/filters/structured")
async def apply_structured_filters(filters: Dict[str, Any] = None):
    """Apply structured data filters."""
    try:
        from ares_api import get_dataframe, apply_structured_filters
        df = get_dataframe()
        filtered_df, active_filters = apply_structured_filters(df, filters or {})
        return {
            "filtered_count": len(filtered_df),
            "total_count": len(df),
            "active_filters": active_filters,
            "data": filtered_df.to_dict('records')[:100],  # Limit to first 100 rows
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error applying filters: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.get("/api/ares/data-sample")
async def get_data_sample(n: int = 5):
    """Get a random sample of the filtered data."""
    try:
        from ares_api import get_dataframe
        df = get_dataframe()
        if len(df) == 0:
            return {"data": []}
        sample = df.sample(min(n, len(df)))
        return {"data": sample.to_dict('records')}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting data sample: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.get("/api/ares/embeddings/{embedding_key}")
async def get_embedding_data(embedding_key: str):
    """Get embedding data for visualization (reduced embeddings, labels, IDs)."""
    try:
        from ares_api import get_dataframe, _global_state
        from ares.databases.embedding_database import META_INDEX_NAMES
        import plotly.graph_objects as go
        
        if embedding_key not in META_INDEX_NAMES:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid embedding key. Must be one of: {META_INDEX_NAMES}"}
            )
        
        reduced_key = f"{embedding_key}_reduced"
        labels_key = f"{embedding_key}_labels"
        ids_key = f"{embedding_key}_ids"
        
        print(f"[DEBUG] Checking for embedding key: {reduced_key}")
        print(f"[DEBUG] Available keys in _global_state: {list(_global_state.keys())[:20]}...")  # Show first 20 keys
        
        # Check if reduced embeddings exist
        if reduced_key not in _global_state:
            available_keys = [k for k in _global_state.keys() if '_reduced' in k]
            all_keys = list(_global_state.keys())
            print(f"[DEBUG] Available _reduced keys: {available_keys}")
            print(f"[DEBUG] All _global_state keys: {all_keys}")
            
            # Try to initialize embeddings if they don't exist
            print(f"[DEBUG] Attempting to ensure embeddings are initialized...")
            try:
                from ares_api import initialize_ares_data
                # Re-initialize to ensure embeddings are loaded
                initialize_ares_data()
                # Check again
                if reduced_key not in _global_state:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "detail": f"Embedding data not found for {embedding_key} after initialization. "
                                     f"Available _reduced keys: {available_keys}. "
                                     f"This may mean embeddings were not generated during data ingestion."
                        }
                    )
            except Exception as init_err:
                print(f"[ERROR] Failed to re-initialize: {init_err}")
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Embedding data not found and re-initialization failed: {str(init_err)}"}
                )
        
        reduced = _global_state[reduced_key]
        labels = _global_state.get(labels_key, None)
        ids = _global_state.get(ids_key, None)
        
        df = get_dataframe()
        
        # Convert numpy arrays to lists for JSON serialization
        # reduced should be 2D array (n_points, 2) for 2D visualization
        if reduced.shape[1] > 2:
            # Use first 2 dimensions
            reduced_2d = reduced[:, :2]
        else:
            reduced_2d = reduced
        
        # Get IDs and raw data
        if ids is not None:
            point_ids = [str(id_val) for id_val in ids]
        elif "id" in df.columns:
            point_ids = df["id"].apply(str).tolist()
        else:
            point_ids = [str(i) for i in range(len(reduced_2d))]
        
        raw_data = df[embedding_key].tolist() if embedding_key in df.columns else [""] * len(reduced_2d)
        
        # Create a simple scatter plot
        fig = go.Figure()
        
        # Add scatter trace
        fig.add_trace(go.Scatter(
            x=reduced_2d[:, 0].tolist(),
            y=reduced_2d[:, 1].tolist(),
            mode='markers',
            marker=dict(
                size=5,
                color='#154e72',
                opacity=0.6,
            ),
            customdata=[[raw_data[i], point_ids[i]] for i in range(len(reduced_2d))],
            hovertemplate='<b>%{customdata[0]}</b><br>ID: %{customdata[1]}<extra></extra>',
            showlegend=False,
        ))
        
        fig.update_layout(
            title='',
            xaxis_title='',
            yaxis_title='',
            plot_bgcolor='transparent',
            paper_bgcolor='transparent',
            font=dict(color='#8a8a8a', size=10),
            height=500,
            margin=dict(l=40, r=40, t=20, b=40),
            xaxis=dict(showgrid=True, gridcolor='#2a2a2a'),
            yaxis=dict(showgrid=True, gridcolor='#2a2a2a'),
        )
        
        # Convert to dict
        fig_dict = fig.to_dict()
        
        return {
            "embedding_key": embedding_key,
            "figure": fig_dict,
            "point_count": len(reduced_2d),
            "point_ids": point_ids,
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting embedding data: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.post("/api/ares/filters/embedding")
async def apply_embedding_filters(selections: Dict[str, Any]):
    """Apply embedding-based filters from user selections."""
    try:
        from ares_api import get_dataframe, apply_structured_filters, get_embedding_filters
        df = get_dataframe()
        
        # First apply structured filters if any
        structured_filtered_df = df
        if selections.get("structured_filters"):
            structured_filtered_df, _ = apply_structured_filters(df, selections["structured_filters"])
        
        # Then apply embedding filters
        embedding_selections = selections.get("embedding_selections", {})
        filtered_df, embedding_figs = get_embedding_filters(df, structured_filtered_df, embedding_selections)
        
        return {
            "filtered_count": len(filtered_df),
            "total_count": len(df),
            "data": filtered_df.to_dict('records')[:100],
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error applying embedding filters: {str(e)}\n\nTraceback:\n{error_trace}"}
        )

@app.get("/api/ares/distributions")
async def get_distributions(
    environment: str | None = None,
    axes: str | None = None,
    dataset_name: str | None = None,
):
    """Get data distribution visualizations.
    
    Args:
        environment: "Indoor" or "Outdoor" mode
        axes: Comma-separated list of axis names (e.g., "Objects,Lighting,Materials")
        dataset_name: Optional dataset name to check ingestion status
    """
    try:
        import traceback
        import json
        from ares_api import get_dataframe, get_data_distributions
        
        # Check if ingestion is in progress for this dataset
        if dataset_name and dataset_name in ingestion_status:
            status = ingestion_status[dataset_name]
            if status == "in_progress":
                print(f"[INFO] Ingestion in progress for {dataset_name}, returning empty visualizations")
                return {"visualizations": [], "ingestion_status": "in_progress"}
            elif status == "failed":
                print(f"[WARNING] Ingestion failed for {dataset_name}, returning empty visualizations")
                return {"visualizations": [], "ingestion_status": "failed"}
        
        df = get_dataframe()
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        
        # Return empty visualizations if database is empty (expected after "New Task")
        if df.empty or len(df) == 0:
            print("[INFO] Database is empty, returning empty visualizations (expected after 'New Task')")
            return {"visualizations": []}
        
        # Parse axes parameter
        # Debug: log what we received from the API request
        print(f"[API] Received axes parameter: {repr(axes)}")
        selected_axes = None
        if axes is not None and axes != "":
            try:
                # Try parsing as JSON array first
                parsed = json.loads(axes)
                # Ensure it's a list (could be empty array [])
                if isinstance(parsed, list):
                    selected_axes = parsed
                else:
                    selected_axes = [parsed] if parsed else []
                print(f"[API] Parsed axes as JSON: {selected_axes}")
            except json.JSONDecodeError as e:
                # Fall back to comma-separated string
                selected_axes = [ax.strip() for ax in axes.split(",") if ax.strip()]
                print(f"[API] Parsed axes as comma-separated (JSON parse failed: {e}): {selected_axes}")
        else:
            # If axes is None or empty string, set to empty list to respect user's explicit deselection
            # The frontend always sends the axes parameter, so None/empty means user unchecked everything
            selected_axes = []
            print(f"[API] Axes parameter is None or empty, setting to empty list (user deselected all)")
        
        # Always call get_data_distributions even if dataframe is empty
        # so it can return empty plots for selected axes
        # use_cache=False to always generate fresh plots from current data
        visualizations = get_data_distributions(
            df,
            environment=environment,
            selected_axes=selected_axes,
            use_cache=False,
        )
        print(f"[DEBUG] Got {len(visualizations)} visualizations")
        return {"visualizations": visualizations}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_detail = f"Error getting distributions: {str(e)}\n\nTraceback:\n{error_trace}"
        print(f"[ERROR] {error_detail}")
        return JSONResponse(
            status_code=500,
            content={"detail": error_detail}
        )

@app.get("/api/ares/success-rate")
async def get_success_rate():
    """Get success rate analytics."""
    try:
        from ares_api import get_dataframe, get_success_rate_analytics
        df = get_dataframe()
        visualizations = get_success_rate_analytics(df)
        return {"visualizations": visualizations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting success rate: {str(e)}")

@app.get("/api/ares/time-series")
async def get_time_series():
    """Get time series analytics."""
    try:
        import traceback
        from ares_api import get_dataframe, get_time_series_analytics
        print("[DEBUG] Getting time series...")
        df = get_dataframe()
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        visualizations = get_time_series_analytics(df)
        print(f"[DEBUG] Got {len(visualizations)} time series visualizations")
        return {"visualizations": visualizations}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_detail = f"Error getting time series: {str(e)}\n\nTraceback:\n{error_trace}"
        print(f"[ERROR] {error_detail}")
        return JSONResponse(
            status_code=500,
            content={"detail": error_detail}
        )

@app.get("/api/ares/videos")
async def get_videos(n: int = 5):
    """Get video grid data."""
    try:
        from ares_api import get_dataframe, get_video_grid_data
        df = get_dataframe()
        videos = get_video_grid_data(df, n_videos=n)
        return {"videos": videos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting videos: {str(e)}")

@app.get("/api/ares/hero/{row_id}")
async def get_hero_display(row_id: str):
    """Get hero display data for a selected row."""
    try:
        from ares_api import get_dataframe, get_hero_display_data
        df = get_dataframe()
        hero_data = get_hero_display_data(row_id, df)
        return hero_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hero display: {str(e)}")

@app.get("/api/ares/robot-array/{row_id}")
async def get_robot_array(row_id: str):
    """Get robot array plots for a selected row."""
    try:
        from ares_api import get_robot_array_plots
        visualizations = get_robot_array_plots(row_id)
        return {"visualizations": visualizations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting robot array: {str(e)}")

@app.post("/api/ares/export")
async def export_ares_data(format: str = "csv", filters: Optional[Dict[str, Any]] = None):
    """Export ARES data."""
    try:
        from ares_api import get_dataframe, apply_structured_filters, export_data
        df = get_dataframe()
        
        # Apply filters if provided
        if filters:
            filtered_df, active_filters = apply_structured_filters(df, filters)
        else:
            filtered_df = df
            active_filters = {}
        
        # Get visualizations (simplified)
        visualizations = []
        
        export_result = export_data(filtered_df, active_filters, visualizations, format)
        return export_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

