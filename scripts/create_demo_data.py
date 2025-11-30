"""
Create sample robotics data for demo purposes.

This script generates a minimal synthetic dataset that can be used
to demonstrate the 6d labs platform features.
"""

import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Add parent directory to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ares.constants import ARES_DATA_DIR
from src.ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)


def generate_sample_data(n_episodes=50):
    """Generate sample robotics data for demo."""
    
    print(f"Generating {n_episodes} sample episodes...")
    
    # Sample robot types and tasks
    robot_types = ['franka', 'ur5', 'kinova', 'fetch']
    task_types = ['pick_and_place', 'push', 'reach', 'grasp']
    datasets = ['demo_dataset_1', 'demo_dataset_2']
    
    data = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(n_episodes):
        # Generate realistic-looking data
        robot = np.random.choice(robot_types)
        task = np.random.choice(task_types)
        dataset = np.random.choice(datasets)
        
        # Success rate varies by task difficulty
        success_prob = 0.7 if task in ['pick_and_place', 'grasp'] else 0.85
        success = np.random.random() < success_prob
        
        # Episode length varies
        length = int(np.random.normal(100, 30))
        length = max(20, min(200, length))
        
        episode = {
            'id': str(uuid.uuid4()),
            'dataset_name': dataset,
            'dataset_formalname': dataset.replace('_', ' ').title(),
            'dataset_filename': dataset,
            'robot_embodiment': robot,
            'task_language_instruction': f"{task.replace('_', ' ')} task with {robot} robot",
            'task_success': 1.0 if success else 0.0,
            'task_success_estimate': success_prob + np.random.normal(0, 0.1),
            'length': length,
            'path': f"episode_{i:04d}",
            'filename': f"episode_{i:04d}.hdf5",
            'ingestion_time': base_time + timedelta(hours=i),
            'description_estimate': f"Robot {robot} performing {task} task",
        }
        
        data.append(episode)
    
    df = pd.DataFrame(data)
    
    # Ensure success estimate is in [0, 1]
    df['task_success_estimate'] = df['task_success_estimate'].clip(0, 1)
    
    return df


def save_to_database(df):
    """Save sample data to the ARES database."""
    
    print("Setting up database...")
    
    # Ensure data directory exists
    os.makedirs(ARES_DATA_DIR, exist_ok=True)
    
    # Setup database
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    
    print(f"Saving {len(df)} episodes to database...")
    
    # Save to database
    df.to_sql('rollout', engine, if_exists='replace', index=False)
    
    print(f"✅ Sample data saved to {ROBOT_DB_PATH}")
    print(f"   Total episodes: {len(df)}")
    print(f"   Datasets: {df['dataset_name'].nunique()}")
    print(f"   Robot types: {df['robot_embodiment'].nunique()}")
    print(f"   Average success rate: {df['task_success'].mean():.1%}")


def main():
    """Main function to generate and save sample data."""
    
    print("=" * 60)
    print("6d labs Platform - Sample Data Generator")
    print("=" * 60)
    print()
    
    # Generate sample data
    df = generate_sample_data(n_episodes=50)
    
    # Save to database
    save_to_database(df)
    
    print()
    print("=" * 60)
    print("Sample data generation complete!")
    print("You can now run the platform:")
    print("  .venv/bin/python -m streamlit run src/ares/app/webapp.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
