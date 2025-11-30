"""
Test script for Cosmos augmentation pipeline.

This script tests the CosmosAugmentor class with a sample video.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ares.augmentation.cosmos2 import CosmosAugmentor


def test_cosmos_augmentation():
    """
    Test the Cosmos augmentation pipeline with a sample video.
    """
    print("=" * 80)
    print("Testing Cosmos Augmentation Pipeline")
    print("=" * 80)
    
    # Configuration
    # Update these paths to point to your actual video file
    video_path = Path("/Users/mac/Downloads/open_droids_data/stack_the_cups_lerobot_v3/videos/episode_videos/episode_01.mp4")
    output_dir = Path("/Users/mac/Downloads/open_droids_data/stack_the_cups_lerobot_v3/videos/augmented_videos")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if video exists
    if not video_path.exists():
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        print("\nPlease update the video_path in this script to point to a valid video file.")
        print("Example: /Users/mac/demo/ares-platform/data/videos/stack_cups/episode_0.mp4")
        return
    
    print(f"\n✓ Video file found: {video_path}")
    print(f"✓ Output directory: {output_dir}")
    
    # Initialize augmentor
    print("\n" + "-" * 80)
    print("Step 1: Initializing CosmosAugmentor...")
    print("-" * 80)
    
    try:
        augmentor = CosmosAugmentor(use_gpu=False)  # Set to True if you have GPU
        print("✓ CosmosAugmentor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize CosmosAugmentor: {e}")
        return
    
    # Set initial caption (user input)
    print("\n" + "-" * 80)
    print("Step 2: Setting initial caption...")
    print("-" * 80)
    
    augmentor.initial_caption = (
        "The scene depicts a bright, modern kitchen with plenty of ambient light. "
        "From a first-person perspective, a [robot type] faces a table. "
        "On the table rest a [object 1 color] [object 1] and a [object 2 color] [object 2]."
    )
    print(f"✓ Initial caption set:\n  {augmentor.initial_caption}")
    
    # Set OpenAI API key (required for prompt variations)
    print("\n" + "-" * 80)
    print("Step 3: Setting OpenAI API key...")
    print("-" * 80)
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")
        print("   Prompt variation will fail. Set it with:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
    else:
        print("✓ OpenAI API key found")
    
    augmentor.openai_api_key = openai_api_key
    
    # Run augmentation
    print("\n" + "-" * 80)
    print("Step 4: Running augmentation pipeline...")
    print("-" * 80)
    print("\nThis will:")
    print("  1. Generate prompt variations using OpenAI API")
    print("  2. Segment robot pixels using Grounding DINO + SAMv2")
    print("  3. Generate binary mask video (white=robot, black=background)")
    print("  4. Generate augmented videos using Cosmos-Transfer2.5")
    print("\nThis may take several minutes...\n")
    
    try:
        result = augmentor.augment_episode(
            original_video_path=video_path,
            output_dir=output_dir
        )
        
        print("\n" + "=" * 80)
        print("✓ AUGMENTATION COMPLETE!")
        print("=" * 80)
        print(f"\nGenerated video path: {result}")
        
        # List all generated files
        print("\nGenerated files:")
        for file in sorted(output_dir.glob("*")):
            file_size = file.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {file.name} ({file_size:.2f} MB)")
        
    except Exception as e:
        print(f"\n❌ Augmentation failed: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


def test_robot_segmentation_only():
    """
    Test only the robot segmentation step to verify mask generation.
    """
    print("=" * 80)
    print("Testing Robot Segmentation (Mask Generation)")
    print("=" * 80)
    
    # Configuration
    video_path = Path("/Users/mac/demo/ares-platform/data/videos/stack_cups/episode_0.mp4")
    output_dir = Path("/Users/mac/demo/ares-platform/data/test_masks")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if video exists
    if not video_path.exists():
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        return
    
    print(f"\n✓ Video file found: {video_path}")
    print(f"✓ Output directory: {output_dir}")
    
    # Initialize augmentor
    print("\nInitializing CosmosAugmentor...")
    try:
        augmentor = CosmosAugmentor(use_gpu=False)
        print("✓ CosmosAugmentor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Run segmentation
    print("\nRunning robot segmentation...")
    print("This will generate a binary mask video where:")
    print("  - White pixels (255) = robot regions")
    print("  - Black pixels (0) = background")
    print()
    
    try:
        mask_video_path = augmentor._segment_robot(str(video_path), output_dir)
        
        if mask_video_path:
            print(f"\n✓ Mask video generated: {mask_video_path}")
            
            # Get file size
            mask_size = Path(mask_video_path).stat().st_size / (1024 * 1024)
            print(f"  File size: {mask_size:.2f} MB")
            
            print("\nYou can now view the mask video to verify robot segmentation.")
            print(f"  open {mask_video_path}")
        else:
            print("\n⚠️  No mask generated (robot not detected or segmentation failed)")
            
    except Exception as e:
        print(f"\n❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Cosmos augmentation pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "segmentation"],
        default="full",
        help="Test mode: 'full' for complete pipeline, 'segmentation' for mask generation only"
    )
    
    args = parser.parse_args()
    
    if args.mode == "full":
        test_cosmos_augmentation()
    elif args.mode == "segmentation":
        test_robot_segmentation_only()
