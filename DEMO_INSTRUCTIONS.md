# 6d labs Platform Demo Instructions

## Overview

This demo showcases the 6d labs robotics data platform with four key features:
1. **Data Pipeline & Ingestion** - Automated data processing
2. **Feature Engineering** - Extract training-ready features
3. **Quality Monitoring** - Analyze and validate data quality
4. **Dataset Curation** - Intelligent dataset optimization

## Prerequisites

- Python 3.11+
- Virtual environment activated
- All dependencies installed

## Quick Start

### 1. Environment Setup

```bash
cd /Users/mac/demo/ares-platform

# Activate virtual environment
source .venv/bin/activate

# Verify dependencies
pip list | grep streamlit
```

### 2. Configure API Keys (Optional)

For AI-powered features, set your OpenAI API key:

```bash
# Edit .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 3. Prepare Sample Data

#### Option A: Use Existing ARES Data

If you have already run the ingestion pipeline:

```bash
# Check if data exists
ls data/robot_data.db
```

#### Option B: Generate Sample Data

Create a minimal sample dataset for demo purposes:

```bash
# Run this Python script to create sample data
python scripts/create_demo_data.py
```

#### Option C: Download Sample Dataset

Download a small robotics dataset:

```bash
# Example: Download a subset of an OXE dataset
# (Adjust based on available datasets)
python main.py --dataset "cmu_stretch" --max_episodes 50
```

### 4. Launch the Platform

```bash
# Start the Streamlit app
.venv/bin/python -m streamlit run src/ares/app/webapp.py
```

The app will open in your browser at `http://localhost:8501`

## Demo Walkthrough

### Part 1: Data Pipeline & Ingestion (2 minutes)

**What to Show:**
1. **Upload Page**
   - Clean, simple interface with "6d labs" branding
   - File upload capability
   - Dataset naming

2. **Upload Sample Data**
   - Click "Choose files" and select sample robotics data
   - Enter dataset name: "Demo Dataset"
   - Click "Process Data"
   - Show progress indicators

3. **Pipeline Overview**
   - Navigate to "🔄 Data Pipeline" tab
   - Point out key metrics:
     - Total Episodes
     - Success Rate
     - Average Length
     - Number of Datasets
   - Show data distribution charts

**Talking Points:**
- "Our platform automatically ingests and preprocesses robot datasets including telemetry, sensor logs, and demonstrations"
- "The pipeline handles data cleaning, normalization, and structuring"
- "Real-time monitoring of ingestion metrics and data flow"

### Part 2: Feature Engineering (2 minutes)

**What to Show:**
1. Navigate to "⚙️ Feature Engineering" tab
2. Show time series visualizations
3. Explain trajectory embeddings (if available)
4. Point out feature extraction workflows

**Talking Points:**
- "Automated feature extraction generates training-ready inputs from raw robot data"
- "Trajectory embeddings convert robot movements into dense vector representations"
- "Feature engineering workflows extract meaningful patterns from sensor data and actions"
- "These features are optimized for AI model training"

### Part 3: Quality Monitoring (2 minutes)

**What to Show:**
1. Navigate to "🔍 Quality Monitoring" tab
2. Show video grid of rollouts
3. Point out quality metrics:
   - Data Completeness
   - Anomalies Detected
   - Quality Score
4. Explain validation workflows

**Talking Points:**
- "Continuous monitoring of data quality across all datasets"
- "Automated detection of outliers, noise, and potential biases"
- "Quality metrics track completeness and consistency"
- "Validation workflows ensure high-quality training data"

### Part 4: Dataset Curation (3 minutes)

**What to Show:**
1. Navigate to "🤖 Dataset Curation" tab
2. **Set Curation Criteria:**
   - Minimum Success Rate: 0.7
   - Maximum Episodes: 100
   - Diversity Weight: 0.5
   - Select robot types
3. Click "🚀 Optimize Dataset"
4. Show results:
   - Original vs Curated episode count
   - Reduction percentage
   - Average success rate improvement
5. Download curated dataset

**Talking Points:**
- "Automated dataset curation optimizes training data selection"
- "Balances performance metrics with diversity requirements"
- "Ensures balanced representation across different scenarios"
- "Customizable criteria based on specific training objectives"
- "Export optimized datasets for immediate use in model training"

### Part 5: Return to Upload (30 seconds)

**What to Show:**
1. Click "⬅️ Upload New Data" in sidebar
2. Show how easy it is to add more datasets
3. Demonstrate the complete workflow loop

**Talking Points:**
- "Seamless workflow for continuous data ingestion"
- "Easy to add new datasets as they become available"
- "All features work together in an integrated platform"

## Demo Tips

### Before the Demo

1. **Test the full flow** with sample data
2. **Clear browser cache** for clean start
3. **Prepare talking points** for each feature
4. **Have backup data** ready in case of issues
5. **Close unnecessary browser tabs** for performance

### During the Demo

1. **Start with upload page** to show clean UX
2. **Use smooth transitions** between tabs
3. **Highlight key metrics** in each section
4. **Explain the "why"** behind each feature
5. **Show real results** from the curation process

### Common Issues

**Issue: No data appears after upload**
- Solution: Check that `data/robot_data.db` exists
- Restart the app if needed

**Issue: Slow performance**
- Solution: Use smaller sample dataset (< 100 episodes)
- Close other applications

**Issue: Missing visualizations**
- Solution: Ensure embeddings were generated during ingestion
- Check that all dependencies are installed

## Customization for Your Demo

### Replace Placeholder Function

The dataset curation uses a placeholder function. To integrate your actual optimization algorithm:

1. Open `src/ares/app/dataset_curation.py`
2. Find the `optimize_dataset_selection()` function
3. Replace the placeholder logic with your algorithm:

```python
def optimize_dataset_selection(
    df: pd.DataFrame,
    criteria: dict[str, t.Any]
) -> pd.DataFrame:
    # YOUR OPTIMIZATION LOGIC HERE
    # Example: Active learning, diversity sampling, etc.
    
    curated_df = your_optimization_algorithm(df, criteria)
    return curated_df
```

### Adjust Branding

To customize branding:
- Update title in `src/ares/app/webapp.py` (line 35)
- Modify colors in `.streamlit/config.toml`
- Add company logo to upload page

## Recording the Demo

### Recommended Setup

1. **Screen Resolution**: 1920x1080 or 1280x720
2. **Browser**: Chrome (for best performance)
3. **Recording Software**: OBS Studio, Loom, or QuickTime
4. **Audio**: Use good microphone for narration

### Recording Checklist

- [ ] Close unnecessary applications
- [ ] Clear browser history/cache
- [ ] Test full workflow once
- [ ] Prepare script/talking points
- [ ] Check audio levels
- [ ] Start recording
- [ ] Follow demo walkthrough above
- [ ] End with clear call-to-action

## Support

For issues or questions:
- Check `walkthrough.md` for implementation details
- Review `implementation_plan.md` for architecture
- Examine code comments in modified files

## Next Steps

After the demo:
1. Integrate actual optimization algorithm in `dataset_curation.py`
2. Connect to real data pipelines
3. Add custom quality metrics
4. Enhance visualizations based on feedback
5. Deploy to production environment

---

**Demo Duration**: ~10 minutes total
**Recommended Audience**: Technical stakeholders, potential clients, internal teams
**Key Message**: Comprehensive robotics data platform for AI training optimization
