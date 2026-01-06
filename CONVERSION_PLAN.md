# Streamlit to TSX Conversion Plan

## Overview
This document outlines the conversion of `/Users/mac/demo/ares-platform/src/ares/app` from Streamlit to TSX (React/TypeScript) frontend.

## Architecture

### Backend (FastAPI)
- **`webapp-backend/ares_api.py`**: Core API module wrapping ARES functionality
- **`webapp-backend/main.py`**: FastAPI endpoints exposing ARES features
- Uses global state dictionary instead of Streamlit session state

### Frontend (React/TypeScript)
- **`webapp-frontend/`**: React components replacing Streamlit UI
- Components communicate with backend via REST API

## File Conversion Status

### Core Files to Convert:
1. ✅ `sections.py` → `sections_api.py` (API version created)
2. ⏳ `init_data.py` → Remove Streamlit, use global state
3. ⏳ `filter_helpers.py` → Remove Streamlit UI, return filter data
4. ⏳ `viz_helpers.py` → Remove Streamlit, return visualization data
5. ⏳ `hero_display.py` → Remove Streamlit, return hero data
6. ⏳ `plot_primitives.py` → Keep as-is (no Streamlit dependencies)
7. ⏳ `export_data.py` → Remove Streamlit UI, return export data
8. ⏳ `data_analysis.py` → Keep as-is (no Streamlit dependencies)
9. ⏳ `webapp.py` → Replace with API endpoints
10. ⏳ `upload_page.py` → Replace with React component

## Key Features to Implement

### 1. Data Loading & Initialization
- Backend: `/api/ares/initialize` endpoint
- Frontend: Data loading component

### 2. Structured Data Filters
- Backend: `/api/ares/filters/structured` endpoint
- Frontend: Filter UI components (dropdowns, sliders, checkboxes)

### 3. Embedding Data Filters
- Backend: `/api/ares/filters/embedding` endpoint
- Frontend: Interactive scatter plot with selection

### 4. Data Visualizations
- Distributions: `/api/ares/distributions`
- Success Rate: `/api/ares/success-rate`
- Time Series: `/api/ares/time-series`
- Frontend: Plotly.js components

### 5. Video Grid
- Backend: `/api/ares/videos`
- Frontend: Video grid component

### 6. Hero Display
- Backend: `/api/ares/hero/{row_id}`
- Frontend: Hero display component with similar examples

### 7. Robot Array Plots
- Backend: `/api/ares/robot-array/{row_id}`
- Frontend: Robot array visualization component

### 8. Data Export
- Backend: `/api/ares/export`
- Frontend: Export controls component

## Implementation Strategy

1. Create API wrapper functions that remove Streamlit dependencies
2. Create FastAPI endpoints exposing all functionality
3. Create React/TypeScript components for each feature
4. Maintain backward compatibility where possible

