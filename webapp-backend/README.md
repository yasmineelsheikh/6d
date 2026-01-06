# Ares Platform Backend API

FastAPI backend for the Ares platform web application.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create a `.env` file):
```
POD_API_URL=https://your-pod-url/run-json
S3_BUCKET=your-bucket-name
S3_REGION=us-west-2
```

3. Run the server:
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /api/datasets/load` - Load a dataset
- `GET /api/datasets/{dataset_name}/info` - Get dataset information
- `GET /api/datasets/{dataset_name}/data` - Get dataset data
- `GET /api/datasets/{dataset_name}/distributions` - Get distribution visualizations
- `POST /api/augmentation/run` - Run Cosmos augmentation
- `GET /api/datasets/{dataset_name}/export` - Export dataset

