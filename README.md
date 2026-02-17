# FastAPI Backend Project

A FastAPI backend application with machine learning capabilities using sentence-transformers and scikit-learn.

## Setup

### 1. Activate Virtual Environment

```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

Dependencies are already installed, but if you need to reinstall:

```bash
pip install -r requirements.txt
```

## Running the Application

### Development Mode (with auto-reload)

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
- **GET** `/` - Root endpoint
- **GET** `/health` - Health check endpoint

### Processing
- **POST** `/api/process` - Process text input
  ```json
  {
    "text": "Your text here"
  }
  ```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
Pdd/
├── venv/                  # Virtual environment
├── app.py                 # Main FastAPI application
├── requirements.txt       # Project dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Installed Packages

- **fastapi** - Modern web framework for building APIs
- **uvicorn** - ASGI server for running FastAPI
- **sentence-transformers** - State-of-the-art sentence embeddings
- **scikit-learn** - Machine learning library
- **pydantic** - Data validation using Python type annotations

## Next Steps

1. Implement your ML logic in the `/api/process` endpoint
2. Add additional endpoints as needed
3. Configure CORS settings for production
4. Add authentication if required
5. Set up database connections if needed
