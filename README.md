# Idea Similarity Backend (FastAPI)

FastAPI backend powering authentication, idea similarity scanning, settings, and persistent profile icon updates.

## Features

- Semantic idea similarity using `sentence-transformers`
- Source breakdown generation for analysis cards
- Auth APIs: login and signup
- Settings APIs for summary, account, subscription, notifications, help
- Unsplash profile icon catalog and profile image update API
- File-based persistence (`user_store.json`) for permanent user/settings data

## Tech Stack

- Python
- FastAPI
- Uvicorn
- sentence-transformers
- scikit-learn
- NumPy

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Development:

```bash
python app.py
```

Or:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8005
```

The current backend runs on port `8005`.

## API Docs

- Swagger: http://localhost:8005/docs
- ReDoc: http://localhost:8005/redoc

## API Endpoints

### Health

- `GET /`
- `GET /health`

### Similarity

- `POST /check`

Request:

```json
{
  "text": "Your manuscript title or content"
}
```

### Auth

- `POST /auth/login`
- `POST /auth/signup`

### Settings

- `GET /settings/summary/{user_id}`
- `GET /settings/account/{user_id}`
- `GET /settings/subscription/{user_id}`
- `GET /settings/notifications/{user_id}`
- `GET /settings/help/{user_id}`
- `GET /settings/profile-icons`
- `PUT /settings/account/profile-image/{user_id}`

## Persistence

User and settings data are persisted in:

- `user_store.json`

Updates that are persisted permanently:

- Signup-created users
- Profile icon updates

## Project Structure

```
idea-similarity-backend/
├── app.py
├── requirements.txt
├── user_store.json
└── README.md
```
