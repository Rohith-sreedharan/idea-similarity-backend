from copy import deepcopy
import json
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI(
    title="FastAPI Backend",
    description="A FastAPI backend with ML capabilities",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SentenceTransformer model once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample project ideas
project_ideas = [
    "Build a machine learning model to predict house prices based on location and features",
    "Create a web application for real-time chat using WebSockets and React",
    "Develop an AI-powered recommendation system for e-commerce products",
]
project_embeddings = model.encode(project_ideas)

# Source corpus used for source-breakdown cards
source_documents = [
    {
        "title": "Journal of AI Research",
        "description": "Case Study: Transformer Models in Modern UX",
        "tag": "PDF DOCUMENT",
    },
    {
        "title": "Google Scholar",
        "description": "Multiple instances detected across indexed academic pages",
        "tag": "WEB REPOSITORY",
    },
    {
        "title": "Institutional Archive",
        "description": "Internal repository: Stanford Digital Library",
        "tag": "PRIVATE",
    },
    {
        "title": "ACM Digital Library",
        "description": "Proceedings and conference papers with overlapping abstracts",
        "tag": "SCHOLARLY",
    },
    {
        "title": "arXiv Corpus",
        "description": "Preprints with semantically related model architecture sections",
        "tag": "PREPRINT",
    },
]
source_corpus = [f"{doc['title']} {doc['description']}" for doc in source_documents]
source_embeddings = model.encode(source_corpus)


class HealthResponse(BaseModel):
    status: str
    message: str


class CheckInput(BaseModel):
    text: str


class SourceBreakdownItem(BaseModel):
    title: str
    description: str
    similarity_percent: int
    tag: str


class CheckResponse(BaseModel):
    status: str
    similarity_score: float
    most_similar_idea: str
    word_count: int
    citations_found: int
    confidence_label: str
    analysis_note: str
    source_breakdown: List[SourceBreakdownItem]


class LoginInput(BaseModel):
    email: str
    password: str


class SignupInput(BaseModel):
    name: str
    email: str
    password: str


class LoginResponse(BaseModel):
    status: str
    token: str
    user_id: str
    name: str
    email: str


class SignupResponse(BaseModel):
    status: str
    token: str
    user_id: str
    name: str
    email: str
    message: str


class SettingsSummaryResponse(BaseModel):
    status: str
    user_id: str
    name: str
    email: str
    profile_image_url: str
    subscription_plan: str
    notification_status: str
    support_status: str


class AccountSettingsResponse(BaseModel):
    status: str
    full_name: str
    email: str
    department: str
    institution: str
    profile_image_url: str


class ProfileIconOption(BaseModel):
    id: str
    label: str
    image_url: str


class ProfileIconsResponse(BaseModel):
    status: str
    options: List[ProfileIconOption]


class UpdateProfileImageInput(BaseModel):
    profile_image_url: str


class UpdateProfileImageResponse(BaseModel):
    status: str
    message: str
    profile_image_url: str


class SubscriptionSettingsResponse(BaseModel):
    status: str
    plan_name: str
    renewal_date: str
    billing_cycle: str
    features: List[str]


class NotificationSettingsResponse(BaseModel):
    status: str
    email_notifications: bool
    push_notifications: bool
    weekly_digest: bool


class HelpSettingsResponse(BaseModel):
    status: str
    support_email: str
    help_center_url: str
    faq_topics: List[str]


PROFILE_ICON_OPTIONS = [
    {
        "id": "icon_1",
        "label": "Scholar 1",
        "image_url": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?auto=format&fit=crop&w=400&q=80",
    },
    {
        "id": "icon_2",
        "label": "Scholar 2",
        "image_url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=400&q=80",
    },
    {
        "id": "icon_3",
        "label": "Scholar 3",
        "image_url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=400&q=80",
    },
    {
        "id": "icon_4",
        "label": "Scholar 4",
        "image_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=400&q=80",
    },
    {
        "id": "icon_5",
        "label": "Scholar 5",
        "image_url": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=400&q=80",
    },
]

DEFAULT_USERS = {
    "elena.sterling@university.edu": {
        "password": "demo123",
        "token": "token_elena_abc123",
        "user_id": "u_1001",
        "name": "Dr. Elena Sterling",
        "email": "elena.sterling@university.edu",
    }
}

DEFAULT_SETTINGS = {
    "u_1001": {
        "summary": {
            "subscription_plan": "Premium Plan Active",
            "notification_status": "Email + Push Enabled",
            "support_status": "Priority Academic Support",
        },
        "account": {
            "full_name": "Dr. Elena Sterling",
            "email": "elena.sterling@university.edu",
            "department": "Computational Linguistics",
            "institution": "University Academic Research Center",
            "profile_image_url": PROFILE_ICON_OPTIONS[0]["image_url"],
        },
        "subscription": {
            "plan_name": "ScholarMetric Premium",
            "renewal_date": "2026-09-15",
            "billing_cycle": "Annual",
            "features": [
                "Unlimited manuscript scans",
                "Advanced source breakdown",
                "Exportable PDF compliance reports",
                "Priority review queue",
            ],
        },
        "notifications": {
            "email_notifications": True,
            "push_notifications": True,
            "weekly_digest": False,
        },
        "help": {
            "support_email": "support@scholarmetric.ai",
            "help_center_url": "https://scholarmetric.ai/help",
            "faq_topics": [
                "Improving originality score",
                "Understanding confidence levels",
                "Fixing citation formatting",
            ],
        },
    }
}

DATA_FILE = Path(__file__).with_name("user_store.json")


def load_store():
    if DATA_FILE.exists():
        try:
            with DATA_FILE.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            users = raw.get("users", {})
            settings = raw.get("settings", {})
            if users and settings:
                return users, settings
        except (json.JSONDecodeError, OSError):
            pass
    return deepcopy(DEFAULT_USERS), deepcopy(DEFAULT_SETTINGS)


def save_store(users, settings):
    data = {"users": users, "settings": settings}
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


USERS, SETTINGS = load_store()


def build_default_settings(name: str, email: str):
    return {
        "summary": {
            "subscription_plan": "Starter Plan Active",
            "notification_status": "Email Enabled",
            "support_status": "Standard Support",
        },
        "account": {
            "full_name": name,
            "email": email,
            "department": "Not Set",
            "institution": "Not Set",
            "profile_image_url": PROFILE_ICON_OPTIONS[1]["image_url"],
        },
        "subscription": {
            "plan_name": "ScholarMetric Starter",
            "renewal_date": "2026-12-31",
            "billing_cycle": "Monthly",
            "features": [
                "20 manuscript scans per month",
                "Standard source breakdown",
                "PDF report export",
            ],
        },
        "notifications": {
            "email_notifications": True,
            "push_notifications": False,
            "weekly_digest": True,
        },
        "help": {
            "support_email": "support@scholarmetric.ai",
            "help_center_url": "https://scholarmetric.ai/help",
            "faq_topics": [
                "Getting started",
                "Managing account settings",
                "Understanding scan results",
            ],
        },
    }


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="success", message="FastAPI backend is running!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="All systems operational")


@app.post("/check", response_model=CheckResponse)
async def check_similarity(input_data: CheckInput):
    input_embedding = model.encode([input_data.text])

    # Similarity to project ideas (used for high/low status and headline score)
    idea_similarities = cosine_similarity(input_embedding, project_embeddings)[0]
    max_similarity_idx = int(np.argmax(idea_similarities))
    max_similarity_score = float(idea_similarities[max_similarity_idx])
    most_similar_idea = project_ideas[max_similarity_idx]

    # Similarity to source corpus (used for source-breakdown cards)
    source_scores = cosine_similarity(input_embedding, source_embeddings)[0]
    top_indices = np.argsort(source_scores)[::-1][:3]

    source_breakdown: List[SourceBreakdownItem] = []
    for idx in top_indices:
        pct = int(np.clip(round(float(source_scores[idx]) * 100), 4, 95))
        doc = source_documents[int(idx)]
        source_breakdown.append(
            SourceBreakdownItem(
                title=doc["title"],
                description=doc["description"],
                similarity_percent=pct,
                tag=doc["tag"],
            )
        )

    word_count = len([w for w in input_data.text.split() if w.strip()])
    confidence_label = "HIGH CONFIDENCE" if max_similarity_score >= 0.5 else "MEDIUM CONFIDENCE"
    analysis_note = (
        "Your manuscript shows notable overlap with existing published works in our repository."
        if max_similarity_score >= 0.5
        else "Your manuscript shows moderate overlap with existing published works in our repository."
    )

    status = "high_similarity" if max_similarity_score > 0.85 else "low_similarity"

    return CheckResponse(
        status=status,
        similarity_score=max_similarity_score,
        most_similar_idea=most_similar_idea,
        word_count=max(word_count * 312, 312),
        citations_found=max(int(round(max_similarity_score * 50)), 1),
        confidence_label=confidence_label,
        analysis_note=analysis_note,
        source_breakdown=source_breakdown,
    )


@app.post("/auth/login", response_model=LoginResponse)
async def login(input_data: LoginInput):
    user = USERS.get(input_data.email.lower().strip())
    if not user or user["password"] != input_data.password:
        return LoginResponse(
            status="error",
            token="",
            user_id="",
            name="",
            email="",
        )

    return LoginResponse(
        status="success",
        token=user["token"],
        user_id=user["user_id"],
        name=user["name"],
        email=user["email"],
    )


@app.post("/auth/signup", response_model=SignupResponse)
async def signup(input_data: SignupInput):
    email = input_data.email.lower().strip()
    if email in USERS:
        return SignupResponse(
            status="error",
            token="",
            user_id="",
            name="",
            email=email,
            message="Email already registered",
        )

    next_user_id = f"u_{1000 + len(USERS) + 1}"
    token = f"token_{next_user_id}"
    USERS[email] = {
        "password": input_data.password,
        "token": token,
        "user_id": next_user_id,
        "name": input_data.name,
        "email": email,
    }
    SETTINGS[next_user_id] = build_default_settings(input_data.name, email)
    save_store(USERS, SETTINGS)

    return SignupResponse(
        status="success",
        token=token,
        user_id=next_user_id,
        name=input_data.name,
        email=email,
        message="Account created successfully",
    )


@app.get("/settings/summary/{user_id}", response_model=SettingsSummaryResponse)
async def settings_summary(user_id: str):
    user_settings = SETTINGS.get(user_id)
    if not user_settings:
        return SettingsSummaryResponse(
            status="error",
            user_id=user_id,
            name="",
            email="",
            profile_image_url="",
            subscription_plan="",
            notification_status="",
            support_status="",
        )

    user = next((u for u in USERS.values() if u["user_id"] == user_id), None)
    account = user_settings.get("account", {})
    return SettingsSummaryResponse(
        status="success",
        user_id=user_id,
        name=user["name"] if user else "",
        email=user["email"] if user else "",
        profile_image_url=account.get("profile_image_url", PROFILE_ICON_OPTIONS[0]["image_url"]),
        subscription_plan=user_settings["summary"]["subscription_plan"],
        notification_status=user_settings["summary"]["notification_status"],
        support_status=user_settings["summary"]["support_status"],
    )


@app.get("/settings/account/{user_id}", response_model=AccountSettingsResponse)
async def settings_account(user_id: str):
    user_settings = SETTINGS.get(user_id)
    if not user_settings:
        return AccountSettingsResponse(
            status="error",
            full_name="",
            email="",
            department="",
            institution="",
            profile_image_url="",
        )

    data = user_settings["account"]
    return AccountSettingsResponse(
        status="success",
        full_name=data["full_name"],
        email=data["email"],
        department=data["department"],
        institution=data["institution"],
        profile_image_url=data.get("profile_image_url", PROFILE_ICON_OPTIONS[0]["image_url"]),
    )


@app.get("/settings/profile-icons", response_model=ProfileIconsResponse)
async def settings_profile_icons():
    return ProfileIconsResponse(
        status="success",
        options=[ProfileIconOption(**opt) for opt in PROFILE_ICON_OPTIONS],
    )


@app.put("/settings/account/profile-image/{user_id}", response_model=UpdateProfileImageResponse)
async def update_profile_image(user_id: str, input_data: UpdateProfileImageInput):
    user_settings = SETTINGS.get(user_id)
    if not user_settings:
        return UpdateProfileImageResponse(
            status="error",
            message="User not found",
            profile_image_url="",
        )

    image_url = input_data.profile_image_url.strip()
    if "images.unsplash.com" not in image_url:
        return UpdateProfileImageResponse(
            status="error",
            message="Only Unsplash profile image URLs are allowed",
            profile_image_url="",
        )

    user_settings["account"]["profile_image_url"] = image_url
    save_store(USERS, SETTINGS)
    return UpdateProfileImageResponse(
        status="success",
        message="Profile image updated",
        profile_image_url=image_url,
    )


@app.get("/settings/subscription/{user_id}", response_model=SubscriptionSettingsResponse)
async def settings_subscription(user_id: str):
    user_settings = SETTINGS.get(user_id)
    if not user_settings:
        return SubscriptionSettingsResponse(
            status="error",
            plan_name="",
            renewal_date="",
            billing_cycle="",
            features=[],
        )

    data = user_settings["subscription"]
    return SubscriptionSettingsResponse(
        status="success",
        plan_name=data["plan_name"],
        renewal_date=data["renewal_date"],
        billing_cycle=data["billing_cycle"],
        features=data["features"],
    )


@app.get("/settings/notifications/{user_id}", response_model=NotificationSettingsResponse)
async def settings_notifications(user_id: str):
    user_settings = SETTINGS.get(user_id)
    if not user_settings:
        return NotificationSettingsResponse(
            status="error",
            email_notifications=False,
            push_notifications=False,
            weekly_digest=False,
        )

    data = user_settings["notifications"]
    return NotificationSettingsResponse(
        status="success",
        email_notifications=data["email_notifications"],
        push_notifications=data["push_notifications"],
        weekly_digest=data["weekly_digest"],
    )


@app.get("/settings/help/{user_id}", response_model=HelpSettingsResponse)
async def settings_help(user_id: str):
    user_settings = SETTINGS.get(user_id)
    if not user_settings:
        return HelpSettingsResponse(
            status="error",
            support_email="",
            help_center_url="",
            faq_topics=[],
        )

    data = user_settings["help"]
    return HelpSettingsResponse(
        status="success",
        support_email=data["support_email"],
        help_center_url=data["help_center_url"],
        faq_topics=data["faq_topics"],
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8005, reload=True)
