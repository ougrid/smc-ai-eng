"""FastAPI-facing auth dependencies. All the FastAPI-specific glue lives
here; app/auth/service.py underneath stays framework-free."""

import uuid

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.models import User
from app.auth.service import decode_token
from app.config import Settings, get_settings
from app.db import get_session

_bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> User:
    if credentials is None:
        raise _UNAUTHORIZED
    try:
        payload = decode_token(credentials.credentials, settings.jwt_secret)
        user_id = uuid.UUID(payload.sub)
    except (jwt.PyJWTError, ValueError):
        raise _UNAUTHORIZED
    user = session.get(User, user_id)
    if user is None:
        raise _UNAUTHORIZED
    return user


def require_role(role: str):
    """Live-session extension stub -- no callers yet in this build."""

    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role != role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user

    return dependency
