"""Pure auth functions -- no FastAPI imports here.

This is the live-session extension surface: swapping refresh tokens, OAuth,
or a different hashing scheme touches only this file.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from pydantic import BaseModel

MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # e.g. malformed hash -- never let a verification error look like a pass
        return False


class TokenPayload(BaseModel):
    sub: str
    role: str
    exp: int


def create_access_token(
    *, sub: str, role: str, secret: str, expiry_min: int, now: datetime | None = None
) -> str:
    now = now or datetime.now(timezone.utc)
    exp = now + timedelta(minutes=expiry_min)
    payload = {
        "sub": sub,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> TokenPayload:
    """Raises jwt.PyJWTError subclasses (ExpiredSignatureError,
    InvalidTokenError, ...) on any problem -- callers map these to 401."""
    payload = jwt.decode(token, secret, algorithms=["HS256"])
    return TokenPayload(sub=payload["sub"], role=payload["role"], exp=payload["exp"])
