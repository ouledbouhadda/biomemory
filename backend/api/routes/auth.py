from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
from datetime import timedelta, datetime
from backend.models.requests import UserRegisterRequest, MFAVerifyRequest
from backend.models.responses import TokenResponse, UserResponse, MFASetupResponse
from backend.security.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token
)
from backend.security.mfa import MFAManager
from backend.security.rate_limiting import RateLimiter
from backend.security.audit_logger import get_audit_logger
from backend.config.settings import get_settings
import backend.main as main_module
router = APIRouter()
settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")
mfa_manager = MFAManager()
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
audit_logger = get_audit_logger()
def get_user_repo():
    repo = main_module.get_user_repository()
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User repository not initialized"
        )
    return repo
async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
    repo = get_user_repo()
    user = repo.get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: UserRegisterRequest):
    repo = get_user_repo()
    allowed, info = rate_limiter.is_allowed(request.email)
    if not allowed:
        audit_logger.log_event(
            event_type="auth",
            user_id=request.email,
            action="register_rate_limited",
            resource="auth",
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        hashed_password = get_password_hash(request.password)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password error: {str(e)}"
        )
    try:
        user = repo.create_user(
            email=request.email,
            hashed_password=hashed_password,
            full_name=request.full_name
        )
    except ValueError as e:
        audit_logger.log_event(
            event_type="auth",
            user_id=request.email,
            action="register_duplicate",
            resource="auth",
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    audit_logger.log_event(
        event_type="auth",
        user_id=request.email,
        action="register",
        resource="auth",
        success=True
    )
    return UserResponse(
        email=user["email"],
        full_name=user["full_name"],
        is_active=user["is_active"],
        mfa_enabled=user["mfa_enabled"]
    )
@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    repo = get_user_repo()
    allowed, info = rate_limiter.is_allowed(form_data.username)
    if not allowed:
        audit_logger.log_event(
            event_type="auth",
            user_id=form_data.username,
            action="login_rate_limited",
            resource="auth",
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    user = repo.get_user_by_email(form_data.username)
    if not user:
        audit_logger.log_event(
            event_type="auth",
            user_id=form_data.username,
            action="login_failed",
            resource="auth",
            success=False,
            details={"reason": "user_not_found"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not verify_password(form_data.password, user["hashed_password"]):
        audit_logger.log_event(
            event_type="auth",
            user_id=form_data.username,
            action="login_failed",
            resource="auth",
            success=False,
            details={"reason": "invalid_password"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    repo.update_last_login(user["email"])
    if user.get("mfa_enabled"):
        access_token = create_access_token(
            data={"sub": user["email"], "mfa_required": True},
            expires_delta=timedelta(minutes=5)
        )
        audit_logger.log_event(
            event_type="auth",
            user_id=form_data.username,
            action="login_mfa_required",
            resource="auth",
            success=True
        )
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            mfa_required=True
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=access_token_expires
    )
    audit_logger.log_event(
        event_type="auth",
        user_id=form_data.username,
        action="login_success",
        resource="auth",
        success=True
    )
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        mfa_required=False
    )
@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        email=current_user["email"],
        full_name=current_user["full_name"],
        is_active=current_user["is_active"],
        mfa_enabled=current_user["mfa_enabled"]
    )
@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(current_user: dict = Depends(get_current_user)):
    repo = get_user_repo()
    secret = mfa_manager.generate_secret(current_user["email"])
    qr_code_base64 = mfa_manager.generate_qr_code(
        secret=secret,
        email=current_user["email"],
        issuer="BIOMEMORY"
    )
    repo.set_mfa_pending(current_user["email"], secret)
    audit_logger.log_event(
        event_type="auth",
        user_id=current_user["email"],
        action="mfa_setup_initiated",
        resource="auth",
        success=True
    )
    return MFASetupResponse(
        secret=secret,
        qr_code=qr_code_base64,
        instructions="Scan this QR code with your authenticator app (Google Authenticator, Authy, etc.) and verify with a code to complete setup."
    )
@router.post("/mfa/verify")
async def verify_mfa(
    request: MFAVerifyRequest,
    current_user: dict = Depends(get_current_user)
):
    repo = get_user_repo()
    if current_user.get("mfa_secret_pending"):
        secret = current_user["mfa_secret_pending"]
        is_setup = True
    elif current_user.get("mfa_enabled"):
        secret = current_user["mfa_secret"]
        is_setup = False
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is not set up for this account"
        )
    is_valid = mfa_manager.verify_token(secret, request.code)
    if not is_valid:
        audit_logger.log_event(
            event_type="auth",
            user_id=current_user["email"],
            action="mfa_verify_failed",
            resource="auth",
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA code"
        )
    if is_setup:
        repo.enable_mfa(current_user["email"], secret)
        audit_logger.log_event(
            event_type="auth",
            user_id=current_user["email"],
            action="mfa_enabled",
            resource="auth",
            success=True
        )
        return {
            "message": "MFA successfully enabled",
            "mfa_enabled": True
        }
    else:
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": current_user["email"]},
            expires_delta=access_token_expires
        )
        audit_logger.log_event(
            event_type="auth",
            user_id=current_user["email"],
            action="mfa_login_success",
            resource="auth",
            success=True
        )
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            mfa_required=False
        )
@router.post("/mfa/disable")
async def disable_mfa(current_user: dict = Depends(get_current_user)):
    repo = get_user_repo()
    if not current_user.get("mfa_enabled"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is not enabled"
        )
    repo.disable_mfa(current_user["email"])
    audit_logger.log_event(
        event_type="auth",
        user_id=current_user["email"],
        action="mfa_disabled",
        resource="auth",
        success=True
    )
    return {
        "message": "MFA successfully disabled",
        "mfa_enabled": False
    }
@router.get("/debug/users")
async def debug_list_users():
    repo = get_user_repo()
    users = repo.list_all_users(limit=100)
    users_info = []
    for user in users:
        users_info.append({
            "email": user.get("email"),
            "full_name": user.get("full_name"),
            "is_active": user.get("is_active"),
            "mfa_enabled": user.get("mfa_enabled"),
            "has_password": bool(user.get("hashed_password")),
            "created_at": user.get("created_at"),
            "last_login": user.get("last_login")
        })
    return {
        "total_users": repo.count_users(),
        "users": users_info
    }