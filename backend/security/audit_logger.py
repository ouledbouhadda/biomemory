from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
from typing import Dict, Any
import json
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audit.log'),
        logging.StreamHandler()
    ]
)
audit_logger = logging.getLogger('biomemory.audit')
class AuditLogger:
    def log_event(
        self,
        event_type: str,
        user_id: str,
        action: str,
        resource: str,
        details: Dict[str, Any] = None,
        success: bool = True
    ):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
            "details": details or {}
        }
        if success:
            audit_logger.info(json.dumps(log_entry))
        else:
            audit_logger.warning(json.dumps(log_entry))
    def log_authentication(
        self,
        user_id: str,
        success: bool,
        method: str = "password",
        ip_address: str = None
    ):
        self.log_event(
            event_type="authentication",
            user_id=user_id,
            action="login",
            resource="auth",
            details={
                "method": method,
                "ip_address": ip_address
            },
            success=success
        )
    def log_search(
        self,
        user_id: str,
        query: Dict[str, Any],
        results_count: int
    ):
        self.log_event(
            event_type="search",
            user_id=user_id,
            action="search",
            resource="experiments",
            details={
                "query_type": "multimodal",
                "results_count": results_count,
                "has_sequence": bool(query.get("sequence")),
                "has_image": bool(query.get("image_base64"))
            },
            success=True
        )
    def log_experiment_upload(
        self,
        user_id: str,
        experiment_id: str,
        success: bool
    ):
        self.log_event(
            event_type="data_upload",
            user_id=user_id,
            action="create",
            resource=f"experiment/{experiment_id}",
            success=success
        )
    def log_data_access(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str
    ):
        self.log_event(
            event_type="data_access",
            user_id=user_id,
            action="read",
            resource=f"{resource_type}/{resource_id}",
            success=True
        )
class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, audit_logger: AuditLogger):
        super().__init__(app)
        self.audit_logger = audit_logger
    async def dispatch(self, request: Request, call_next):
        start_time = datetime.utcnow()
        user_id = "anonymous"
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from jose import jwt
                from backend.config.settings import get_settings
                settings = get_settings()
                token = auth_header.split(" ")[1]
                payload = jwt.decode(
                    token,
                    settings.SECRET_KEY,
                    algorithms=[settings.ALGORITHM]
                )
                user_id = payload.get("sub", "anonymous")
            except:
                pass
        response = await call_next(request)
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if response.status_code >= 400 or self._is_sensitive_endpoint(request.url.path):
            log_entry = {
                "timestamp": start_time.isoformat(),
                "user_id": user_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "ip_address": request.client.host if request.client else "unknown"
            }
            audit_logger.info(json.dumps(log_entry))
        return response
    def _is_sensitive_endpoint(self, path: str) -> bool:
        sensitive_patterns = [
            "/auth/",
            "/experiments/",
            "/design/"
        ]
        return any(pattern in path for pattern in sensitive_patterns)
_audit_logger_instance = None
def get_audit_logger() -> AuditLogger:
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger()
    return _audit_logger_instance