from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from backend.config.settings import get_settings
settings = get_settings()
class UserRepository:
    COLLECTION_NAME = "biomemory_users"
    def __init__(self, client: QdrantClient):
        self.client = client
        self._ensure_collection_exists()
    def _ensure_collection_exists(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE)
                )
                print(f"✅ Collection '{self.COLLECTION_NAME}' créée dans Qdrant")
        except Exception as e:
            print(f"⚠️ Erreur lors de la création de la collection users: {e}")
    def create_user(self, email: str, hashed_password: str, full_name: Optional[str] = None) -> Dict[str, Any]:
        existing_user = self.get_user_by_email(email)
        if existing_user:
            raise ValueError(f"Email '{email}' already registered")
        user_id = str(uuid.uuid4())
        now = datetime.utcnow()
        user_data = {
            "id": user_id,
            "email": email,
            "full_name": full_name,
            "hashed_password": hashed_password,
            "is_active": True,
            "mfa_enabled": False,
            "mfa_secret": None,
            "mfa_secret_pending": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_login": None
        }
        point = PointStruct(
            id=user_id,
            vector=[0.0],
            payload=user_data
        )
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point]
        )
        return user_data
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        try:
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="email",
                            match=MatchValue(value=email)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            if results and results[0]:
                points = results[0]
                if points:
                    return points[0].payload
            return None
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération de l'utilisateur: {e}")
            return None
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.client.retrieve(
                collection_name=self.COLLECTION_NAME,
                ids=[user_id],
                with_payload=True,
                with_vectors=False
            )
            if result:
                return result[0].payload
            return None
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération de l'utilisateur: {e}")
            return None
    def update_user(self, email: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_email(email)
        if not user:
            return None
        user.update(updates)
        user["updated_at"] = datetime.utcnow().isoformat()
        point = PointStruct(
            id=user["id"],
            vector=[0.0],
            payload=user
        )
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point]
        )
        return user
    def delete_user(self, email: str) -> bool:
        user = self.get_user_by_email(email)
        if not user:
            return False
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=[user["id"]]
        )
        return True
    def list_all_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        try:
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            if results and results[0]:
                points = results[0]
                return [point.payload for point in points]
            return []
        except Exception as e:
            print(f"⚠️ Erreur lors de la liste des utilisateurs: {e}")
            return []
    def count_users(self) -> int:
        try:
            collection_info = self.client.get_collection(self.COLLECTION_NAME)
            return collection_info.points_count
        except Exception as e:
            print(f"⚠️ Erreur lors du comptage: {e}")
            return 0
    def update_last_login(self, email: str):
        self.update_user(email, {
            "last_login": datetime.utcnow().isoformat()
        })
    def enable_mfa(self, email: str, mfa_secret: str) -> bool:
        result = self.update_user(email, {
            "mfa_enabled": True,
            "mfa_secret": mfa_secret,
            "mfa_secret_pending": None
        })
        return result is not None
    def disable_mfa(self, email: str) -> bool:
        result = self.update_user(email, {
            "mfa_enabled": False,
            "mfa_secret": None,
            "mfa_secret_pending": None
        })
        return result is not None
    def set_mfa_pending(self, email: str, mfa_secret: str) -> bool:
        result = self.update_user(email, {
            "mfa_secret_pending": mfa_secret
        })
        return result is not None