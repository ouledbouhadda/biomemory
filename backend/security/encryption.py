from cryptography.fernet import Fernet
from backend.config.settings import get_settings
settings = get_settings()
class DataEncryption:
    def __init__(self):
        key = settings.ENCRYPTION_KEY.encode()
        try:
            self.cipher = Fernet(key)
        except:
            self.cipher = Fernet(Fernet.generate_key())
    def encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()
    def decrypt(self, encrypted_data: str) -> str:
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    def encrypt_dict(self, data: dict) -> dict:
        import json
        encrypted_dict = {}
        for key, value in data.items():
            if isinstance(value, str):
                encrypted_dict[key] = self.encrypt(value)
            else:
                encrypted_dict[key] = self.encrypt(json.dumps(value))
        return encrypted_dict
    def decrypt_dict(self, encrypted_data: dict) -> dict:
        import json
        decrypted_dict = {}
        for key, value in encrypted_data.items():
            decrypted_value = self.decrypt(value)
            try:
                decrypted_dict[key] = json.loads(decrypted_value)
            except:
                decrypted_dict[key] = decrypted_value
        return decrypted_dict
_encryption_service = None
def get_encryption_service() -> DataEncryption:
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = DataEncryption()
    return _encryption_service