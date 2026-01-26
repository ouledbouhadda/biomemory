import pyotp
import qrcode
from io import BytesIO
import base64
class MFAManager:
    def __init__(self, issuer_name: str = "BioMemory"):
        self.issuer_name = issuer_name
    def generate_secret(self, user_email: str) -> str:
        return pyotp.random_base32()
    def get_totp_uri(self, user_email: str, secret: str) -> str:
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=user_email,
            issuer_name=self.issuer_name
        )
    def generate_qr_code(self, totp_uri: str) -> str:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
    def verify_totp(self, secret: str, token: str) -> bool:
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    def get_current_token(self, secret: str) -> str:
        totp = pyotp.TOTP(secret)
        return totp.now()