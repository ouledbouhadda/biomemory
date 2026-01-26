from fastapi import UploadFile, HTTPException
import os
import uuid
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
from typing import Optional
import PyPDF2
from backend.config.settings import get_settings
settings = get_settings()
class FileService:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = settings.MAX_UPLOAD_SIZE
        self.allowed_extensions = {
            'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
            'document': ['.pdf', '.txt', '.md'],
            'data': ['.csv', '.json', '.fasta', '.fastq']
        }
    async def save_upload(
        self,
        file: UploadFile,
        user_id: Optional[str] = None
    ) -> Path:
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        if file_size > self.max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {self.max_size / (1024*1024):.1f}MB"
            )
        file_ext = Path(file.filename).suffix.lower()
        if not self._is_allowed_extension(file_ext):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed: {file_ext}"
            )
        unique_name = f"{uuid.uuid4()}{file_ext}"
        if user_id:
            user_dir = self.upload_dir / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            file_path = user_dir / unique_name
        else:
            file_path = self.upload_dir / unique_name
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return file_path
    async def image_to_base64(
        self,
        file_path: Path,
        max_size: tuple = (800, 800)
    ) -> str:
        try:
            image = Image.open(file_path)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image_format = image.format or 'PNG'
            image.save(buffer, format=image_format)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/{image_format.lower()};base64,{img_base64}"
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing image: {str(e)}"
            )
    async def extract_pdf_text(self, file_path: Path) -> str:
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text_parts = []
                for page in pdf_reader.pages:
                    text_parts.append(page.extract_text())
                return "\n".join(text_parts)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting PDF text: {str(e)}"
            )
    async def read_fasta(self, file_path: Path) -> list:
        sequences = []
        current_id = None
        current_seq = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_id:
                            sequences.append({
                                'id': current_id,
                                'sequence': ''.join(current_seq)
                            })
                        current_id = line[1:]
                        current_seq = []
                    else:
                        current_seq.append(line)
                if current_id:
                    sequences.append({
                        'id': current_id,
                        'sequence': ''.join(current_seq)
                    })
            return sequences
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error parsing FASTA: {str(e)}"
            )
    def _is_allowed_extension(self, extension: str) -> bool:
        all_allowed = []
        for exts in self.allowed_extensions.values():
            all_allowed.extend(exts)
        return extension in all_allowed
    async def delete_file(self, file_path: Path):
        if file_path.exists():
            file_path.unlink()
_file_service = None
def get_file_service() -> FileService:
    global _file_service
    if _file_service is None:
        _file_service = FileService()
    return _file_service