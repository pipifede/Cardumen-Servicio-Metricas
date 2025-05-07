from pathlib import Path

def save_uploaded_file(uploaded_file, upload_dir: Path) -> Path:
    file_path = upload_dir / uploaded_file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(uploaded_file.file.read())
    return file_path

def get_uploaded_file_path(filename: str, upload_dir: Path) -> Path:
    return upload_dir / filename

def delete_file(file_path: Path) -> None:
    if file_path.exists():
        file_path.unlink()