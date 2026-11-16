"""Local crop-label editor. Originals are never deleted by the UI."""
from pathlib import Path
import os
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", Path(__file__).parent/"static/images")).resolve()
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
app = FastAPI(title="WATERec crop annotator")


def image_path(name):
    path = IMAGE_DIR/name
    if Path(name).name != name or path.suffix.lower() not in EXTENSIONS or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if path.resolve().parent != IMAGE_DIR:
        raise HTTPException(status_code=404, detail="Image outside image directory")
    return path


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent/"templates/index.html").read_text(encoding="utf-8")


@app.get("/api/images")
def list_images():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for path in sorted(IMAGE_DIR.iterdir()):
        if path.suffix.lower() in EXTENSIONS and path.is_file() and path.resolve().parent == IMAGE_DIR:
            label = path.with_suffix(".txt")
            result.append({"name": path.name, "label": label.read_text(encoding="utf-8") if label.is_file() else ""})
    return result


@app.get("/images/{name}")
def show_image(name: str):
    return FileResponse(image_path(name))


class Label(BaseModel):
    text: str = Field(min_length=1, max_length=256)


@app.put("/api/labels/{name}")
def save_label(name: str, label: Label):
    target = image_path(name).with_suffix(".txt")
    text = label.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="A label cannot be empty")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=IMAGE_DIR,
                                     prefix=".label-", delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(target)
    return {"name": name, "label": text}
