"""Local manual bounding-box annotator for the selected Aquarium images."""

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote
import json

ROOT = Path(__file__).resolve().parent
IMAGES = sorted(p.name for p in ROOT.glob("*.jpg"))
CLASSES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
LABEL_DIR = ROOT / "manual_annotations"
LABEL_DIR.mkdir(exist_ok=True)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            return self.reply(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/images":
            return self.reply(200, {"images": IMAGES, "classes": CLASSES, "done": [p.stem for p in LABEL_DIR.glob("*.json") if json.loads(p.read_text()).get("reviewed")]})
        if path.startswith("/images/"):
            name = path.removeprefix("/images/")
            if name not in IMAGES:
                return self.reply(404, {"error": "Unknown image"})
            return self.reply(200, (ROOT / name).read_bytes(), "image/jpeg")
        if path.startswith("/api/annotations/"):
            name = path.removeprefix("/api/annotations/")
            if name not in IMAGES:
                return self.reply(404, {"error": "Unknown image"})
            file = LABEL_DIR / (Path(name).stem + ".json")
            return self.reply(200, json.loads(file.read_text()) if file.exists() else {"image": name, "boxes": [], "reviewed": False})
        if path == "/api/export":
            records = [json.loads(p.read_text()) for p in sorted(LABEL_DIR.glob("*.json"))]
            return self.reply(200, {"classes": CLASSES, "annotations": records})
        self.reply(404, {"error": "Unknown path"})

    def do_POST(self):
        path = unquote(self.path.split("?", 1)[0])
        if not path.startswith("/api/annotations/"):
            return self.reply(404, {"error": "Unknown path"})
        name = path.removeprefix("/api/annotations/")
        if name not in IMAGES:
            return self.reply(404, {"error": "Unknown image"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 100000:
                raise ValueError("Invalid payload size")
            record = json.loads(self.rfile.read(size))
            boxes = record["boxes"]
            if not isinstance(boxes, list) or len(boxes) > 500:
                raise ValueError("Invalid boxes")
            clean = []
            for box in boxes:
                cls = box["class"]
                x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
                if cls not in CLASSES or not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 - x and 0 < h <= 1 - y):
                    raise ValueError("Invalid box")
                clean.append({"class": cls, "x": x, "y": y, "w": w, "h": h})
            result = {"image": name, "boxes": clean, "reviewed": bool(record.get("reviewed", False))}
            target = LABEL_DIR / (Path(name).stem + ".json")
            temp = target.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, indent=2) + "\n")
            temp.replace(target)
            return self.reply(200, {"saved": True})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self.reply(400, {"error": str(exc)})


if __name__ == "__main__":
    if len(IMAGES) != 200:
        raise SystemExit(f"Expected 200 task images, found {len(IMAGES)}")
    print("Manual annotation tool: http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
