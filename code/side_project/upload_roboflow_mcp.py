"""Upload the 200 image-only task ZIP through Roboflow's official MCP server."""

from pathlib import Path
import json
import os
import time
import urllib.request

KEY = os.environ["ROBOFLOW_API_KEY"]
URL = "https://mcp.roboflow.com/mcp"
PROJECT = "kevins-workspace-qyqw5/dl_task"
ROOT = Path(__file__).resolve().parent
ZIP = ROOT.parent.parent / 'data/side_project' / os.environ.get("ROBOFLOW_UPLOAD_ZIP", "aquarium_200_images_for_roboflow.zip")
ANNOTATED = os.environ.get("ROBOFLOW_ANNOTATED", "0") == "1"


def request(payload, session=None):
    headers = {"x-api-key": KEY, "Content-Type": "application/json", "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
    if session:
        headers["Mcp-Session-Id"] = session
    req = urllib.request.Request(URL, json.dumps(payload).encode(), headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        body = response.read().decode()
        lines = [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]
        data = json.loads(lines[-1] if lines else body)
        return response.headers.get("Mcp-Session-Id"), data


def tool(name, args, session):
    _, packet = request({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": args}}, session)
    result = packet.get("result", {})
    if result.get("isError") or "error" in packet:
        raise RuntimeError(json.dumps(packet)[:1000])
    return result.get("structuredContent") or json.loads(result["content"][0]["text"])


def main():
    session, packet = request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "aquarium-side-project", "version": "1.0"}}})
    if "result" not in packet:
        raise RuntimeError("MCP initialization failed")
    print("MCP connected", flush=True)
    prepared = tool("image_upload", {"project_id": PROJECT, "split": "train", "batch_name": "Aquarium manual 200", "tag_names": ["aquarium-manual-200"], "annotation_overwrite": False}, session)
    signed = prepared.get("signedUrl") or prepared.get("signed_url")
    task_id = prepared.get("taskId") or prepared.get("task_id")
    if not signed or not task_id:
        raise RuntimeError("Upload preparation did not provide a signed URL and task ID: " + str(list(prepared)))
    print("Upload task prepared", flush=True)
    body = ZIP.read_bytes()
    req = urllib.request.Request(signed, body, {"Content-Type": "application/zip"}, method="PUT")
    with urllib.request.urlopen(req, timeout=180) as response:
        print("ZIP PUT status", response.status, "bytes", len(body), flush=True)
    for n in range(120):
        state = tool("image_upload_status", {"task_id": task_id}, session)
        status = state.get("status")
        if n % 4 == 0 or status in ("completed", "failed"):
            print("status", status, "progress", state.get("progress"), flush=True)
        if status in ("completed", "failed"):
            safe = {k: v for k, v in state.items() if k not in ("signedUrl", "signed_url")}
            name = "roboflow_annotation_import_result.json" if ANNOTATED else "roboflow_upload_result.json"
            result_dir = ROOT.parent.parent / 'results/side_project'
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / name).write_text(json.dumps(safe, indent=2) + "\n")
            print("result", json.dumps(safe)[:3000], flush=True)
            if status == "failed":
                raise RuntimeError("Roboflow upload failed")
            return
        time.sleep(5)
    raise TimeoutError("Upload still processing; task ID " + task_id)


if __name__ == "__main__":
    main()
