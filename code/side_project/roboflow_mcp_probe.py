"""Probe Roboflow MCP using a key read once from standard input, never saved."""

import json
import os
import sys
import urllib.error
import urllib.request

KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
if not KEY:
    raise SystemExit("No API key supplied")
URL = "https://mcp.roboflow.com/mcp"
HEADERS = {
    "x-api-key": KEY,
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-06-18",
}


def call(payload, session=None):
    headers = dict(HEADERS)
    if session:
        headers["Mcp-Session-Id"] = session
    req = urllib.request.Request(URL, json.dumps(payload).encode(), headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            return r.status, r.headers.get("Mcp-Session-Id"), body
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode()[:400]


status, session, body = call({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "aquarium-side-project", "version": "1.0"}}})
print("initialize_status", status, "session", bool(session), flush=True)
if status != 200:
    print("response", body[:300], flush=True)
    raise SystemExit(1)
status, _, body = call({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, session)
print("tools_status", status, flush=True)
if status == 200:
    data = body.split("data: ", 1)[-1].split("\n", 1)[0] if body.startswith("event:") or body.startswith("data:") else body
    obj = json.loads(data)
    tools = obj.get("result", {}).get("tools", [])
    print("tool_count", len(tools))
    for t in tools:
        if t["name"] in ("annotation_jobs_images_list", "annotation_jobs_review_image", "annotations_get", "images_search"):
            print("TOOL", t["name"], json.dumps(t.get("inputSchema", {})), t.get("description", "")[:1800].encode("ascii", "replace").decode(), flush=True)
    status, _, body = call({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "annotation_jobs_images_list", "arguments": {"project_id": "kevins-workspace-qyqw5/dl_task", "job_id": "D1sCHos9xAXCv8zeDIzq", "limit": 3}}}, session)
    print("images_status", status, body[:3000], flush=True)
else:
    print("response", body[:300])
