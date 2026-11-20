"""Optional vision-model crack suggestions; coordinates are a review draft."""
import base64
import json
import os
import time
import httpx
from PIL import Image, ImageDraw

from fastapi import HTTPException


def suggest_cracks(image, prediction_mask):
    key = os.environ.get('GROQ_API_KEY')
    if not key:
        raise HTTPException(503, 'Set GROQ_API_KEY in the server environment and restart CrackLab to use AI suggestions')

    def data_url(value):
        import io
        buffer = io.BytesIO()
        value.save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')

    preview = image.copy()
    preview.thumbnail((1024, 1024))
    existing = prediction_mask.copy()
    existing.thumbnail(preview.size)
    # A single comparison image avoids provider-side multi-image parsing issues.
    comparison = Image.new('RGB', (preview.width * 2, preview.height + 32), 'white')
    comparison.paste(preview.convert('RGB'), (0, 32))
    comparison.paste(existing.convert('RGB'), (preview.width, 32))
    labels = ImageDraw.Draw(comparison)
    labels.text((8, 8), 'ROAD IMAGE', fill='black')
    labels.text((preview.width + 8, 8), 'CURRENT MASK', fill='black')
    configured_model = os.environ.get('CRACKLAB_VISION_MODEL')
    candidates = [configured_model] if configured_model else [
        'qwen/qwen3.8-27b',
    ]
    base_payload = {
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Compare the road photo (first image) with the existing white-on-black crack mask (second image). Find visible pavement cracks missed by the mask. Return JSON with paths, an array of objects each containing points and width, plus a short note. Each point is [x,y] using integers 0 to 1000 relative to the full image. width is an integer from 1 to 30 in the same scale. Trace visible crack centerlines only; avoid seams, shadows, markings, grass and guesses. Keep paths short enough to follow bends. If there are no clear missed cracks, return an empty paths array. These are draft suggestions for human review.'},
            {'type': 'image_url', 'image_url': {'url': data_url(comparison)}},
        ]}],
        'response_format': {'type': 'json_object'},
        'max_completion_tokens': 2500,
    }
    last_error = None
    result = None
    selected_model = candidates[0]
    for model in candidates:
        payload = {**base_payload, 'model': model}
        try:
            for attempt in range(3):
                try:
                    response = httpx.post('https://api.groq.com/openai/v1/chat/completions', json=payload,
                                          headers={'Authorization': 'Bearer ' + key}, timeout=90)
                    response.raise_for_status()
                    result = response.json()
                    selected_model = model
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 502 and attempt < 2:
                        time.sleep(1 + attempt * 2)
                        continue
                    raise
            if result is not None:
                break
        except httpx.HTTPStatusError as exc:
            try:
                error_body = exc.response.json()
                detail = error_body.get('error', {}).get('message') or json.dumps(error_body)[:300]
            except (ValueError, AttributeError):
                detail = exc.response.text[:300]
            last_error = f'Vision API returned HTTP {exc.response.status_code}: {str(detail)[:300]}'
            if exc.response.status_code == 400:
                fallback_payload = dict(payload)
                fallback_payload.pop('response_format', None)
                try:
                    fallback = httpx.post('https://api.groq.com/openai/v1/chat/completions', json=fallback_payload,
                                          headers={'Authorization': 'Bearer ' + key}, timeout=90)
                    fallback.raise_for_status()
                    result = fallback.json()
                    selected_model = model
                    break
                except httpx.HTTPStatusError as fallback_exc:
                    try:
                        fallback_body = fallback_exc.response.json()
                        fallback_detail = fallback_body.get('error', {}).get('message') or json.dumps(fallback_body)[:300]
                    except (ValueError, AttributeError):
                        fallback_detail = fallback_exc.response.text[:300]
                    last_error = f'Vision API returned HTTP {fallback_exc.response.status_code}: {str(fallback_detail)[:300]}'
                except httpx.RequestError as fallback_exc:
                    last_error = 'Vision API could not be reached'
            if exc.response.status_code not in (400, 404, 429, 502, 503) or model == candidates[-1]:
                raise HTTPException(502, last_error) from exc
        except httpx.RequestError as exc:
            last_error = 'Vision API could not be reached'
            if model == candidates[-1]:
                raise HTTPException(502, last_error) from exc
    if result is None:
        raise HTTPException(502, last_error or 'Vision API returned no result')
    try:
        output = result['choices'][0]['message']['content']
        parsed = json.loads(output)
        paths = []
        raw_paths = parsed.get('paths', [])
        if not isinstance(raw_paths, list):
            raw_paths = []
        for path in raw_paths[:80]:
            if not isinstance(path, dict) or not isinstance(path.get('points'), list):
                continue
            points = path['points'][:100]
            if len(points) >= 2 and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(n, (int, float)) and 0 <= n <= 1000 for n in p) for p in points):
                normalized = [[int(round(n)) for n in point] for point in points]
                try:
                    width = max(1, min(30, int(path.get('width', 8))))
                except (TypeError, ValueError):
                    width = 8
                paths.append({'points': normalized, 'width': width})
        return {'paths': paths, 'note': str(parsed.get('note', 'AI draft ready for review.'))[:500], 'model': selected_model}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise HTTPException(502, f'Vision API returned an invalid suggestion ({type(exc).__name__}: {str(exc)[:120]})') from exc
