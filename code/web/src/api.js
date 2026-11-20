export async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { const error = await response.json(); detail = typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail); } catch { /* Keep the HTTP status when no JSON error is available. */ }
    throw new Error(detail);
  }
  return response.json();
}
export const post = value => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) });
export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('The saved image could not be opened.'));
    image.src = src;
  });
}
