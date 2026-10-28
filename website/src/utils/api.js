const DEFAULT_API_BASE = 'http://127.0.0.1:5001';
// Generous default: free-tier hosts can take up to a minute to wake from sleep
const DEFAULT_ANALYZE_TIMEOUT_MS = 90000;

/**
 * Strip a trailing "/predict" (legacy VITE_API_URL format) from a URL,
 * returning the origin + any remaining base path.
 */
const stripPredictPath = (value) => {
  try {
    const url = new URL(value);
    url.pathname = url.pathname.replace(/\/predict\/?$/, '') || '/';
    url.search = '';
    url.hash = '';
    return url.toString();
  } catch {
    return null;
  }
};

/** Resolve the backend base URL from env (VITE_API_BASE > VITE_API_URL > default). */
export const getApiBase = () => {
  const explicit = import.meta.env.VITE_API_BASE?.trim();
  if (explicit) return explicit;

  const legacy = import.meta.env.VITE_API_URL?.trim();
  if (legacy) {
    const stripped = stripPredictPath(legacy);
    if (stripped) return stripped;
  }

  return DEFAULT_API_BASE;
};

/**
 * Join an endpoint onto the API base. Tolerates trailing slashes on the base
 * and preserves any base path (e.g. https://host/api + "analyze" -> https://host/api/analyze).
 */
export const apiUrl = (endpoint, base = getApiBase()) => {
  const normalizedBase = base.endsWith('/') ? base : `${base}/`;
  const path = String(endpoint).replace(/^\/+/, '');
  return new URL(path, normalizedBase).toString();
};

/** Client-side limit for /analyze (VITE_ANALYZE_TIMEOUT_MS, default 90s). */
export const getAnalyzeTimeoutMs = () => {
  const value = Number(import.meta.env.VITE_ANALYZE_TIMEOUT_MS);
  return Number.isFinite(value) && value > 0 ? value : DEFAULT_ANALYZE_TIMEOUT_MS;
};

let warmUpStarted = false;

/**
 * Fire-and-forget GET to the health route so a sleeping backend starts waking
 * while the user is still typing. Runs once per page load; the response is
 * opaque (no-cors) and ignored.
 */
export const warmUpBackend = () => {
  if (warmUpStarted) return;
  warmUpStarted = true;
  fetch(apiUrl('/'), { mode: 'no-cors', cache: 'no-store' }).catch(() => {});
};
