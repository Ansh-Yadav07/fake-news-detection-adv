const DEFAULT_API_BASE = 'http://127.0.0.1:5001';

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
