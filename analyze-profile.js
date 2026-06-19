/**
 * analyze-profile.js
 * ====================
 * Vercel serverless function (Node runtime).
 *
 * Why this file exists:
 * The browser can never be trusted with the RXGUARD_API_KEY — anything
 * shipped in frontend JS is visible to every visitor via devtools/Network
 * tab. This function runs server-side only. It holds the real key as a
 * private environment variable (set in the Vercel dashboard, never
 * committed, never sent to the browser) and forwards the request to the
 * actual RxGuard backend, then relays the response back to the browser.
 *
 * Deploy: place this whole `serverless-proxy/` folder (or just the
 * `api/` directory) alongside your frontend HTML in a Vercel project.
 * Vercel auto-detects anything under /api as a serverless function —
 * no extra config needed for this simple case.
 *
 * Required environment variables (set in Vercel: Project -> Settings -> Environment Variables):
 *   RXGUARD_API_KEY      — the same 64-char hex key your backend's .env uses
 *   RXGUARD_BACKEND_URL  — e.g. https://rxguard-api.onrender.com
 */

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ status: 'error', message: 'Method not allowed.' });
  }

  const backendUrl = process.env.RXGUARD_BACKEND_URL;
  const apiKey = process.env.RXGUARD_API_KEY;

  if (!backendUrl || !apiKey) {
    console.error('Proxy misconfigured: RXGUARD_BACKEND_URL or RXGUARD_API_KEY is not set.');
    return res.status(500).json({
      status: 'error',
      message: 'Server is misconfigured. Contact the administrator.',
    });
  }

  try {
    const upstream = await fetch(`${backendUrl}/api/analyze-profile`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,
      },
      body: JSON.stringify(req.body),
    });

    const data = await upstream.json();
    return res.status(upstream.status).json(data);
  } catch (err) {
    console.error('Proxy -> backend request failed:', err);
    return res.status(502).json({
      status: 'error',
      message: 'Could not reach the inference backend. Please try again shortly.',
    });
  }
}
