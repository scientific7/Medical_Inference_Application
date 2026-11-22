# Imaging workstation

The Streamlit frontend uses the authenticated FastAPI interface for overview,
approved model selection, inference, deployment, and request history. The
visualization module maps returned prediction metadata onto the image held in
the current browser session.

Install with `pip install -e '.[frontend]'` and start the backend separately.
Run `streamlit run apps/frontend/app.py`. Set `INFERENCE_API_URL` to the backend
origin when it differs from `http://127.0.0.1:8000`. Sign in with a short-lived
OIDC API access token. Deployment mutations require the administrator role.
Use a TLS ingress for deployed frontend sessions.

Run `pytest tests/test_frontend.py` with the frontend extra installed.
