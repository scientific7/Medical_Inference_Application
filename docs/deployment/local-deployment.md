# Local deployment

Use Python 3.10 and create a virtual environment. Install the application and
dashboard with `python -m pip install -e '.[test,frontend]'`. Add the framework
extra for the model packages you intend to serve.

Configure the backend environment and run
`uvicorn medical_imaging_inference.app:app --host 127.0.0.1`. In another
terminal, run `streamlit run apps/frontend/app.py`. Use the configured OIDC
provider to obtain the API access token. Point the backend at the separate
governance service and configure registry discovery. Use
`python -m pytest -q` to check the local implementation with synthetic inputs.
