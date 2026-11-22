"""Streamlit research workstation for the inference API."""

import io
import json
from datetime import datetime, time, timedelta, timezone
from urllib.parse import quote

import streamlit as st
from client import API, APIError, model_path
from PIL import Image
from visualizations import annotated_image, mask_images

from medical_imaging_inference.evaluations import comparison_key

TASKS = ("classification", "segmentation", "detection")
PAGES = (
    "Overview",
    "Model registry",
    "Deployment",
    "Inference",
    "History",
    "Evaluation",
    "System administration",
)
st.set_page_config(page_title="Medical Imaging Workstation", layout="wide")


def table(rows, empty):
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info(empty)


def overview(api):
    health = api.request("/health")
    st.caption(f"Backend {health['status']}")
    metrics = api.request("/api/v1/system/metrics")
    for column, label, value in zip(
        st.columns(4),
        ("Requests", "Successful", "Failed", "Average latency"),
        (
            metrics["total"],
            metrics["successful"],
            metrics["failed"],
            "—" if metrics["average_latency_ms"] is None else f"{metrics['average_latency_ms']:.1f} ms",
        ),
    ):
        column.metric(label, value)
    if metrics["daily_requests"]:
        st.line_chart(metrics["daily_requests"], x="day", y="count")
        st.bar_chart(metrics["tasks"], x="task", y="count")
    else:
        st.info("No inference has been recorded.")
    st.subheader("Recent requests")
    table(api.request("/api/v1/inference/history", params={"limit": 10})["items"], "No recent activity.")
    st.subheader("Model availability")
    try:
        models = api.request("/api/v1/models")["items"]
        active = api.request("/api/v1/models/active")["items"]
        for column, label, value in zip(
            st.columns(4),
            ("Registered models", "Approved models", "Versions", "Loaded"),
            (
                len({m["model_id"] for m in models}),
                len({m["model_id"] for m in models if m["approved"]}),
                len(models),
                len(active),
            ),
        ):
            column.metric(label, value)
    except APIError as exc:
        st.warning(str(exc))


def registry(api):
    rows = api.request("/api/v1/models")["items"]
    for column, field in zip(st.columns(5), ("task", "framework", "architecture", "version", "approved")):
        values = sorted({row[field] for row in rows if row.get(field) is not None})
        chosen = column.selectbox(
            field.title(), [None, *values], format_func=lambda value: "All" if value is None else str(value)
        )
        if chosen is not None:
            rows = [row for row in rows if row.get(field) == chosen]
    table(rows, "No versions match the filters.")
    if rows:
        selected = st.selectbox(
            "Model details", rows, format_func=lambda item: f"{item['model_id']} · {item['version']}"
        )
        model_id, version = model_path(selected).split("/")
        st.json(api.request(f"/api/v1/models/{model_id}/versions/{version}"))


def choose_model(api, task=None):
    params = {"approved": "true"}
    if task:
        params["task"] = task
    rows = api.request("/api/v1/models", params=params)["items"]
    if not rows:
        st.info("No approved versions are available.")
        return None
    model_id = st.selectbox("Model", sorted({row["model_id"] for row in rows}))
    return st.selectbox(
        "Version",
        [row for row in rows if row["model_id"] == model_id],
        format_func=lambda row: row["version"],
    )


def deployment(api):
    st.subheader("Loaded versions")
    active = api.request("/api/v1/models/active")["items"]
    table(active, "No loaded versions.")
    operation = st.radio("Operation", ("Load", "Unload"), horizontal=True)
    if operation == "Load":
        model = choose_model(api)
    elif active:
        model = st.selectbox(
            "Loaded version", active, format_func=lambda row: f"{row['model_id']} · {row['version']}"
        )
    else:
        model = None
    if model and st.checkbox(f"Confirm {operation.lower()}") and st.button(operation):
        with st.spinner(f"{operation} in progress…"):
            api.request(
                f"/api/v1/models/{operation.lower()}",
                "POST",
                json={key: model[key] for key in ("model_id", "version")},
            )
        st.success(f"{operation} completed. Refresh to see current loaded versions.")


def show_prediction(image, result):
    output = result["output"]
    st.caption(f"{result['model_id']} · {result['model_version']} | {result['latency_ms']:.1f} ms")
    if result["task"] == "classification":
        left, right = st.columns((1, 2))
        left.image(image, caption="Original image")
        right.subheader(", ".join(output["predicted_classes"]) or "No class met the threshold")
        right.bar_chart(output["top_k"], x="label", y="score")
        right.caption("Uncalibrated scores")
    elif result["task"] == "segmentation":
        mask, overlay = mask_images(image, output)
        for column, view, label in zip(
            st.columns(3), (image, mask, overlay), ("Original", "Mask", "Overlay")
        ):
            column.image(view, caption=label)
        st.write("Classes", output["class_labels"] or output["mask_values"])
    else:
        annotated, rows = annotated_image(image, output)
        left, right = st.columns(2)
        left.image(image, caption="Original")
        right.image(annotated, caption="Detections")
        table(rows, "No detections returned.")
    st.download_button(
        "Export prediction metadata",
        json.dumps(result, indent=2),
        file_name=f"{result['prediction_id']}.json",
        mime="application/json",
    )


def inference(api):
    task = st.selectbox("Task", TASKS, format_func=str.title)
    model = choose_model(api, task)
    if model is None:
        return
    uploads = st.file_uploader(
        "Upload images",
        type=("jpg", "jpeg", "png", "tif", "tiff", "webp"),
        accept_multiple_files=True,
    )
    if not uploads:
        return
    if len(uploads) > 8 or sum(item.size for item in uploads) > 40 * 1024 * 1024:
        st.error("A batch accepts up to 8 images and 40 MiB total.")
        return
    files = []
    previews = []
    for upload in uploads:
        if upload.size > 20 * 1024 * 1024:
            st.error("Each image must be at most 20 MiB.")
            return
        content = upload.getvalue()
        try:
            with Image.open(io.BytesIO(content)) as source:
                if source.width * source.height > 40_000_000:
                    st.error("An image exceeds the 40 megapixel limit.")
                    return
                image = source.convert("RGB")
        except (OSError, Image.DecompressionBombError):
            st.error("An image could not be decoded.")
            return
        files.append((upload.name, content, upload.type))
        previews.append(image)
    st.image(previews, caption=[f"Image {i + 1}" for i in range(len(previews))], width=180)
    if st.button("Run inference", type="primary"):
        if len(files) == 1:
            with st.spinner("Validating and processing image…"):
                result = api.request(
                    f"/api/v1/predictions/{model_path(model)}", "POST", files={"image": files[0]}
                )
            show_prediction(previews[0], result)
        else:
            job = api.request(
                "/api/v1/inference/batch",
                "POST",
                data={key: model[key] for key in ("model_id", "version")},
                files=[("images", file) for file in files],
            )
            st.session_state["inference_batch"] = job["batch_id"]
    batch_id = st.session_state.get("inference_batch")
    if batch_id:
        st.button("Refresh batch status")
        job = api.request(f"/api/v1/inference/batch/{batch_id}")
        st.progress(len(job["items"]) / job["total"], text=f"Batch {job['status']}")
        table(job["items"], "Waiting for the first result.")
        completed = [item for item in job["items"] if item["status"] == "completed"]
        if completed:
            item = st.selectbox(
                "Batch prediction", completed, format_func=lambda row: f"Image {row['index'] + 1}"
            )
            st.json(api.request(f"/api/v1/inference/history/{item['prediction_id']}"))


def history(api):
    left, middle, right = st.columns(3)
    model_id = left.text_input("Model ID")
    task = middle.selectbox("Task", ("", *TASKS))
    status = right.selectbox("Status", ("", "completed", "failed"))
    dates = st.date_input("Date range (UTC)", value=())
    filters = {
        key: value for key, value in dict(model_id=model_id, task=task, status=status).items() if value
    }
    if len(dates) == 2:
        filters["start"] = datetime.combine(dates[0], time.min, timezone.utc).isoformat()
        filters["end"] = datetime.combine(dates[1] + timedelta(days=1), time.min, timezone.utc).isoformat()
    rows = api.request("/api/v1/inference/history", params=filters)["items"]
    table(rows, "No matching requests.")
    if rows:
        request_id = st.selectbox("Request details", [row["prediction_id"] for row in rows])
        st.json(api.request(f"/api/v1/inference/history/{quote(request_id, safe='')}"))


def evaluation(api):
    reports = api.request("/api/v1/evaluations")["items"]
    if not reports:
        st.info("No approved evaluation metrics are available for display.")
        return
    selected = st.multiselect(
        "Model versions",
        reports,
        max_selections=4,
        format_func=lambda row: f"{row['model_id']} · {row['version']}",
    )
    if len(selected) > 1:
        keys = [comparison_key(report) for report in selected]
        if keys[0] is None or len(set(keys)) != 1:
            st.warning("These reports have different or unverified datasets or evaluation protocols.")
        else:
            names = set.intersection(*(set(report["metrics"]) for report in selected))
            rows = [
                {
                    "metric": name,
                    **{
                        f"{report['model_id']} · {report['version']}": report["metrics"][name]
                        for report in selected
                    },
                }
                for name in sorted(names)
                if all(isinstance(report["metrics"][name], (int, float)) for report in selected)
            ]
            table(rows, "No shared scalar metrics.")
    for report in selected:
        with st.expander(f"{report['model_id']} · {report['version']}", expanded=True):
            st.json(report)


def administration(api):
    status = api.request("/api/v1/system/status")
    for column, key in zip(st.columns(4), ("backend", "database", "governance", "catalogue_status")):
        column.metric(key.replace("_", " ").title(), status[key])
    st.subheader("Loaded versions")
    table(status.pop("loaded_models"), "No loaded versions.")
    st.subheader("Configuration")
    st.json(status)


st.sidebar.title("Imaging workstation")
page = st.sidebar.radio("Navigation", PAGES)


def clear_batch():
    st.session_state.pop("inference_batch", None)


token = st.sidebar.text_input("OIDC access token", type="password", on_change=clear_batch)
st.title(page)
st.caption("For research use only. Results must not be used as a diagnosis.")
if token:
    try:
        dict(zip(PAGES, (overview, registry, deployment, inference, history, evaluation, administration)))[
            page
        ](API(token))
    except APIError as exc:
        st.error(str(exc))
else:
    st.info("Enter your short-lived OIDC access token to connect.")
