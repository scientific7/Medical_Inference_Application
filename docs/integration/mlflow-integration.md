# MLflow integration

The inference registry pages use the paginated MLflow model version API.
Set `MLFLOW_TRACKING_URI` and provide the scoped tracking token when needed.
Tag registry versions with `governance.model_id` and `governance.version`
if the identifiers differ from MLflow. Each entry is checked independently
with governance. The upstream run tracker does not publish these mappings
automatically. See [registry discovery](../INTEGRATION.md#registry-discovery).
