#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 6 ]]; then
  echo "Usage: deploy.sh REGION SECRET_ARN ARTIFACT_ROOT LOG_GROUP BACKEND_DIGEST FRONTEND_DIGEST" >&2
  exit 2
fi
region=$1
secret=$2
artifact_root=$3
log_group=$4
backend=$5
frontend=$6
for image in "$backend" "$frontend"; do
  if [[ ! "$image" =~ ^[[:alnum:].:/_-]+@sha256:[a-f0-9]{64}$ ]]; then
    echo "An immutable image digest is required" >&2
    exit 2
  fi
  docker pull "$image"
done

install -d -m 0700 /etc/medical-imaging
exec 9>/etc/medical-imaging/deploy.lock
flock 9
env_file=$(mktemp /etc/medical-imaging/backend.XXXXXX)
trap 'rm -f "$env_file"' EXIT
aws --region "$region" secretsmanager get-secret-value --secret-id "$secret" \
  --query SecretString --output text > "$env_file"
docker network inspect medical-imaging >/dev/null 2>&1 || docker network create medical-imaging
docker volume create medical-imaging-state >/dev/null

replace_container() {
  local name=$1
  shift
  if docker container inspect "$name" >/dev/null 2>&1; then
    docker rm -f "$name" >/dev/null
  fi
  docker run -d --name "$name" --restart unless-stopped --network medical-imaging \
    --read-only --tmpfs /tmp:mode=1777 --security-opt no-new-privileges \
    --cap-drop ALL --log-driver awslogs --log-opt "awslogs-region=$region" \
    --log-opt "awslogs-group=$log_group" "$@"
}

replace_container medical-imaging-api --env-file "$env_file" \
  -e INFERENCE_ENVIRONMENT=production -e "INFERENCE_S3_ARTIFACT_ROOT=$artifact_root" \
  -v medical-imaging-state:/var/lib/inference "$backend"
replace_container medical-imaging-frontend -p 8501:8501 \
  -e INFERENCE_API_URL=http://medical-imaging-api:8000 "$frontend"

for path in health _stcore/health; do
  if [[ "$path" == health ]]; then
    docker exec medical-imaging-api python -c "import time, urllib.request; time.sleep(2); urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).close()"
  else
    curl --fail --silent --show-error --retry 10 --retry-connrefused "http://127.0.0.1:8501/$path"
  fi
done
