#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
CONFIG_FILE="${VIITORVOICE_V2_DEPLOY_CONFIG:-${SCRIPT_DIR}/viitorvoice/grpc_server/deploy.env}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy no_proxy

usage() {
  cat <<USAGE
Usage:
  $0 [--config FILE] <start|stop|restart|status|logs> [encoder|llm|decoder|orchestrator|http|provider|all] [-- extra server args]

Examples:
  $0 start all
  $0 start llm -- --no-warmup
  $0 stop encoder
  $0 status all
  $0 logs orchestrator
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing config file: ${CONFIG_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

validate_cuda_device_id() {
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  local device_id="${VIITORVOICE_DEVICE_ID:-0}"
  local count=0
  local item trimmed

  if [[ -z "${visible}" || "${visible}" == "-1" || "${visible}" == "NoDevFiles" ]]; then
    return
  fi
  if [[ ! "${device_id}" =~ ^[0-9]+$ ]]; then
    echo "VIITORVOICE_DEVICE_ID must be a non-negative integer, got: ${device_id}" >&2
    exit 1
  fi

  IFS=',' read -r -a visible_devices <<< "${visible}"
  for item in "${visible_devices[@]}"; do
    trimmed="${item//[[:space:]]/}"
    if [[ -n "${trimmed}" ]]; then
      count=$((count + 1))
    fi
  done
  if [[ "${count}" -gt 0 && "${device_id}" -ge "${count}" ]]; then
    echo "VIITORVOICE_DEVICE_ID=${device_id} is out of range for CUDA_VISIBLE_DEVICES=${visible}." >&2
    echo "Device ids are logical after CUDA_VISIBLE_DEVICES remapping; valid ids: 0..$((count - 1))." >&2
    echo "For a single selected GPU, use: CUDA_VISIBLE_DEVICES=<physical_gpu> VIITORVOICE_DEVICE_ID=0." >&2
    exit 1
  fi
}

validate_cuda_device_id

PYTHON_BIN="${VIITORVOICE_V2_VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${VIITORVOICE_V2_VENV_DIR}. Create it with:" >&2
  echo "  cd ${PROJECT_ROOT}" >&2
  echo "  uv venv .venv --python 3.12" >&2
  exit 1
fi

ACTION="${1:-}"
SERVICE="${2:-all}"
if [[ -z "${ACTION}" ]]; then
  usage
  exit 1
fi
shift || true
if [[ $# -gt 0 ]]; then
  shift || true
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_ARGS=("$@")

mkdir -p "${VIITORVOICE_V2_STATE_DIR}" "${VIITORVOICE_V2_LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

services_for() {
  case "${1}" in
    all) printf '%s\n' encoder llm decoder orchestrator http ;;
    encoder|llm|decoder|orchestrator|http|provider) printf '%s\n' "${1}" ;;
    *)
      echo "Unknown service: ${1}" >&2
      exit 1
      ;;
  esac
}

module_for() {
  case "${1}" in
    encoder) echo "viitorvoice.grpc_server.encoder.server" ;;
    llm) echo "viitorvoice.grpc_server.llm.server" ;;
    decoder) echo "viitorvoice.grpc_server.decoder.server" ;;
    orchestrator) echo "viitorvoice.grpc_server.orchestrator.server" ;;
    http) echo "viitorvoice.grpc_server.http.server" ;;
    provider) echo "viitorvoice.grpc_server.provider.server" ;;
  esac
}

port_for() {
  case "${1}" in
    encoder) echo "${VIITORVOICE_V2_ENCODER_PORT}" ;;
    llm) echo "${VIITORVOICE_V2_LLM_PORT}" ;;
    decoder) echo "${VIITORVOICE_V2_DECODER_PORT}" ;;
    orchestrator) echo "${VIITORVOICE_V2_ORCH_PORT}" ;;
    http) echo "${VIITORVOICE_V2_HTTP_PORT}" ;;
    provider) echo "${VIITORVOICE_V2_PROVIDER_PORT}" ;;
  esac
}

pid_file_for() {
  echo "${VIITORVOICE_V2_STATE_DIR}/${1}.pid"
}

log_file_for() {
  echo "${VIITORVOICE_V2_LOG_DIR}/${1}.log"
}

is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" >/dev/null 2>&1
}

start_one() {
  local service="$1"
  local pid_file log_file module port
  pid_file="$(pid_file_for "${service}")"
  log_file="$(log_file_for "${service}")"
  module="$(module_for "${service}")"
  port="$(port_for "${service}")"

  if is_running "${pid_file}"; then
    echo "${service} already running: pid $(cat "${pid_file}")"
    return
  fi
  rm -f "${pid_file}"

  echo "Starting ${service} on ${VIITORVOICE_V2_SERVICE_HOST}:${port}"
  setsid "${PYTHON_BIN}" -m "${module}" \
    --host "${VIITORVOICE_V2_SERVICE_HOST}" \
    --port "${port}" \
    --log-level "${VIITORVOICE_V2_LOG_LEVEL}" \
    "${EXTRA_ARGS[@]}" \
    >"${log_file}" 2>&1 < /dev/null &
  echo "$!" >"${pid_file}"
  echo "${service} pid $(cat "${pid_file}"), log ${log_file}"
}

stop_one() {
  local service="$1"
  local pid_file pid
  pid_file="$(pid_file_for "${service}")"
  if ! is_running "${pid_file}"; then
    rm -f "${pid_file}"
    echo "${service} not running"
    return
  fi
  pid="$(cat "${pid_file}")"
  echo "Stopping ${service}: pid ${pid}"
  kill "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      rm -f "${pid_file}"
      echo "${service} stopped"
      return
    fi
    sleep 1
  done
  echo "Force stopping ${service}: pid ${pid}"
  kill -9 "${pid}" >/dev/null 2>&1 || true
  rm -f "${pid_file}"
}

status_one() {
  local service="$1"
  local pid_file
  pid_file="$(pid_file_for "${service}")"
  if is_running "${pid_file}"; then
    echo "${service}: running pid $(cat "${pid_file}")"
  else
    echo "${service}: stopped"
  fi
}

logs_one() {
  local service="$1"
  local log_file
  log_file="$(log_file_for "${service}")"
  if [[ ! -f "${log_file}" ]]; then
    echo "No log for ${service}: ${log_file}" >&2
    return 1
  fi
  tail -n "${VIITORVOICE_V2_LOG_LINES:-120}" "${log_file}"
}

case "${ACTION}" in
  start)
    while IFS= read -r item; do start_one "${item}"; done < <(services_for "${SERVICE}")
    ;;
  stop)
    if [[ "${SERVICE}" == "all" ]]; then
      for item in http orchestrator decoder llm encoder; do stop_one "${item}"; done
    else
      stop_one "${SERVICE}"
    fi
    ;;
  restart)
    "${BASH_SOURCE[0]}" --config "${CONFIG_FILE}" stop "${SERVICE}"
    "${BASH_SOURCE[0]}" --config "${CONFIG_FILE}" start "${SERVICE}" -- "${EXTRA_ARGS[@]}"
    ;;
  status)
    while IFS= read -r item; do status_one "${item}"; done < <(services_for "${SERVICE}")
    ;;
  logs)
    while IFS= read -r item; do logs_one "${item}"; done < <(services_for "${SERVICE}")
    ;;
  *)
    usage
    exit 1
    ;;
esac
