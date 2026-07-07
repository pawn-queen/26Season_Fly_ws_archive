#!/usr/bin/env bash
set -Eeuo pipefail

SESSION="${SESSION:-uav_sim}"
PX4_DIR="${PX4_DIR:-$HOME/uav/PX4-1.17.0-2026_Season}"
WS_DIR="${WS_DIR:-$HOME/uav/26Season_Fly_ws_archive}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

PX4_TARGET="${PX4_TARGET:-px4_sitl gz_x500_depth}"
AGENT_PORT="${AGENT_PORT:-8888}"

AGENT_DELAY="${AGENT_DELAY:-3}"
BRIDGE_DELAY="${BRIDGE_DELAY:-6}"
DETECT_DELAY="${DETECT_DELAY:-9}"
CONTROL_DELAY="${CONTROL_DELAY:-12}"

DETECT_SHOW_IMAGE="${DETECT_SHOW_IMAGE:-true}"
DETECT_ARGS="${DETECT_ARGS:-}"
CONTROL_ARGS="${CONTROL_ARGS:-}"
CONTROL_HEADLESS="${CONTROL_HEADLESS:-false}"

ATTACH_AFTER_START=true
RESTART_EXISTING=false

usage() {
    cat <<'EOF'
Usage:
  scripts/sim_stack.sh [start] [options]
  scripts/sim_stack.sh stop
  scripts/sim_stack.sh restart [options]
  scripts/sim_stack.sh attach
  scripts/sim_stack.sh status

Options:
  --no-attach                 Start tmux session but do not attach to it.
  --restart-existing          Kill an existing session with the same name first.
  --session NAME              tmux session name. Default: uav_sim
  --px4-dir DIR               PX4 source directory.
  --ws-dir DIR                ROS 2 workspace directory.
  --detect-show-image BOOL    true/false. Default: true
  --control-headless          Add --headless to ros2 run control test.
  --detect-args "ARGS"        Extra args appended after detect node --ros-args.
  --control-args "ARGS"       Extra args appended after ros2 run control test.

Environment overrides:
  PX4_TARGET, AGENT_PORT, AGENT_DELAY, BRIDGE_DELAY, DETECT_DELAY, CONTROL_DELAY

Examples:
  scripts/sim_stack.sh start
  scripts/sim_stack.sh start --control-headless --detect-show-image false
  DETECT_ARGS="-p weights_path:=/abs/model.pt" scripts/sim_stack.sh restart
  CONTROL_ARGS="--recon-search-timeout 12" scripts/sim_stack.sh start
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

quote() {
    printf '%q' "$1"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

session_exists() {
    tmux has-session -t "$SESSION" >/dev/null 2>&1
}

validate_paths() {
    require_cmd tmux
    require_cmd make
    require_cmd MicroXRCEAgent

    [[ -d "$PX4_DIR" ]] || die "PX4_DIR does not exist: $PX4_DIR"
    [[ -d "$WS_DIR" ]] || die "WS_DIR does not exist: $WS_DIR"
    [[ -f "$ROS_SETUP" ]] || die "ROS setup file does not exist: $ROS_SETUP"
    [[ -f "$WS_DIR/install/setup.bash" ]] || die "Workspace setup not found. Build first: cd $WS_DIR && colcon build"
    [[ -x "$WS_DIR/bridge.sh" ]] || die "bridge.sh is missing or not executable: $WS_DIR/bridge.sh"
}

tmux_run() {
    local target="$1"
    local command="$2"
    tmux send-keys -t "$target" "bash -lc $(quote "$command")" C-m
}

create_window() {
    local name="$1"
    tmux new-window -t "$SESSION:" -n "$name"
}

stop_stack() {
    if session_exists; then
        tmux kill-session -t "$SESSION"
        echo "Stopped tmux session: $SESSION"
    else
        echo "No tmux session found: $SESSION"
    fi
}

status_stack() {
    if session_exists; then
        echo "Session '$SESSION' is running."
        tmux list-windows -t "$SESSION"
    else
        echo "Session '$SESSION' is not running."
    fi
}

attach_stack() {
    session_exists || die "No tmux session found: $SESSION"
    tmux attach-session -t "$SESSION"
}

start_stack() {
    validate_paths

    if session_exists; then
        if [[ "$RESTART_EXISTING" == true ]]; then
            stop_stack
        else
            die "tmux session '$SESSION' already exists. Use restart, stop, attach, or --restart-existing."
        fi
    fi

    local ws_setup
    local control_headless_arg=""
    ws_setup="$WS_DIR/install/setup.bash"
    if [[ "$CONTROL_HEADLESS" == true ]]; then
        control_headless_arg="--headless"
    fi

    tmux new-session -d -s "$SESSION" -n px4
    tmux set-window-option -t "$SESSION" remain-on-exit on >/dev/null

    tmux_run "$SESSION:px4" \
        "cd $(quote "$PX4_DIR") && make ${PX4_TARGET}"

    create_window agent
    tmux_run "$SESSION:agent" \
        "sleep ${AGENT_DELAY}; MicroXRCEAgent udp4 -p ${AGENT_PORT}"

    create_window bridge
    tmux_run "$SESSION:bridge" \
        "sleep ${BRIDGE_DELAY}; source $(quote "$ROS_SETUP"); cd $(quote "$WS_DIR"); ./bridge.sh"

    create_window detect
    tmux_run "$SESSION:detect" \
        "sleep ${DETECT_DELAY}; source $(quote "$ROS_SETUP"); source $(quote "$ws_setup"); cd $(quote "$WS_DIR"); ros2 run detect test --ros-args -p show_image:=${DETECT_SHOW_IMAGE} ${DETECT_ARGS}"

    create_window control
    tmux_run "$SESSION:control" \
        "sleep ${CONTROL_DELAY}; source $(quote "$ROS_SETUP"); source $(quote "$ws_setup"); cd $(quote "$WS_DIR"); ros2 run control test ${control_headless_arg} ${CONTROL_ARGS}"

    tmux select-window -t "$SESSION:px4"

    echo "Started tmux session: $SESSION"
    echo "Windows: px4, agent, bridge, detect, control"
    echo "Attach: scripts/sim_stack.sh attach"
    echo "Stop:   scripts/sim_stack.sh stop"

    if [[ "$ATTACH_AFTER_START" == true ]]; then
        tmux attach-session -t "$SESSION"
    fi
}

action="${1:-start}"
if [[ $# -gt 0 ]]; then
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-attach)
            ATTACH_AFTER_START=false
            shift
            ;;
        --restart-existing)
            RESTART_EXISTING=true
            shift
            ;;
        --session)
            SESSION="${2:?missing value for --session}"
            shift 2
            ;;
        --px4-dir)
            PX4_DIR="${2:?missing value for --px4-dir}"
            shift 2
            ;;
        --ws-dir)
            WS_DIR="${2:?missing value for --ws-dir}"
            shift 2
            ;;
        --detect-show-image)
            DETECT_SHOW_IMAGE="${2:?missing value for --detect-show-image}"
            shift 2
            ;;
        --control-headless)
            CONTROL_HEADLESS=true
            shift
            ;;
        --detect-args)
            DETECT_ARGS="${2:?missing value for --detect-args}"
            shift 2
            ;;
        --control-args)
            CONTROL_ARGS="${2:?missing value for --control-args}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

case "$action" in
    start)
        start_stack
        ;;
    restart)
        RESTART_EXISTING=true
        start_stack
        ;;
    stop)
        stop_stack
        ;;
    attach)
        attach_stack
        ;;
    status)
        status_stack
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        die "Unknown action: $action"
        ;;
esac
