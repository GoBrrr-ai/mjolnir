#!/usr/bin/env bash
#
# setup-thor.sh — Jetson AGX Thor (sm_110a) host pre-configuration.
#
# Ported from the ~/thor repo's scripts/setup.sh + utils.sh, made
# self-contained and Thor-only (SoM=thor, 128 GB, NVIDIA carrier — the
# constants setup.sh hardcoded): no external ROOT/scripts dependency, no
# CI branches. This is for a physical box and needs sudo. Every step is
# idempotent — re-running where already configured is a no-op.
#
# The fan step also installs the two tuned performance fan profiles
# (`recommended` + `max`, per-profile gated, with the .bck rollback point)
# and selects the default — see the FAN section below.
#
# Preferred entry point: `mjolnir hw setup` (maps CLI flags to the options
# below):
#
#   -y, --yes              don't prompt before starting
#   --no-upgrade           skip apt update + full-upgrade (default: run)
#   --keep-gui             don't switch the boot target to multi-user
#   --fan-profile <p>      recommended (default) | max | cool | quiet
#   --swap-size <GB>       swap file size (default: 32)
#   --skip <a,b,c>         skip steps: gui,upgrade,pip,docker,jtop,nsight,
#                          memory,fan,host,clocks,power
#   --reboot               reboot at the end if a step required one
#   --dry-run              print the plan, change nothing

set -euo pipefail

# ── Thor constants (from setup.sh's hardcoded block) ─────────────────────────
SOC_TYPE="thor"
SOC_MEMORY=128
IS_NVIDIA_CARRIER=true

REBOOT_REQUIRED=0

ASSUME_YES=0
UPGRADE=1
KEEP_GUI=0
FAN_PROFILE="recommended"
SWAP_SIZE=32
SKIP=""
DO_REBOOT=0
DRY_RUN=0

usage() {
  grep '^#   ' "$0" | sed 's/^#   //'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1; shift ;;
    --no-upgrade) UPGRADE=0; shift ;;
    --keep-gui) KEEP_GUI=1; shift ;;
    --fan-profile) FAN_PROFILE="${2:?}"; shift 2 ;;
    --swap-size) SWAP_SIZE="${2:?}"; shift 2 ;;
    --skip) SKIP="${2:?}"; shift 2 ;;
    --reboot) DO_REBOOT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "❌ unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# ================================================================================
# UTILITIES
# ================================================================================

is_true() {
    [[ "${1,,}" =~ ^(true|1|yes|enabled|y)$ ]]
}

file_exists() {
    local file=${1:-}
    [[ -n "$file" && -f "$file" ]]
}

is_command_available() {
    local command=$1
    if command -v "${command}" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

is_user_in_group() {
    getent group "$1" | grep -qw "$USER"
}

# Convert size string to bytes (e.g., "4G" to bytes)
convert_to_bytes() {
    local size=$1
    local value=${size%[GM]}
    local unit=${size#$value}

    case $unit in
        G)
            echo $((value * 1024 * 1024 * 1024))
            ;;
        M)
            echo $((value * 1024 * 1024))
            ;;
        *)
            echo "Error: Invalid size unit. Use M for MB or G for GB"
            exit 1
            ;;
    esac
}

is_skipped() {
    case ",${SKIP}," in *",${1},"*) return 0 ;; esac
    return 1
}

ask_yes_no() {
    while true; do
        read -p "❔ ${1} (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer Y or y for YES or N or n for NO.";;
        esac
    done
}

ask_should_proceed() {
    echo
    while true; do
        read -p "❔ Do you want to continue? (Press ENTER for Yes, 'q' to Quit) " input
        if [ -z "$input" ] || is_true "$input"; then
            echo "✅ Continuing..."
            echo
            return 0
        elif [[ "$input" == "q" ]]; then
            echo "❌ Exiting..."
            exit 0
        else
            echo "❌ Invalid input: ${input}. Please press ENTER to continue or 'q' to quit."
        fi
    done
}

check_systemd() {
    if ! is_command_available systemctl; then
        echo "Error: This script requires systemd"
        exit 1
    fi
}

systemctl_stop_service() {
  check_systemd
  sudo systemctl stop "$1"
}

systemctl_enable_service() {
  check_systemd
  sudo systemctl enable --now "$1"
}

systemctl_start_service() {
  check_systemd
  sudo systemctl start "$1"
}

systemctl_restart_service() {
  check_systemd
  sudo systemctl restart "$1"
}

systemctl_service_exists() {
  check_systemd

  # status exit code 4 = unit not found
  systemctl status "$1" &>/dev/null
  [[ $? -ne 4 ]]
}

# Return 0 if the unit is active (running), 1 otherwise
systemctl_service_is_active() {
  check_systemd
  systemctl is-active --quiet "$1"
}

# Return 0 if the unit is enabled, 1 otherwise
systemctl_service_is_enabled() {
  check_systemd
  systemctl is-enabled --quiet "$1"
}

systemctl_disable_service() {
    if ! systemctl_service_exists "$1"; then
        return 0
    fi

    if systemctl_service_is_active "$1"; then
        echo "ℹ️  Stopping the $1 service..."
        systemctl_stop_service "$1"
    fi

    if systemctl_service_is_enabled "$1"; then
        echo "ℹ️  Disabling the $1 service..."
        sudo systemctl disable "$1"
    fi
}

ensure_installed() {
    # Usage: ensure_installed <command> <package>
    local cmd="$1" pkg="${2:-$1}"

    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "📦 Installing $pkg..."
        sudo apt update
        sudo apt install -y "$pkg"
    else
        echo "✅ $pkg is already installed at $(command -v "$cmd")"
    fi
}

l4t_root_device() {
    local root_device
    root_device=$(findmnt -n -o SOURCE /)

    if [[ $root_device == *"nvme"* ]]; then
        echo "nvme"
    elif [[ $root_device == *"mmcblk"* ]]; then
        # eMMC devices have mmcblk0boot0 and mmcblk0boot1 partitions
        if [[ -b /dev/mmcblk0boot0 ]] && [[ -b /dev/mmcblk0boot1 ]]; then
            echo "emmc"
        else
            echo "sdcard"
        fi
    elif [[ $root_device == *"/dev/sd"* ]]; then
        echo "usb_sata"
    else
        echo "unknown"
    fi
}

is_l4t_installed_on_sdcard() { [[ "$(l4t_root_device)" == "sdcard" ]]; }

# the step functions live next to this script (bundled together into the wheel)
source "$(dirname "$(readlink -f "$0")")/setup-thor-steps.sh"

# ================================================================================
# MAIN
# ================================================================================

print_plan() {
    echo
    echo "Plan (dry-run — nothing will change):"

    local i=0
    _plan() {
        local name="$1" label="$2"
        i=$((i + 1))
        if is_skipped "$name"; then
            printf '  %2d  ·  %-8s  %s\n' "$i" "$name" "$label (skipped)"
        else
            printf '  %2d  →  %-8s  %s\n' "$i" "$name" "$label"
        fi
    }
    _plan gui      "$( [ "$KEEP_GUI" = "1" ] && echo 'keep GUI (boot target unchanged)' || echo 'disable GUI on boot (multi-user.target)')"
    _plan upgrade  "$( [ "$UPGRADE" = "1" ] && echo 'apt update + full-upgrade (+ nvidia-jetpack, curl, wget)' || echo 'skip the system upgrade')"
    _plan pip      "pip + setuptools + wheel (get-pip.py fallback)"
    _plan docker   "docker-ce + nvidia-container-toolkit, user in docker group, nvidia default runtime, service up + enabled"
    _plan jtop     "jetson-stats (jtop)"
    _plan nsight   "Nsight Compute (ncu — the verify stages profile with it)"
    _plan memory   "${SWAP_SIZE} GB swap file at /mnt (zRAM off)"
    _plan fan      "install recommended+max fan profiles (gated, .bck backup), FAN_DEFAULT_PROFILE: ${FAN_PROFILE}"
    _plan host     "disable nvargus-daemon / cups / ModemManager"
    _plan clocks   "jetson_clocks.service (lock max CPU/GPU/EMC clocks)"
    _plan power    "nvpmodel → MAXN/SUPER (reboot to apply)"
    echo
}

# Bail early if not running on a Jetson (no device-tree)
if [ ! -r /proc/device-tree/compatible ]; then
    echo "❌ /proc/device-tree/compatible not found; not a Jetson platform."
    exit 1
fi

# Prepare a "table" with spaces or tabs
cat <<EOF | column -t
  ✅ User    ${USER}
  ✅ SoM     ${SOC_TYPE}
  ✅ Memory  ${SOC_MEMORY}GB
  ✅ DevKit  ${IS_NVIDIA_CARRIER}
EOF

echo
if is_command_available nvidia-smi; then
    nvidia-smi
fi

if [ "$DRY_RUN" = "1" ]; then
    print_plan
    exit 0
fi

if [ "$ASSUME_YES" != "1" ]; then
    ask_should_proceed
fi

# 1. Boot target: headless server (multi-user), GUI off by default
if is_skipped gui; then
    echo "ℹ️  Skipping: gui"
elif [ "$KEEP_GUI" = "1" ]; then
    echo "ℹ️  Keeping the GUI boot target (--keep-gui)."
else
    configure_gui "no"
fi

# 2. System upgrade (apt), JetPack + curl + wget
if is_skipped upgrade; then
    echo "ℹ️  Skipping: upgrade"
else
    upgrade_system
fi

# 3. Python tooling
if is_skipped pip; then
    echo "ℹ️  Skipping: pip"
else
    install_pip
fi

# 4. Docker + NVIDIA runtime
if is_skipped docker; then
    echo "ℹ️  Skipping: docker"
else
    install_docker
    setup_docker_group
    configure_docker_runtime
    ensure_docker_running_and_enabled
fi

# 5. Monitor
if is_skipped jtop; then
    echo "ℹ️  Skipping: jtop"
else
    install_jtop
fi

# 5b. Nsight Compute (ncu)
if is_skipped nsight; then
    echo "ℹ️  Skipping: nsight"
else
    install_nsight_compute
fi

# 6. Memory: swap file + zRAM off
if is_skipped memory; then
    echo "ℹ️  Skipping: memory"
else
    configure_jetson_memory
fi

# 7. Fan profile
if is_skipped fan; then
    echo "ℹ️  Skipping: fan"
else
    do_fan
fi

# 8. Host service cleanup
if is_skipped host; then
    echo "ℹ️  Skipping: host"
else
    host_optimizations
fi

# 9. Max clocks (jetson_clocks)
if is_skipped clocks; then
    echo "ℹ️  Skipping: clocks"
else
    start_jetson_clocks_service
fi

# 10. Power mode (MAXN/SUPER)
if is_skipped power; then
    echo "ℹ️  Skipping: power"
else
    setup_power_mode
fi

echo "✅ Jetson Thor configuration completed."

if [ "$REBOOT_REQUIRED" = "1" ]; then
    if [ "$DO_REBOOT" = "1" ]; then
        echo "Rebooting, please wait..."
        sudo reboot
    elif [ -t 0 ]; then
        if ask_yes_no "A reboot is required to apply the power-mode change — reboot now?"; then
            echo "Rebooting, please wait..."
            sudo reboot
        else
            echo "ℹ️  Reboot whenever convenient (the power mode is already scheduled)."
        fi
    else
        echo "ℹ️  Reboot required to apply the power-mode change (rerun with --reboot, or reboot manually)."
    fi
fi
