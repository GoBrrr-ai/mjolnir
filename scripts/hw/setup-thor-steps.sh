#!/usr/bin/env bash
# setup-thor-steps.sh — the idempotent setup steps, sourced by setup-thor.sh
# ( UTILITIES live there; these functions only read the option globals at call time ).

# ================================================================================
# SYSTEM / HOST
# ================================================================================

upgrade_system() {
    if is_true "$UPGRADE"; then
        echo "📥 Updating package lists..."
        sudo apt update &> /dev/null

        echo "📥 Upgrading system..."
        sudo apt full-upgrade -y &> /dev/null
        echo "✅ System upgraded."

        ensure_installed nvidia-jetpack &> /dev/null
    else
        echo "ℹ️  Skipping system upgrade (--no-upgrade)."
    fi

    ensure_installed curl curl
    ensure_installed wget wget
}

# ================================================================================
# INSTALLERS
# ================================================================================

install_pip() {
    # If pip already on PATH, we're done
    if is_command_available pip; then
        echo "✅ pip is already installed."
        return
    fi

    echo "ℹ️  Installing pip and supporting tools..."
    if ! sudo apt install -y python3-pip python3-setuptools python3-pkg-resources; then
        echo "⚠️ Attempting to fix broken packages..."
        sudo apt --fix-broken install -y
        if ! sudo apt install -y python3-pip python3-setuptools python3-pkg-resources; then
            echo "🚨 Falling back to get-pip.py bootstrap..."

            ensure_installed curl curl

            curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
            sudo python3 /tmp/get-pip.py
            rm -f /tmp/get-pip.py
        fi
    fi

    echo "ℹ️  Checking for pip3 in standard paths..."
    if ! is_command_available pip3; then
        echo "⚠️ pip3 not found in PATH — checking ~/.local/bin..."
        export PATH="$HOME/.local/bin:$PATH"
        if ! is_command_available pip3; then
            echo "❌ pip3 still not found. Check your installation manually."
            return 1
        else
            echo "✅ Found pip3 in ~/.local/bin — added to PATH for this session."
        fi
    fi

    echo "ℹ️  Verifying pip3 version..."
    pip3 --version

    echo "ℹ️  Upgrading pip and tools..."
    python3 -m pip install --upgrade --user pip setuptools wheel

    echo "ℹ️  Ensuring ~/.local/bin is in PATH..."
    if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo '✅ Added ~/.local/bin to ~/.bashrc for future sessions.'
    fi

    echo "✅ pip3 installation complete and usable in this session."
}

install_jtop() {
    # https://github.com/rbonghi/jetson_stats
    if ! command -v jtop &> /dev/null; then
        echo "📦 Installing jtop..."
        pip3 install -U "setuptools<71.0.0"
        sudo env "PATH=$HOME/.local/bin:$PATH" pip3 install -U jetson-stats
        systemctl_restart_service jtop.service
        echo "✅ jtop was installed successfully."
    else
        echo "✅ jtop is already installed."
    fi
}

# ================================================================================
# DOCKER
# ================================================================================

ensure_docker_repo() {
    local docker_keyring_path="/etc/apt/keyrings/docker.asc"

    if ! file_exists "${docker_keyring_path}"; then
        echo "🔑 Adding Docker GPG key and repository..."

        sudo apt update
        sudo apt install ca-certificates curl
        sudo install -m 0755 -d /etc/apt/keyrings
        sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        sudo chmod a+r /etc/apt/keyrings/docker.asc

        # Add the repository to Apt sources:
        echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt update
    fi
}

install_docker() {
    ensure_docker_repo

    if ! is_command_available docker; then
        echo "📦 Installing docker..."

        sudo apt update
        sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin nvidia-container-toolkit

        echo "✅ docker was installed successfully."
    else
        echo "✅ docker is already installed."
    fi

    if ! docker compose version &> /dev/null; then
        echo "📦 Installing docker-compose..."
        sudo apt install -y docker-compose-plugin
        echo "✅ docker-compose was installed successfully."
    else
        echo "✅ docker-compose is already installed."
    fi
}

# Add user to docker group
setup_docker_group() {
    # Check if user is in docker group
    if is_user_in_group docker; then
        echo "✅ User '$USER' is already in the docker group."
    else
        echo "Adding $USER to the docker group..."
        sudo usermod -aG docker "$USER"

        # Group membership takes effect at next login; the live socket may
        # stay unwritable until then.
        if [[ -S /var/run/docker.sock && ! -w /var/run/docker.sock ]]; then
            echo "ℹ️  The docker socket stays unwritable until you log in again."
        fi
    fi
}

# Function to check Docker runtime configuration
check_docker_runtime() {
    local daemon_config="/etc/docker/daemon.json"

    if ! is_command_available docker; then
        return 1
    fi

    if ! file_exists $daemon_config; then
        return 1
    fi

    if grep -q '"default-runtime": "nvidia"' $daemon_config 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Configure Docker with NVIDIA runtime as default
configure_docker_runtime() {
    local daemon_config="/etc/docker/daemon.json"

    # THOR:
    # -----
    # sudo nvidia-ctk runtime configure --runtime=docker
    # to configure it to following state:
    # {
    #     "runtimes": {
    #         "nvidia": {
    #             "args": [],
    #             "path": "nvidia-container-runtime"
    #         }
    #     }
    # }
    # and always use --runtime=nvidia when running docker docntainers with GPU
    # -----

    if check_docker_runtime; then
        echo "✅ NVIDIA runtime is already set as default in Docker: ${daemon_config}"
    else
        echo "❌ NVIDIA runtime is NOT set as default - configuring now..."

        # Check if jq is installed
        ensure_installed jq jq

        # Configure NVIDIA runtime as default
        local tmp
        tmp=$(mktemp)
        if file_exists $daemon_config; then
            sudo jq '.
                | .runtimes += {
                    "nvidia": {
                        "path": "nvidia-container-runtime",
                        "runtimeArgs": []
                    }
                    }
                | .["default-runtime"] = "nvidia"
                ' "$daemon_config" > "$tmp"
            sudo mv "$tmp" "$daemon_config"
        else
            cat > "$tmp" <<'EOF'
{
"runtimes": {
    "nvidia": {
    "path": "nvidia-container-runtime",
    "runtimeArgs": []
    }
},
"default-runtime": "nvidia"
}
EOF
            sudo mv "$tmp" "$daemon_config"
        fi

        # The tmp file is user-owned; daemon.json must be root:root 644.
        sudo chown root:root "$daemon_config"
        sudo chmod 644 "$daemon_config"

        # Restart Docker to apply changes
        echo "ℹ️  Restarting Docker service to apply changes..."
        systemctl_restart_service docker
        echo "✅ NVIDIA runtime is now set as default."
    fi
}

ensure_docker_running_and_enabled() {
    # 1. Does the unit exist?
    if ! systemctl_service_exists docker; then
        echo "❌ Docker service is not installed. Please install Docker first."
        return 1
    fi

    # 2. Start it if it's not running
    if ! systemctl_service_is_active docker; then
        echo "▶️ Starting Docker service..."
        systemctl_start_service docker

        if systemctl_service_is_active docker; then
            echo "✅ Docker service started."
        else
            echo "❌ Failed to start Docker service."
            return 1
        fi
    else
        echo "✅ Docker service is already running."
    fi

    # 3. Enable it at boot if it's not enabled
    if ! systemctl_service_is_enabled docker; then
        echo "⚙️ Enabling Docker service at boot..."
        systemctl_enable_service docker

        if systemctl_service_is_enabled docker; then
            echo "✅ Docker service is now enabled at boot."
        else
            echo "❌ Failed to enable Docker service at boot."
            return 1
        fi
    else
        echo "✅ Docker service is already enabled at boot."
    fi
}

# ================================================================================
# SWAP
# ================================================================================

check_swap_exists() {
    local swap_file="${1:-}"

    # non-empty and regular file
    file_exists "${swap_file}" || return 1

    # check if it's active
    if swapon --noheadings --raw --show=NAME | grep -Fqx "${swap_file}"; then
        return 0
    else
        return 1
    fi
}

# Check if swap file exists and is active
check_swap_file_status() {
    local swap_file="${1:-}"

    if file_exists "$swap_file"; then
        if check_swap_exists "$swap_file"; then
            echo "active"
        else
            echo "inactive"
        fi
    else
        echo "missing"
    fi
}

# Helper function for robust swap file cleanup
cleanup_swap_file() {
    local swap_file="$1"

    [[ -z "${swap_file}" ]] && return 0

    sync 2>/dev/null || true
    sudo swapoff "${swap_file}" 2>/dev/null || true
    sudo rm -f "${swap_file}" 2>/dev/null || true
}

# Create and enable swap file with safety checks
setup_swap_file() {
    local swap_size=$SWAP_SIZE
    local swap_file_path="/mnt/${swap_size}GB.swap"
    local swap_state
    local size_bytes
    swap_state=$(check_swap_file_status "${swap_file_path}")
    size_bytes=$(convert_to_bytes "${swap_size}G")

    # Validate inputs
    if [[ -z "${swap_file_path}" ]]; then
        echo "❌ Error: swap_file_path is not set"
        return 1
    fi

    if [[ -z "${size_bytes}" ]] || [[ "${size_bytes}" -le 0 ]]; then
        echo "❌ Error: Invalid swap size: ${swap_size}"
        return 1
    fi

    # Validate swap file path (should be absolute and in safe location)
    if [[ "${swap_file_path}" != /* ]]; then
        echo "❌ Error: swap_file_path must be absolute: ${swap_file_path}"
        return 1
    fi

    # Ensure path is in a safe location (not root, boot, etc.)
    case "${swap_file_path}" in
        /boot/*|/proc/*|/sys/*|/dev/*|/run/*)
            echo "❌ Error: Unsafe swap file location: ${swap_file_path}"
            return 1
            ;;
    esac

    # Get directory and check if it exists
    local swap_dir=$(dirname "${swap_file_path}")
    if [[ ! -d "${swap_dir}" ]]; then
        echo "❌ Error: Directory does not exist: ${swap_dir}"
        return 1
    fi

    # Check write permissions on directory
    if [[ ! -w "${swap_dir}" ]] && ! sudo test -w "${swap_dir}"; then
        echo "❌ Error: No write permission for directory: ${swap_dir}"
        return 1
    fi

    # Check available space (add 10% buffer)
    local available_space
    available_space=$(df --output=avail -B1 "${swap_dir}" | tail -n1)
    if [[ "${size_bytes}" -gt "${available_space}" ]]; then
        echo "❌ Error: Insufficient disk space"
        echo "   Requested: $(numfmt --to=iec "${size_bytes}" 2>/dev/null || echo "${size_bytes} bytes")"
        echo "   Available: $(numfmt --to=iec "${available_space}" 2>/dev/null || echo "${available_space} bytes")"
        return 1
    fi

    # If swap file exists, disable and remove it first
    if [ "${swap_state}" != "missing" ]; then
        cleanup_swap_file "${swap_file_path}"
        # Remove from fstab if present
        sudo sed -i "\\#^${swap_file_path}#d" /etc/fstab
    fi

    # Create swap only when not installed on sdcard
    if ! is_l4t_installed_on_sdcard; then
        # Calculate count more safely
        local count_mb=$((size_bytes/1024/1024))
        if [[ $((count_mb * 1024 * 1024)) -ne "${size_bytes}" ]]; then
            echo "❌ Error: Size calculation overflow or precision loss"
            return 1
        fi

        # Use fallocate if available (faster and safer than dd)
        if command -v fallocate >/dev/null 2>&1; then
            if ! sudo fallocate -l "${size_bytes}" "${swap_file_path}"; then
                echo "❌ Error: Failed to create swap file with fallocate"
                return 1
            fi
        else
            # Fallback to dd with additional safety
            if ! sudo dd if=/dev/zero of="${swap_file_path}" bs=1M count="${count_mb}" status=progress conv=fsync 2>/dev/null; then
                echo "❌ Error: Failed to create swap file with dd"
                # Clean up partial file
                cleanup_swap_file "${swap_file_path}"
                return 1
            fi
        fi

        # Set proper permissions
        if ! sudo chmod 600 "${swap_file_path}"; then
            echo "❌ Error: Failed to set swap file permissions"
            cleanup_swap_file "${swap_file_path}"
            return 1
        fi

        # Initialize swap
        if ! sudo mkswap "${swap_file_path}" >/dev/null; then
            echo "❌ Error: Failed to initialize swap file"
            cleanup_swap_file "${swap_file_path}"
            return 1
        fi

        # Enable swap
        if ! sudo swapon "${swap_file_path}"; then
            echo "❌ Error: Failed to enable swap file"
            cleanup_swap_file "${swap_file_path}"
            return 1
        fi

        # Add to fstab if not already present (using awk for precise matching)
        if ! awk -v path="${swap_file_path}" '$1 == path && $3 == "swap" {found=1} END {exit !found}' /etc/fstab 2>/dev/null; then
            if ! echo "${swap_file_path} none swap sw,pri=1 0 0" | sudo tee -a /etc/fstab >/dev/null; then
                echo "⚠️  Warning: Failed to add swap to fstab (swap is still active)"
            fi
        fi

        # Final verification
        if check_swap_exists "${swap_file_path}"; then
            echo "✅ Swap file setup complete ($(numfmt --to=iec ${size_bytes}))"
            return 0
        else
            echo "❌ Failed to verify swap file setup"
            return 1
        fi
    else
        echo "✅ Skipping swap file creation (L4T installed on SD card)"
        return 0
    fi
}

# ================================================================================
# zRAM
# ================================================================================

check_zram_state() {
    if swapon --noheadings --raw --show=NAME | grep -q 'zram'; then
        echo "active"
    elif file_exists /etc/modules-load.d/zram.conf; then
        echo "configured"
    else
        echo "disabled"
    fi
}

# Disable zRAM (the Thor setup uses a file-backed swap instead)
setup_zram() {
    local zram_enabled=${1:-"no"}

    # If zRAM should be enabled
    if is_true "${zram_enabled}"; then
        echo "❌ zRAM is not enabled by the mjolnir Thor setup (file-backed swap instead)."
        return 1
    fi

    local zram_state
    zram_state=$(check_zram_state)
    # Disable zRAM if it's enabled
    if [ "${zram_state}" != "disabled" ]; then
        # Stop and disable the zram service
        systemctl_disable_service zram
        systemctl_disable_service nvzramconfig

        local d
        for d in /dev/zram*; do
            [[ -e "$d" ]] || continue
            sudo swapoff "$d" 2>/dev/null || true
            echo 1 | sudo tee /sys/block/"$(basename "$d")"/reset >/dev/null || true
            echo "✅ Disabled $d..."
        done

        # Unload the zram module
        sudo rmmod zram 2>/dev/null || true

        sudo rm -f /etc/modules-load.d/zram.conf /etc/modprobe.d/zram.conf \
                  /etc/udev/rules.d/99-zram.rules /etc/systemd/system/zram.service
        sudo systemctl daemon-reload

        echo "✅ zRAM has been disabled"
    else
        echo "✅ zRAM is already disabled"
    fi
}

# Configure memory settings
configure_jetson_memory() {
    # Setup swap file
    setup_swap_file

    # Configure zRAM (the Thor setup keeps it off)
    setup_zram "false"

    # Display current memory configuration
    echo -e "\n=== Current Memory Configuration ==="
    free -h

    # Display swap configuration if supported
    if ! is_l4t_installed_on_sdcard; then
        echo -e "\nSwap configuration:"
        swapon -s
    fi

    echo
    echo "✅ Memory setup complete"
}

# ================================================================================
# GUI
# ================================================================================

# Function to get current GUI state
get_gui_state() {
    if systemctl get-default | grep -q "graphical.target"; then
        echo "enabled"
    else
        echo "disabled"
    fi
}

# Function to enable GUI
enable_gui() {
    sudo systemctl set-default graphical.target

    if is_true "$(get_gui_state)"; then
        echo "✅ GUI has been enabled on boot"
        return 0
    else
        echo "❌ Failed to enable GUI"
        return 1
    fi
}

# Function to disable GUI
disable_gui() {
    sudo systemctl set-default multi-user.target

    if ! is_true "$(get_gui_state)"; then
        echo "✅ GUI has been disabled on boot"
        return 0
    else
        echo "❌ Failed to disable GUI"
        return 1
    fi
}

configure_gui() {
    local gui_enabled=${1-"no"}
    local current_state
    current_state=$(get_gui_state)

    if is_true "$gui_enabled"; then
        if [ "$current_state" = "disabled" ]; then
            enable_gui
        else
            echo "✅ GUI is already enabled"
        fi
    else
        if [ "$current_state" = "enabled" ]; then
            disable_gui
        else
            echo "✅ GUI is already disabled"
        fi
    fi
}

# ================================================================================
# FAN (merged from scripts/hw/fan-profiles.sh — one process with the rest)
#
# 1. BACKUP (first run only): untouched $CONF is dumped to $CONF.bck.
#    The .bck ALWAYS holds the original — the clean rollback point.
# 2. INJECT (per-profile gated): the two performance fan profiles
#    (tuned on the AGX Thor) are inserted before the THERMAL_GROUP
#    section — each ONLY if its block is not already present:
#      FAN_PROFILE recommended  — balanced warmth/acoustics
#      FAN_PROFILE max          — sustained full-load cooling (bench)
# 3. FAN MODE SELECTION: FAN_DEFAULT_PROFILE <--fan-profile> (default
#    "recommended", not the stock "cool").
# 4. Restart nvfancontrol and print the daemon's active state.
# ================================================================================

FAN_CONF="${MJOLNIR_NVFANCONF:-/etc/nvfancontrol.conf}"

write_fan_profile_blocks() {
  local dir="$1"
  cat > "$dir/profile_recommended.conf" <<'EOF'
	FAN_PROFILE recommended {
		#TEMP	  HYST	PWM	RPM
		0	   0	255	5371
		10	   0	220	4700
		20	   0	180	3900
		30	   0	140	3000
		40	   0	102	2400
		55	   0	90	2100
		70	   0	80	1800
		115	   0	80	1800
	}
EOF

  cat > "$dir/profile_max.conf" <<'EOF'
	FAN_PROFILE max {
		#TEMP	  HYST	PWM	RPM
		0	   0	255	5371
		15	   0	240	5000
		25	   0	220	4700
		35	   0	195	4200
		50	   0	170	3700
		70	   0	120	2700
		115	   0	120	2700
	}
EOF
}

# $1=src $2=dst — inserts the missing profile block(s) before the
# THERMAL_GROUP anchor and selects the fan mode (FAN_DEFAULT_PROFILE).
fan_transform() {
  awk -v do_rec="$DO_REC" -v do_max="$DO_MAX" -v sel="$FAN_PROFILE" \
      -v src="$1" \
      -v f_rec="$FAN_TMP/profile_recommended.conf" \
      -v f_max="$FAN_TMP/profile_max.conf" '
    /^[[:space:]]*FAN_DEFAULT_PROFILE/ { sub(/FAN_DEFAULT_PROFILE.*/, "FAN_DEFAULT_PROFILE " sel) }
    !done && $0 ~ /^[[:space:]]*THERMAL_GROUP[[:space:]]0/ {
      done=1
      if (do_rec) while ((getline l < f_rec) > 0) print l
      if (do_max) while ((getline l < f_max) > 0) print l
    }
    { print }
    END {
      if (!done) {
        print "setup-thor: no THERMAL_GROUP anchor in " src " — refusing to guess" > "/dev/stderr"
        exit 3
      }
    }
  ' "$1" > "$2"
}

do_fan() {
  local conf="$FAN_CONF"
  local bck="$conf.bck"

  if ! file_exists "$conf"; then
    echo "ℹ️  No nvfancontrol conf at $conf — skipping fan."
    return 0
  fi

  if ! is_command_available nvfancontrol; then
    echo "ℹ️  nvfancontrol not available — skipping fan."
    return 0
  fi

  local have_rec have_max cur_default
  have_rec=$(grep -cE "FAN_PROFILE recommended[[:space:]]*\{" "$conf" || true)
  have_max=$(grep -cE "FAN_PROFILE max[[:space:]]*\{" "$conf" || true)
  cur_default="$(awk '/^[[:space:]]*FAN_DEFAULT_PROFILE/ {print $2; exit}' "$conf")"
  cur_default="${cur_default:-<unset>}"

  local do_rec=0 do_max=0
  [ "$have_rec" -eq 0 ] && do_rec=1
  [ "$have_max" -eq 0 ] && do_max=1
  DO_REC=$do_rec
  DO_MAX=$do_max

  # Stock profiles ship with the L4T conf; recommended/max are installed here.
  case "$FAN_PROFILE" in
    cool|quiet)
      if ! grep -qE "FAN_PROFILE ${FAN_PROFILE}[[:space:]]*\{" "$conf"; then
        echo "❌ fan profile '$FAN_PROFILE' not in $conf" >&2
        return 1
      fi
      ;;
  esac

  local pending=0
  [ -e "$bck" ] || pending=1
  [ "$do_rec" = 1 ] && pending=1
  [ "$do_max" = 1 ] && pending=1
  [ "$cur_default" != "$FAN_PROFILE" ] && pending=1

  echo "fan profiles: $conf"
  if [ -e "$bck" ]; then
    echo "  backup:      $bck (exists — original preserved)"
  else
    echo "  backup:      $conf -> $bck (first run — will be created)"
  fi
  if [ "$do_rec" = 1 ]; then
    echo "  inject:      FAN_PROFILE recommended (not present — will be added)"
  else
    echo "  inject:      FAN_PROFILE recommended (already present — skipped)"
  fi
  if [ "$do_max" = 1 ]; then
    echo "  inject:      FAN_PROFILE max (not present — will be added)"
  else
    echo "  inject:      FAN_PROFILE max (already present — skipped)"
  fi
  if [ "$cur_default" != "$FAN_PROFILE" ]; then
    echo "  fan mode:    FAN_DEFAULT_PROFILE $cur_default -> $FAN_PROFILE"
  else
    echo "  fan mode:    FAN_DEFAULT_PROFILE $FAN_PROFILE (already selected)"
  fi

  if [ "$pending" = 0 ]; then
    echo "✅ Fan already configured (profile: $FAN_PROFILE)."
    return 0
  fi

  FAN_TMP="$(mktemp -d)"
  trap 'rm -rf "$FAN_TMP"' EXIT
  write_fan_profile_blocks "$FAN_TMP"

  # First run only: the .bck always holds the untouched original.
  [ -e "$bck" ] || sudo cp -a -- "$conf" "$bck"
  fan_transform "$conf" "$FAN_TMP/new.conf"
  sudo cp -- "$FAN_TMP/new.conf" "$conf"

  # Remove the old status so the daemon reloads fresh, then restart.
  sudo rm -f /var/lib/nvfancontrol/status
  systemctl_restart_service nvfancontrol

  echo "✅ Fan profile installed; default: $FAN_PROFILE."
  echo "  active state:"
  sudo nvfancontrol -q 2>&1 | grep -iE 'profile|governor' | sed 's/^/    /' || true
}

# ================================================================================
# HOST / SYSTEM CONFIGURATION
# ================================================================================

host_optimizations() {
    systemctl_disable_service nvargus-daemon.service
    systemctl_disable_service cups
    systemctl_disable_service ModemManager
}

check_jetson_clocks() {
    local state
    state=$(sudo jetson_clocks --show | grep -oP '(?<=FreqOverride=)\d+')

    is_true "$state"
}

start_jetson_clocks_service() {
  local service_name="jetson_clocks.service"
  local service_path="/etc/systemd/system/${service_name}"

  # 1) Ensure the unit file exists
  if ! systemctl_service_exists "$service_name"; then
    echo "ℹ️  $service_name not found; creating at $service_path"
    sudo tee "$service_path" >/dev/null <<'EOF'
[Unit]
Description=Jetson Clocks Service
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

    echo "ℹ️  Reloading systemd daemon"
    sudo systemctl daemon-reload

    echo "ℹ️  Enabling $service_name to start on boot"
    systemctl_enable_service "$service_name"
  fi

  # 2) Check if the service is active
  if ! systemctl_service_is_active "$service_name"; then
    echo "ℹ️  Starting $service_name"
    systemctl_start_service "$service_name"
    echo "✅ $service_name started."
    echo "ℹ️  Current clock settings:"
    sudo jetson_clocks --show
  fi

  # 3) Final health check
  if check_jetson_clocks; then
    echo "✅ Jetson Clocks is enabled and running."
  else
    echo "❌ Failed to enable Jetson Clocks lock."
    return 1
  fi
}

# ================================================================================
# POWER MODE
# ================================================================================

check_nvpmodel_installed() {
    if ! is_command_available nvpmodel; then
        echo "❌ Error: Missing required dependency: nvpmodel"
        echo "   Please check your Jetson system installation"
        exit 1
    fi
}

get_power_mode() {
    check_nvpmodel_installed

    nvpmodel -q | grep "NV Power Mode" | cut -d':' -f2 | xargs
}

get_power_modes() {
    local file="${1:-/etc/nvpmodel.conf}"
    # declare global associative array
    declare -gA JETSON_POWER_MODES=()

    local line id name
    while IFS= read -r line; do
        if [[ $line =~ \<[[:space:]]*POWER_MODEL[[:space:]]+ID=([0-9]+)[[:space:]]+NAME=([^[:space:]>]+) ]]; then
            id="${BASH_REMATCH[1]}"
            name="${BASH_REMATCH[2]}"
            JETSON_POWER_MODES[$id]="$name"
        fi
    done < "$file"
}

get_power_mode_index_by_name() {
  local substr="$1"
  local id

  for id in "${!JETSON_POWER_MODES[@]}"; do
    if [[ "${JETSON_POWER_MODES[$id]}" == *"$substr"* ]]; then
      printf '%s\n' "$id"
      return 0
    fi
  done
  return 1
}

get_power_mode_config_file() {
    check_nvpmodel_installed

    nvpmodel -q --verbose 2>&1 | grep -m1 'Config file:' | awk -F': ' '{print $NF}'
}

# Configure power mode
setup_power_mode() {
    local current_mode
    local power_config_path
    current_mode="$(get_power_mode)"
    power_config_path="$(get_power_mode_config_file)"

    get_power_modes

    # First try "SUPER", then "MAXN"
    if MAXN_POWER_MODE_ID=$(get_power_mode_index_by_name SUPER); then
        :
    elif MAXN_POWER_MODE_ID=$(get_power_mode_index_by_name MAXN); then
        :
    else
        echo "❌ No matching MAXN or MAXN_SUPER power mode!"
        exit 1
    fi

    if [ "$current_mode" = "${JETSON_POWER_MODES[$MAXN_POWER_MODE_ID]}" ]; then
        echo "✅ Power mode already set to $current_mode, skipping..."
        return 0
    fi

    echo "ℹ️  Setting power mode to mode ${JETSON_POWER_MODES[$MAXN_POWER_MODE_ID]} (this will be applied after reboot)..."

    # Use -f flag to suppress the interactive reboot prompt
    if sudo nvpmodel -m "$MAXN_POWER_MODE_ID" -f "$power_config_path" > /dev/null; then
        echo "✅ Power mode change scheduled. A reboot will be required to apply this change."
        REBOOT_REQUIRED=1
        return 0
    else
        echo "❌ Failed to set power mode"
        return 1
    fi
}
