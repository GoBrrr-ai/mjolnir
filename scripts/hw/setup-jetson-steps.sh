#!/usr/bin/env bash
# setup-jetson-steps.sh — the idempotent OS/host setup steps, sourced by
# setup-jetson.sh (with setup-jetson-perf.sh: the fan/clock/power state).
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
        # PEP 668 (ubuntu 24.04 / JP7): a plain pip install exits non-zero there —
        # and under set -e that kills the whole run; no-op where no marker exists.
        local pip_flags=""
        [ -f "$(python3 -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')/EXTERNALLY-MANAGED" ] && pip_flags="--break-system-packages"
        pip3 install $pip_flags -U "setuptools<71.0.0"
        sudo env "PATH=$HOME/.local/bin:$PATH" pip3 install $pip_flags -U jetson-stats
        # jetson-stats >= 7 no longer registers its unit during pip install, so the
        # restart exits 5 ("unit does not exist") — install it on demand instead.
        systemctl_restart_service jtop.service || sudo jtop --install-service || true
        echo "✅ jtop was installed successfully."
    else
        echo "✅ jtop is already installed."
    fi
}

# ================================================================================
# NSIGHT COMPUTE (ncu — the verify stages profile with it)
# ================================================================================

install_nsight_compute() {
    # Nsight Compute ships in the JetPack repo, named for the CUDA minor it
    # targets (cuda-nsight-compute-13-2 on JP 7.2). The meta follows the
    # box's CUDA install; the ncu binary lands in
    # /opt/nvidia/nsight-compute/<version>/. Idempotent: skip when present.
    if ls /opt/nvidia/nsight-compute/*/ncu &> /dev/null; then
        echo "✅ Nsight Compute is already installed."
        return
    fi
    local cuda_dot
    cuda_dot=$(ls /usr/local/ 2>/dev/null | grep -oP '^cuda-\d+\.\d+$' | sort -V | tail -1 | cut -d- -f2 || true)
    local meta=""
    if [ -n "$cuda_dot" ]; then
        meta="cuda-nsight-compute-${cuda_dot/./-}"
        apt-cache show "$meta" &> /dev/null || meta=""
    fi
    if [ -z "$meta" ]; then
        # No CUDA dir or no matching meta: take the newest plain package.
        meta=$(apt-cache search --names-only '^nsight-compute-[0-9]' | awk '{print $1}' | sort -V | tail -1)
    fi
    if [ -z "$meta" ]; then
        echo "❌ no nsight-compute package found (is the JetPack apt repo configured?)"
        return 1
    fi
    echo "📦 Installing Nsight Compute ($meta)..."
    sudo apt update
    sudo apt install -y "$meta"
    local ncu_bin
    ncu_bin=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)
    if [ -n "$ncu_bin" ]; then
        echo "✅ Nsight Compute installed: $ncu_bin"
    else
        echo "❌ installed but no ncu binary found under /opt/nvidia/nsight-compute/"
        return 1
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
