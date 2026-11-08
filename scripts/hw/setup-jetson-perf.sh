#!/usr/bin/env bash
# setup-jetson-perf.sh — the Jetson platform performance state, sourced by
# setup-jetson.sh (after setup-jetson-steps.sh): the nvfancontrol fan
# profiles, the verified jetson_clocks lock and the MAXN power mode. The
# OS/host provisioning lives in setup-jetson-steps.sh.

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
        print "setup-jetson: no THERMAL_GROUP anchor in " src " — refusing to guess" > "/dev/stderr"
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

jetson_clocks_lock_verified() {
    # "Locked" = jetson_clocks actually pinned the clocks: it locks a devfreq
    # device by writing max_freq into min_freq (/usr/bin/jetson_clocks), so
    # every GPU devfreq device must report min_freq == max_freq. Checking
    # `systemctl is-active` here was a false positive: the unit is a oneshot
    # with RemainAfterExit=true, which stays "active" forever after the first
    # run — even after nvpmodel has reset the pins.
    local d min max found=0
    for d in /sys/class/devfreq/*gpu*; do
        [ -e "$d/min_freq" ] || continue
        found=1
        min=$(cat "$d/min_freq")
        max=$(cat "$d/max_freq")
        if [ "$min" != "$max" ]; then
            echo "❌ $d is not locked (min_freq=$min != max_freq=$max)"
            return 1
        fi
    done
    if [ "$found" = "0" ]; then
        echo "❌ no GPU devfreq device found under /sys/class/devfreq/"
        return 1
    fi
    return 0
}

start_jetson_clocks_service() {
  local service_name="jetson_clocks.service"
  local service_path="/etc/systemd/system/${service_name}"
  local wrapper_path="/usr/local/bin/mjolnir-jetson-clocks-apply.sh"

  # 1) The apply wrapper: jetson_clocks + verify + retries. The boot-time
  #    run races nvpmodel.service (applying the power mode resets the
  #    devfreq pins), so the lock is verified and retried — and a unit
  #    that cannot verify FAILS loudly instead of staying green.
  sudo tee "$wrapper_path" >/dev/null <<'EOF'
#!/bin/bash
# Lock max clocks (jetson_clocks), then VERIFY min_freq == max_freq on every
# GPU devfreq device; retry — the boot run races nvpmodel.service. Non-zero
# exit on failure so systemd marks the unit failed instead of silently green.
verify() {
    local d min max found=0
    for d in /sys/class/devfreq/*gpu*; do
        [ -e "$d/min_freq" ] || continue
        found=1
        min=$(cat "$d/min_freq"); max=$(cat "$d/max_freq")
        [ "$min" = "$max" ] || return 1
    done
    [ "$found" = "1" ]
}
for attempt in 1 2 3 4 5; do
    /usr/bin/jetson_clocks || true
    if verify; then
        echo "jetson_clocks: lock verified (attempt $attempt)"
        exit 0
    fi
    echo "jetson_clocks: attempt $attempt did not verify; retrying in 5s" >&2
    sleep 5
done
echo "jetson_clocks: lock did NOT verify after 5 attempts" >&2
exit 1
EOF
    sudo chmod +x "$wrapper_path"

  # 2) Idempotent unit: nvpmodel-ordered (it resets the pins when it applies
  #    the mode), calling the verifying wrapper.
  local unit_wanted
  unit_wanted='[Unit]
Description=Jetson Clocks Service (lock max CPU/GPU/EMC clocks, verified)
After=multi-user.target nvpmodel.service
Wants=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/mjolnir-jetson-clocks-apply.sh
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
'
  if ! systemctl_service_exists "$service_name" \
     || ! diff -q <(sudo cat "$service_path") <(printf '%s' "$unit_wanted") &>/dev/null; then
    echo "ℹ️  Writing $service_path"
    printf '%s' "$unit_wanted" | sudo tee "$service_path" >/dev/null
    sudo systemctl daemon-reload
  fi

  # 3) Enabled unconditionally (the old code only enabled on first create —
  #    a pre-existing but disabled unit stayed disabled).
  if ! systemctl_service_is_enabled "$service_name"; then
    echo "ℹ️  Enabling $service_name (start on boot)"
    systemctl_enable_service "$service_name"
  fi

  # 4) Apply now — restart re-runs the oneshot even if already active.
  echo "ℹ️  Applying the clock lock (jetson_clocks + verify)"
  systemctl_restart_service "$service_name"

  # 5) The honest health check: the actual pins, not the unit state.
  if jetson_clocks_lock_verified; then
    echo "✅ Jetson Clocks locked (GPU devfreq min_freq == max_freq)."
    sudo jetson_clocks --show 2>/dev/null | head -4 || true
  else
    echo "❌ The clock lock did not verify — see $service_name's journal"
    echo "   (journalctl -u $service_name). Check nvpmodel mode: nvpmodel -q"
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

    # JetPack 7's nvpmodel asks an interactive "reboot now?" on transitions that
    # require one; without a tty (ssh/cron/CI) it dies there — exit 234, mode
    # untouched. --force applies cleanly, rebooting itself only when the
    # transition demands it; pre-7 versions have no --force -> plain path.
    if ! sudo nvpmodel -m "$MAXN_POWER_MODE_ID" -f "$power_config_path" --force > /dev/null 2>&1; then
        sudo nvpmodel -m "$MAXN_POWER_MODE_ID" -f "$power_config_path" > /dev/null || { echo "❌ Failed to set power mode"; return 1; }
    fi
    echo "✅ Power mode change scheduled. A reboot will be required to apply this change."
    REBOOT_REQUIRED=1
    return 0
}
