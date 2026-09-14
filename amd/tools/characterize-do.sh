#!/usr/bin/env bash
# Characterize a GPU on DigitalOcean: create a droplet, read the device, write a
# profile, destroy it. The AMD counterpart of characterize-cloud.sh, and the
# only one of the three that reaches CDNA hardware.
#
#   amd/tools/characterize-do.sh gpu-mi300x1-192gb [region]
#   amd/tools/characterize-do.sh gpu-h200x1-141gb  [region]
#
# KNOWN LIMIT: MI300X is not reachable this way. The AMD Developer Cloud runs on
# DigitalOcean and its token is an ordinary dop_v1_ one, so it is tempting to
# assume the console and the API front the same inventory. They do not.
# gpu-mi325x1-256gb creates fine through the API -- that is where the verified
# mi325x profile came from -- while gpu-mi300x1-192gb returns
#
#     {"id":"unprocessable_entity","message":"Size is not available in this region."}
#
# in every region tried, at the same time as the AMD console offers MI300X for
# immediate creation. `amd/tools/do-sizes.sh mi3` explains why, and is the thing to
# run before ever attempting a create: the size reports available:true with an
# EMPTY region list, so the API knows the product and will never place it for
# this token. No region argument can fix that. The error message is misleading
# -- it says "not available in this region" for a size that is available in no
# region at all.
#
# So for MI300X: create the machine in the AMD console, then collect the dump
# by hand -- the SSH block below is the only part that needs the device, and
# rocminfo-to-profile.py runs at home:
#
#     export PATH=$PATH:/opt/rocm/bin
#     { rocminfo; rocm-smi --showallinfo; } > mi300x.rocminfo.txt 2>&1
#
# Do not iterate on the droplet. The AMD image ships hipcc without the HIP
# headers, which is what made this script collect raw text in the first place.
#
# Destruction is the whole discipline here. These droplets cost between $2.59
# and $11.19 an hour, and this runs against an account with a fixed credit
# balance rather than a card, so a droplet left up does not produce a surprising
# bill -- it produces no more runs at all. Three defences, in order of how much
# they are trusted:
#
#   1. destroy() is registered on EXIT/INT/TERM *before* anything is created,
#      and confirms the droplet is gone rather than assuming the request landed.
#   2. A watchdog process destroys the droplet after VGPU_DO_MAX_MINUTES no
#      matter what the main script is doing, which covers a hang.
#   3. The watchdog is disowned from the shell, so it survives the parent being
#      killed -- the one case a trap cannot cover.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
size="${1:?usage: characterize-do.sh <size-slug> [region]}"
region="${2:-atl1}"
outdir="${VGPU_CLOUD_OUT:-/tmp/vgpu-cloud}"
key_file="${VGPU_DO_KEY_FILE:-$HOME/.ssh/id_rsa}"
token_file="${VGPU_DO_TOKEN_FILE:-$HOME/.ssh/amd_developer_api}"
max_min="${VGPU_DO_MAX_MINUTES:-25}"
mkdir -p "$outdir"

[[ -r "$token_file" ]] || { echo "no DigitalOcean token at $token_file"; exit 1; }
[[ -r "$key_file" ]] || { echo "no ssh key at $key_file"; exit 1; }
TOK=$(tr -d ' \n\r' < "$token_file")
api() { curl -s -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" "$@"; }

# Check the inputs before renting anything.
for f in amd/tools/characterize-hip.cpp nvidia/tools/characterize.cu; do
  [[ -r "$here/$f" ]] || { echo "missing $here/$f (run from the repository)"; exit 1; }
done

DROPLET_ID=""
WATCHDOG=""
destroy() {
  if [[ -n "$WATCHDOG" ]]; then kill "$WATCHDOG" 2>/dev/null; fi
  if [[ -n "$DROPLET_ID" ]]; then
    echo "[$size] destroying droplet $DROPLET_ID"
    api -X DELETE "https://api.digitalocean.com/v2/droplets/$DROPLET_ID" >/dev/null
    # Confirm. A silent failure here costs the remaining credit.
    for _ in $(seq 1 20); do
      code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOK" \
             "https://api.digitalocean.com/v2/droplets/$DROPLET_ID")
      [[ "$code" == "404" ]] && { echo "[$size] destroyed (confirmed gone)"; return; }
      sleep 5
    done
    echo "[$size] WARNING: droplet $DROPLET_ID may still exist -- check the console"
  fi
}
trap destroy EXIT INT TERM

key_id=$(api "https://api.digitalocean.com/v2/account/keys" |
  python3 -c "import json,sys; print((json.load(sys.stdin).get('ssh_keys') or [{}])[0].get('id',''))")
[[ -n "$key_id" ]] || { echo "[$size] no ssh key registered on the account"; exit 1; }

echo "[$size] creating droplet in $region"
resp=$(api -X POST -d "{
  \"name\": \"vgpu-characterize-$$\",
  \"region\": \"$region\",
  \"size\": \"$size\",
  \"image\": \"gpu-amd-base\",
  \"ssh_keys\": [$key_id],
  \"tags\": [\"vgpu-characterize\"]
}" "https://api.digitalocean.com/v2/droplets")
DROPLET_ID=$(python3 -c "
import json,sys
d=json.loads('''$resp''')
print((d.get('droplet') or {}).get('id',''))
" 2>/dev/null)
if [[ -z "$DROPLET_ID" ]]; then
  printf '%s\n' "$resp" > "$outdir/${size}.create-error.json"
  echo "[$size] create failed: $(head -c 400 <<<"$resp")"
  exit 1
fi
echo "[$size] droplet $DROPLET_ID"

# Watchdog: destroys the droplet after the deadline whatever else happens, and
# survives this script being killed.
# Its output goes to a file, not to whatever this script inherited. A
# background subshell keeps the inherited stdout open for as long as it lives,
# so a caller that pipes this script (into tee, tail, anything) blocks for the
# full watchdog sleep after the work has finished -- which looks exactly like a
# hung run against a GPU that bills by the hour. It is not; it is this.
( sleep $((max_min * 60))
  curl -s -X DELETE -H "Authorization: Bearer $TOK" \
       "https://api.digitalocean.com/v2/droplets/$DROPLET_ID" >/dev/null
  echo "watchdog destroyed $DROPLET_ID after ${max_min}m"
) >"$outdir/${size}.watchdog.log" 2>&1 & WATCHDOG=$!
disown "$WATCHDOG" 2>/dev/null || true

IP=""
for _ in $(seq 1 60); do
  IP=$(api "https://api.digitalocean.com/v2/droplets/$DROPLET_ID" | python3 -c "
import json,sys
d=json.load(sys.stdin).get('droplet',{})
v4=[n for n in (d.get('networks',{}).get('v4') or []) if n.get('type')=='public']
print(v4[0]['ip_address'] if v4 else '')
" 2>/dev/null)
  [[ -n "$IP" ]] && break
  sleep 10
done
[[ -n "$IP" ]] || { echo "[$size] no public IP"; exit 1; }
echo "[$size] ip $IP; waiting for ssh"

SSH="ssh -i $key_file -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -o ConnectTimeout=10 -o LogLevel=ERROR root@$IP"
for _ in $(seq 1 60); do $SSH true 2>/dev/null && break; sleep 10; done
$SSH true 2>/dev/null || { echo "[$size] ssh never came up"; exit 1; }

echo "[$size] uploading and running"
scp -i "$key_file" -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR "$here/amd/tools/characterize-hip.cpp" root@"$IP":/tmp/ 2>/dev/null

# Dump the device, do not compile on it.
#
# The first two attempts tried to build a HIP program here and failed the same
# way twice: the AI/ML image ships hipcc but not the HIP development headers,
# so hip/hip_runtime.h does not exist anywhere under /opt/rocm. Each attempt
# was a fresh droplet on an account with fixed credit, which is an expensive
# way to learn a fact about an image.
#
# rocminfo and rocm-smi are already installed and report everything the profile
# schema needs: the gfx target, compute units, wavefront size, LDS, workgroup
# and grid limits, memory size and clocks. So this collects raw text and the
# parsing happens at home, where iterating costs nothing.
$SSH 'set -uo pipefail
  export PATH=$PATH:/opt/rocm/bin
  echo "===== rocminfo ====="
  rocminfo 2>/dev/null || echo "(rocminfo unavailable)"
  echo "===== rocm-smi showallinfo ====="
  rocm-smi --showallinfo 2>/dev/null || echo "(rocm-smi unavailable)"
  echo "===== rocm-smi product ====="
  rocm-smi --showproductname --showpower --showtemp --showmeminfo vram 2>/dev/null || true
' > "$outdir/${size}.rocminfo.txt" 2>&1
echo "[$size] collected $(wc -l < "$outdir/${size}.rocminfo.txt") lines of device data"

if grep -q "Compute Unit" "$outdir/${size}.rocminfo.txt" 2>/dev/null; then
  echo "[$size] device data -> $outdir/${size}.rocminfo.txt"
  echo "[$size] now run: amd/tools/rocminfo-to-profile.py $outdir/${size}.rocminfo.txt"
else
  echo "[$size] FAILED: rocminfo returned nothing usable; head of what came back:"
  head -20 "$outdir/${size}.rocminfo.txt" | sed 's/^/    /'
  exit 1
fi
