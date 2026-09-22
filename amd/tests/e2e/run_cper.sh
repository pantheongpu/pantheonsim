#!/usr/bin/env bash
# CPER records for an AMD GPU's ECC errors, as amdgpu writes them and amd-smi
# dumps them: `amd-smi ras --cper --folder` writes one .cper per record with
# its header as .json, named and numbered as amd-smi names them, and each
# record has amd_cper.h's layout -- read here byte by byte, independently of
# the code that wrote it. --severity picks records, --file-limit keeps the
# newest. (The ACA registers and the AFIDs printed beside each file were
# checked against AMD's own decoder, amdsmi's ras-decode: 24 corrected, 22
# uncorrected.)
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
command -v python3 >/dev/null || { echo "SKIP: python3 reads the records"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state" VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
v() { "$vgpu" "$@" 2>&1; }

v fault inject --gpu 0 --ecc corrected >/dev/null
v fault inject --gpu 1 --ecc uncorrected >/dev/null
v fault inject --gpu 0 --ecc corrected >/dev/null

out=$(v smi --amd ras --cper --folder "$tmp/cper")
expect "the table names each file and its AFIDs" \
  "timestamp gpu_id severity file_name list of afids|0 NONFATAL-CORRECTED corrected-1.cper 24|0 NONFATAL-CORRECTED corrected-2.cper 24|1 NONFATAL-UNCORRECTED uncorrected-3.cper 22" \
  "$(awk 'NR == 1 {$1=$1; print; next} {print $3, $4, $5, $6}' <<< "$out" | paste -sd'|')"
expect "a .cper and a .json per record" \
  "corrected-1.cper corrected-1.json corrected-2.cper corrected-2.json uncorrected-3.cper uncorrected-3.json" \
  "$(ls "$tmp/cper" | paste -sd' ')"

# amd_cper.h, packed: the 128-byte header, one 72-byte section descriptor,
# one 272-byte AMD non-standard error section.
read_record() {
  python3 - "$1" <<'EOF'
import struct, sys, uuid
b = open(sys.argv[1], "rb").read()
sig, rev, sig_end, sec_cnt, sev, valid, length = struct.unpack_from("<4sHIHIII", b, 0)
sec, mnt, hr, _, day, mon, yr, cent = struct.unpack_from("<8B", b, 24)
platform = b[32:48].split(b"\0")[0].decode()
creator = b[64:80].split(b"\0")[0].decode()
notify = str(uuid.UUID(bytes_le=b[80:96])).upper()
record_id = b[96:104].split(b"\0")[0].decode()
off, slen, rminor, rmajor, svalid, _, flags = struct.unpack_from("<IIBBBBI", b, 128)
sec_type = str(uuid.UUID(bytes_le=b[144:160])).upper()
ssev, = struct.unpack_from("<I", b, 176)
fru = b[180:200].split(b"\0")[0].decode()
cnts, = struct.unpack_from("<Q", b, off)
ctx_type, arr = struct.unpack_from("<HH", b, off + 128)
regs = struct.unpack_from("<16Q", b, off + 144)
print(len(b), sig.decode(), hex(rev), hex(sig_end), sec_cnt, sev, hex(valid), length, platform, creator, notify,
      record_id, off, slen, "%x.%x" % (rmajor, rminor), hex(svalid), hex(flags), sec_type, ssev, fru,
      (cnts >> 2) & 63, (cnts >> 8) & 63, ctx_type, arr, "%016x" % regs[1], "%x" % regs[5],
      "%d/%02d/%02d" % (cent * 100 + yr, mon, day))
EOF
}
IFS=' ' read -r -a c <<< "$(read_record "$tmp/cper/corrected-1.cper")"
expect "a corrected record: the header" \
  "472 CPER 0x100 0xffffffff 1 2 0x3 472 0x1002:0x74A1 amdgpu 2DCE8BB1-BDD7-450E-B9AD-9CF4EBD4F890 0:1" \
  "${c[*]:0:12}"
expect "its section: AMD non-standard, the primary one, from OAM0" \
  "200 272 22.1 0x2 0x1 32AC0C78-2623-48F6-81A2-AC691780551D 2 OAM0 1 1 1 128" "${c[*]:12:12}"
expect "the UMC bank's ACA status and IPID" "dc2040000000011b 209600191f00" "${c[*]:24:2}"
expect "stamped today, in UTC" "$(date -u +%Y/%m/%d)" "${c[26]}"
IFS=' ' read -r -a u <<< "$(read_record "$tmp/cper/uncorrected-3.cper")"
expect "an uncorrected one is non-fatal, a machine check, the other GPU's first" \
  "0 E8F56FFE-919C-4CC5-BA88-65ABE14913BB 1:1" "${u[5]} ${u[10]} ${u[11]}"
expect "flagged latent, as poison is, with Deferred in its status" "0x11 0 OAM1 dc2050000000011b" \
  "${u[16]} ${u[18]} ${u[19]} ${u[24]}"
expect "the .json is the header amd-smi reports" \
  "non_fatal_uncorrected MCE CPER 256 472 0x1002:0x74A1 amdgpu 1:1" \
  "$(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(j["error_severity"], j["notify_type"], j["signature"], j["revision"], j["record_length"], j["platform_id"], j["creator_id"], j["record_id"])' "$tmp/cper/uncorrected-3.json")"

expect "--severity picks the records written" "uncorrected-1.cper" \
  "$(v smi --amd ras --cper --severity nonfatal-uncorrected --folder "$tmp/only" | awk 'NR > 1 {print $5}')"
v smi --amd ras --cper --folder "$tmp/limit" --file-limit 2 >/dev/null
expect "--file-limit keeps the newest" "2" "$(ls "$tmp/limit"/*.cper | wc -l | tr -d ' ')"
expect "each with its .json" "2" "$(ls "$tmp/limit"/*.json | wc -l | tr -d ' ')"
expect "--file-limit needs a folder" "2" "$(v smi --amd ras --cper --file-limit 2 >/dev/null; echo $?)"
expect "--follow is refused by name" "yes" \
  "$(grep -q -- '--follow is not modelled' <<< "$(v smi --amd ras --cper --follow)" && echo yes || echo no)"
exit $fail
