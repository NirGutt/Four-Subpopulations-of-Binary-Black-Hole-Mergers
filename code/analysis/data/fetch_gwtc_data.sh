#!/usr/bin/env bash

# Download all event-level PE posterior files from GWTC-2.1 through GWTC-5,
# together with the cumulative Cartesian-spin injection files used for
# population inference.
#
# Usage:
#   ./fetch_gwtc_data.sh [DATA_DIR]
#
# Examples:
#   ./fetch_gwtc_data.sh
#   ./fetch_gwtc_data.sh ../data
#
# The default output directory is ../data. Existing complete files are skipped.
# Interrupted downloads are resumed from temporary ".part" files. The script
# uses wget when available and otherwise falls back to curl on Mac.

set -euo pipefail

DistancePrior="${DistancePrior:-cosmo}"
DATA_DIR="${1:-../data}"

POSTERIOR_DIR="${DATA_DIR}/posteriors"
INJECTION_DIR="${DATA_DIR}/injections"

ZENODO_API="https://zenodo.org/api/records"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: required command 'python3' was not found." >&2
    exit 1
fi

if command -v wget >/dev/null 2>&1; then
    DOWNLOAD_COMMAND="wget"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOAD_COMMAND="curl"
else
    echo "Error: either 'wget' or 'curl' is required." >&2
    exit 1
fi

mkdir -p "${POSTERIOR_DIR}" "${INJECTION_DIR}"

case "${DistancePrior}" in
    cosmo|nocosmo)
        ;;
    *)
        echo "Error: DistancePrior must be 'cosmo' or 'nocosmo'." >&2
        exit 1
        ;;
esac

# Produce a tab-separated manifest:
#
# filename<TAB>download URL<TAB>md5 checksum<TAB>size in bytes
zenodo_manifest() {
    local record_id="$1"
    local filename_regex="$2"

    python3 - \
        "${record_id}" \
        "${filename_regex}" \
        "${ZENODO_API}" <<'PY'
import json
import re
import sys
import urllib.error
import urllib.request

record_id, filename_regex, api_root = sys.argv[1:]

api_url = f"{api_root}/{record_id}"

request = urllib.request.Request(
    api_url,
    headers={"User-Agent": "GWTC-population-data-downloader/1.0"},
)

try:
    with urllib.request.urlopen(request, timeout=120) as response:
        record = json.load(response)
except urllib.error.HTTPError as error:
    raise SystemExit(
        f"Could not read Zenodo record {record_id}: HTTP {error.code}"
    ) from error
except urllib.error.URLError as error:
    raise SystemExit(
        f"Could not read Zenodo record {record_id}: {error.reason}"
    ) from error

pattern = re.compile(filename_regex)
matches = []

files = record.get("files", [])

if isinstance(files, dict):
    files = files.get("entries", {}).values()

for item in files:
    filename = item.get("key", "")

    if not pattern.fullmatch(filename):
        continue

    links = item.get("links", {})
    url = links.get("content") or links.get("self")

    if not url:
        raise SystemExit(
            f"Zenodo did not provide a download URL for {filename}"
        )

    checksum = item.get("checksum", "")

    if checksum.startswith("md5:"):
        checksum = checksum[4:]

    size = item.get("size", "")
    matches.append((filename, url, checksum, size))

if not matches:
    raise SystemExit(
        f"No files in Zenodo record {record_id} matched: {filename_regex}"
    )

for row in sorted(matches):
    print(*row, sep="\t")
PY
}

file_md5() {
    python3 - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.md5()

with open(sys.argv[1], "rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(block)

print(digest.hexdigest())
PY
}

human_size() {
    python3 - "$1" <<'PY'
import sys

size = int(sys.argv[1]) if sys.argv[1] else 0
units = ("B", "KiB", "MiB", "GiB", "TiB")

value = float(size)
unit = units[0]

for unit in units:
    if value < 1024 or unit == units[-1]:
        break
    value /= 1024

print(f"{value:.1f} {unit}")
PY
}

download_url() {
    local url="$1"
    local output="$2"

    if [[ "${DOWNLOAD_COMMAND}" == "wget" ]]; then
        wget \
            --continue \
            --tries=10 \
            --timeout=60 \
            --waitretry=5 \
            --output-document="${output}" \
            "${url}"
    else
        curl \
            --fail \
            --location \
            --continue-at - \
            --retry 10 \
            --retry-delay 5 \
            --connect-timeout 60 \
            --output "${output}" \
            "${url}"
    fi
}

download_zenodo_record() {
    local label="$1"
    local record_id="$2"
    local destination="$3"
    local filename_regex="$4"

    local manifest
    local number_of_files

    mkdir -p "${destination}"

    manifest="$(mktemp)"
    trap 'rm -f "${manifest:-}"' RETURN

    echo
    echo "Reading ${label} file list from Zenodo record ${record_id}..."

    zenodo_manifest \
        "${record_id}" \
        "${filename_regex}" > "${manifest}"

    number_of_files="$(wc -l < "${manifest}" | tr -d ' ')"

    echo "Found ${number_of_files} matching file(s)."

    while IFS=$'\t' read -r filename url expected_md5 size; do
        local output="${destination}/${filename}"
        local partial="${output}.part"

        if [[ -f "${output}" ]]; then
            if [[ -z "${expected_md5}" ]] ||
               [[ "$(file_md5 "${output}")" == "${expected_md5}" ]]; then
                echo "Already complete: ${output}"
                continue
            fi

            echo "Checksum mismatch for existing file: ${output}" >&2
            echo "Move or remove that file, then run this script again." >&2
            exit 1
        fi

        echo
        echo "Downloading ${filename} ($(human_size "${size}"))"

        download_url "${url}" "${partial}"

        if [[ -n "${expected_md5}" ]] &&
           [[ "$(file_md5 "${partial}")" != "${expected_md5}" ]]; then
            echo "Checksum verification failed: ${partial}" >&2
            exit 1
        fi

        mv "${partial}" "${output}"

        echo "Verified: ${output}"
    done < "${manifest}"

    rm -f "${manifest}"
    trap - RETURN
}

echo "Output directory: ${DATA_DIR}"
echo "GWTC-2.1/GWTC-3 distance prior: ${DistancePrior}"
echo "Download command: ${DOWNLOAD_COMMAND}"

# GWTC-2.1
download_zenodo_record \
    "GWTC-2.1 posterior" \
    "6513631" \
    "${POSTERIOR_DIR}/GWTC-2.1" \
    "IGWN-GWTC2p1-v2-GW[0-9]{6}_[0-9]{6}_PEDataRelease_mixed_${DistancePrior}\.h5"

# GWTC-3
download_zenodo_record \
    "GWTC-3 posterior" \
    "5546663" \
    "${POSTERIOR_DIR}/GWTC-3" \
    "IGWN-GWTC3p0-v1-GW[0-9]{6}_[0-9]{6}_PEDataRelease_mixed_${DistancePrior}\.h5"

# GWTC-4
download_zenodo_record \
    "GWTC-4 posterior" \
    "16053484" \
    "${POSTERIOR_DIR}/GWTC-4" \
    "IGWN-GWTC4p0-.*-GW[0-9]{6}_[0-9]{6}-combined_PEDataRelease\.hdf5"

# GWTC-5 Stable Release 9, part 1
download_zenodo_record \
    "GWTC-5 Stable Release 9 posterior, part 1" \
    "20276105" \
    "${POSTERIOR_DIR}/GWTC-5" \
    "IGWN-GWTC5p0-.*-GW[0-9]{6}_[0-9]{6}-combined_PEDataRelease\.hdf5"

# GWTC-5 Stable Release 9, part 2
download_zenodo_record \
    "GWTC-5 Stable Release 9 posterior, part 2" \
    "20291739" \
    "${POSTERIOR_DIR}/GWTC-5" \
    "IGWN-GWTC5p0-.*-GW[0-9]{6}_[0-9]{6}-combined_PEDataRelease\.hdf5"

# Cumulative O1+O2+O3+O4a+O4b Cartesian-spin injections
download_zenodo_record \
    "GWTC-5 cumulative injections" \
    "19500052" \
    "${INJECTION_DIR}" \
    "mixture-semi_o1_o2-real_o3_o4a_o4b-cartesian_spins_20260410130052UTC-clipped\.hdf"

echo
echo "All requested GWTC posterior and injection files are present."
