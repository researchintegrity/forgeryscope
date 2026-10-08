#!/usr/bin/env bash
# Run the image on the example figures, without network access and as a
# user with no passwd entry (as ELIES runs it), and check the results.
#
#   docker/smoke_test.sh [image]   (default: forgeryscope:latest)
set -euo pipefail

IMAGE="${1:-forgeryscope:latest}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
chmod 777 "$OUT"

run() {
    docker run --rm --network none --security-opt no-new-privileges --user 12345:12345 \
        -v "$ROOT/examples/sample_data:/input_vol:ro" -v "$OUT:/output_vol" \
        "$IMAGE" --input "/input_vol/$1" --output /output_vol --device cpu --timeout 900
}

verdict() {
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["verdict"])' "$OUT/$1_result.json"
}

run kaggle_test_img_45.png
[ "$(verdict kaggle_test_img_45)" = duplicated ] || { echo "FAIL: kaggle_test_img_45 should be duplicated"; exit 1; }
[ -s "$OUT/kaggle_test_img_45_matches.png" ] && [ -s "$OUT/kaggle_test_img_45_clusters.png" ] \
    || { echo "FAIL: result images missing"; exit 1; }

run wblot_sample.png
[ "$(verdict wblot_sample)" = authentic ] || { echo "FAIL: wblot_sample should be authentic"; exit 1; }
[ ! -e "$OUT/wblot_sample_matches.png" ] || { echo "FAIL: no result image expected for an authentic figure"; exit 1; }

set +e
run missing.png
code=$?
set -e
[ "$code" -eq 1 ] || { echo "FAIL: a missing input should exit with 1, got $code"; exit 1; }

echo "OK"
