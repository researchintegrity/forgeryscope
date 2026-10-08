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

# A 16-bit grayscale copy and a copy stored rotated with an EXIF orientation
# tag must be read as the original figure
mkdir "$OUT/variants" && chmod 777 "$OUT/variants"
docker run --rm --network none --user 12345:12345 --entrypoint python \
    -v "$ROOT/examples/sample_data:/input_vol:ro" -v "$OUT/variants:/output_vol" "$IMAGE" -c '
import numpy as np
from PIL import Image
figure = Image.open("/input_vol/kaggle_test_img_45.png").convert("RGB")
Image.fromarray(np.array(figure.convert("L")).astype(np.uint16) * 257).save("/output_vol/gray16.png")
exif = Image.Exif()
exif[0x0112] = 6
figure.transpose(Image.Transpose.ROTATE_90).save("/output_vol/rotated.jpg", quality=95, exif=exif)
'
for variant in gray16.png rotated.jpg; do
    docker run --rm --network none --security-opt no-new-privileges --user 12345:12345 \
        -v "$OUT/variants:/input_vol:ro" -v "$OUT:/output_vol" \
        "$IMAGE" --input "/input_vol/$variant" --output /output_vol --device cpu --timeout 900
    stem="${variant%.*}"
    [ "$(verdict "$stem")" = duplicated ] || { echo "FAIL: $variant should be duplicated"; exit 1; }
    python3 -c 'import json, sys; image = json.load(open(sys.argv[1]))["image"]; sys.exit((image["width"], image["height"]) != (2008, 1444))' \
        "$OUT/${stem}_result.json" || { echo "FAIL: $variant should be read upright as 2008x1444"; exit 1; }
done

set +e
run missing.png
code=$?
set -e
[ "$code" -eq 1 ] || { echo "FAIL: a missing input should exit with 1, got $code"; exit 1; }

echo "OK"
