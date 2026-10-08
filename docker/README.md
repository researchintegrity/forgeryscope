# Forgeryscope tool image

A Docker image that runs the single-image pipeline of Forgeryscope on one
figure and writes its findings to a directory. [ELIES](https://github.com/researchintegrity/elies)
uses it as a copy-move detection method (`method: "forgeryscope"`), but it
works on its own too.

The pipeline is `forgeryscope.pipeline.DuplicationDetector`, the
single-image example of the main README with every model loaded once:

1. YOLO detects the western blot and microscopy panels.
2. Embedding models select candidate panel pairs.
3. LightGlue (SIFT features for microscopy, ALIKED fine-tuned on blots for
   western blots) locates the duplicated regions of each candidate pair.
4. When no pair of blot panels is a candidate, individual blot lanes are
   compared.

All model weights are downloaded while the image is built (their SHA256 is
checked), so the container needs no network access when it runs.

## Build

From the repository root (BuildKit is required, as in any recent Docker):

```bash
docker build -f docker/Dockerfile -t forgeryscope:latest .
```

The default image uses CPU PyTorch. For a GPU image, build with CUDA wheels
and run it with the NVIDIA container runtime:

```bash
docker build -f docker/Dockerfile -t forgeryscope:latest \
    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 .
```

| Build argument | Default | Use |
|---|---|---|
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cpu` | Where PyTorch 2.11 comes from (CPU or a CUDA index). |
| `FORGERYSCOPE_MODEL_BASE_URL` | the `models-v1` release of `vlad3996/forgeryscope` | Where the checkpoints are downloaded from (e.g. a mirror). |
| `LIGHTGLUE_COMMIT`, `ALIKED_COMMIT` | pinned commits | LightGlue code and the ALIKED weights. |
| `BASE_IMAGE` | `python:3.11-slim` | Base image (e.g. one that trusts a corporate proxy's CA). |

## Run

```bash
docker run --rm --network none \
    -v /path/to/figures:/input_vol:ro -v /path/to/results:/output_vol \
    forgeryscope:latest --input /input_vol/figure.png --output /output_vol
```

| Option | Default | |
|---|---|---|
| `--input`, `-i` | required | Image to analyse. |
| `--output`, `-o` | required | Directory for the results (created if needed). |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` (a GPU when available). |
| `--timeout` | `0` (none) | Stop after this many seconds. |
| `--max-pixels` | `200000000` | Largest image accepted (width x height). |

Images are read as image viewers show them: the EXIF orientation is applied,
and 16-bit grayscale images are scaled to 8 bits.

The container can run as any user (`--user 1000:1000`): it writes only to
`/tmp` and the output directory.

Progress is printed on stdout as `[STATUS] ...` lines, errors on stderr as
`[ERROR] ...`. Exit codes: `0` done (whether or not duplication was found),
`1` invalid input (missing, unreadable or too large image), `2` detection
failed, `3` timed out.

## Results

For an input `figure.png`:

- `figure_result.json`, always written.
- `figure_matches.png`: the figure with the detected panels boxed in grey,
  and each group of duplicated regions filled, outlined and numbered in its
  colour. Only written when duplication is found.
- `figure_clusters.png`: the duplicated regions on black, one colour per
  group. Only written when duplication is found.

The report:

```json
{
  "tool": "forgeryscope",
  "version": "1.0.1",
  "image": {"filename": "figure.png", "width": 2008, "height": 1444},
  "device": "cpu",
  "verdict": "duplicated",
  "panels": [
    {"id": 0, "label": "Blots", "confidence": 0.972, "bbox": [305.7, 1095.1, 792.1, 1310.1]}
  ],
  "detections": [
    {
      "id": 1,
      "colour": "#e6194b",
      "level": "panel",
      "panel_type": "Blots",
      "panel_ids": [3, 4],
      "similarity": 0.9746,
      "bbox": [305, 286, 792, 1421],
      "area": 85536,
      "pairs": [
        {
          "panel_ids": [3, 4],
          "embedding_score": 0.9746,
          "keypoint_match": true,
          "inliers": 37,
          "total_matches": 52,
          "match_score": 0.81,
          "regions": [[[305.9, 1321.8], ...], [[306.1, 286.1], ...]]
        }
      ],
      "lane_matches": []
    }
  ],
  "outputs": {"matches_image": "figure_matches.png", "clusters_image": "figure_clusters.png"},
  "parameters": {"panel_conf_threshold": 0.7, ...},
  "timings": {"load_models": 10.1, "detection": 2.0}
}
```

- `verdict`: `authentic` (no detection) or `duplicated`. An authentic verdict
  with no `panels` means no blot or microscopy panel was found, so nothing
  was compared.
- `detections`: one entry per group of duplicated regions, numbered and
  coloured as in the images. `level` is `panel` (panel or region matching)
  or `lane` (western blot lane matching).
- `pairs` (panel level): the candidate pairs behind the group.
  `keypoint_match` is `false` for a pair of blots kept on embedding
  similarity alone (no keypoint match): its regions are then the whole
  panels. `regions` are the duplicated region of each panel, as polygons in
  image coordinates.
- `lane_matches` (lane level): pairs of similar lanes and their boxes.
- `bbox` and `panels[].bbox` are `[x0, y0, x1, y1]` in pixels.

## Check an image

```bash
docker/smoke_test.sh forgeryscope:latest
```

runs the image on the example figures without network access and as a user
with no passwd entry, and checks the verdicts and outputs.

## Licences

The Forgeryscope code is MIT-licensed. The YOLO detector weights are
licensed by Ultralytics under AGPL-3.0, LightGlue under Apache-2.0, and
ALIKED under BSD-3-Clause. See the main README.
