# OmniParser V2 API

Self-hosted version of Microsoft's [OmniParser V2](https://huggingface.co/microsoft/OmniParser-v2.0) Image-to-text model.

> OmniParser is a general screen parsing tool, which interprets/converts UI screenshot to structured format, to improve existing LLM based UI agent. Training Datasets include: 1) an interactable icon detection dataset, which was curated from popular web pages and automatically annotated to highlight clickable and actionable regions, and 2) an icon description dataset, designed to associate each UI element with its corresponding function.

## Why?

There's already a great HuggingFace gradio [app](https://huggingface.co/spaces/microsoft/OmniParser-v2) for this model. It even offers an API. But

- Gradio is much slower than serving the model directly (like we do here)
- HF is rate-limited

## How it works

If you look at the Dockerfile, we start off with the [V2 HF demo image](https://huggingface.co/spaces/microsoft/OmniParser-v2) to retrieve the util functions. Then we add a simple FastAPI server (under main.py) to serve the model.

The V2 weights (`icon_detect/model.pt` + `icon_caption`) are pulled from the [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0) repo into `weights/` at build time, so the first request doesn't have to wait for the download. Override with `OMNIPARSER_WEIGHTS_REPO` / `OMNIPARSER_WEIGHTS_DIR` if you want to point at your own copy.

## Getting Started

### Requirements

- GPU
- 16 GB Ram (swap recommended)

### Locally

1. Clone the repository
2. Build the docker image: `docker build -t omni-parser-app .`
3. Run the docker container: `docker run -p 7860:7860 omni-parser-app`

### Self-hosted API

I suggest hosting on [fly.io](https://fly.io) because it's quick and simple to deploy with a CLI.

This repo is ready-made for deployment on fly.io (see fly.toml for configuration). Just run `fly launch` and follow the prompts.

## Docs

Visit `http://localhost:7860/docs` for the API documentation. The main route is `POST /process_image` (there's also a `GET /health`).

Query parameters:

| Name            | Default | Description                                      |
| --------------- | ------- | ------------------------------------------------ |
| `box_threshold` | `0.05`  | Confidence threshold for the icon detector        |
| `iou_threshold` | `0.1`   | Overlap threshold used to drop duplicate boxes    |
| `use_paddleocr` | `true`  | PaddleOCR when true, EasyOCR when false           |
| `imgsz`         | `640`   | Detector input size, 640-1920. Bigger = more boxes, slower |

The response contains

- `image` — the image with bounding boxes drawn on, base64 encoded PNG
- `parsed_content_list` — the parsed elements, each with `type` (`icon` or `text`), `bbox` (`[x1, y1, x2, y2]` as ratios), `interactivity`, `content` and `source`
- `label_coordinates` — box id to `[x, y, w, h]` as ratios of the image size

```json
{
  "image": "iVBORw0KGgo...",
  "parsed_content_list": [
    {
      "type": "text",
      "bbox": [0.11, 0.04, 0.19, 0.06],
      "interactivity": false,
      "content": "Inbox",
      "source": "box_ocr_content_ocr"
    }
  ],
  "label_coordinates": { "0": [0.11, 0.04, 0.08, 0.02] }
}
```

> **Upgrading from the V1 version of this repo:** `parsed_content_list` used to be a newline-joined string and `label_coordinates` a stringified dict. Both are now real JSON, and `use_paddleocr` / `imgsz` are new.

## Examples

| Before Image                       | After Image                   |
| ---------------------------------- | ----------------------------- |
| ![Before](examples/screenshot.png) | ![After](examples/after.webp) |

## Related Projects

Check out [OneQuery](https://query-rho.vercel.app), an agent that browses the web and returns structured responses for any query, simple or complex. OneQuery is built using OmniParser to enhance its capabilities.
