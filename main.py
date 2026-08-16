"""FastAPI server for Microsoft OmniParser V2."""

import io
import os
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

# The OmniParser V2 image ships the helpers as a `util` package, the V1 image kept
# them at the top level. Support both so this file works on either base image.
try:
    from util.utils import (
        check_ocr_box,
        get_caption_model_processor,
        get_som_labeled_img,
        get_yolo_model,
    )
except ImportError:  # OmniParser V1 layout
    from utils import (
        check_ocr_box,
        get_caption_model_processor,
        get_som_labeled_img,
        get_yolo_model,
    )

WEIGHTS_REPO = os.getenv("OMNIPARSER_WEIGHTS_REPO", "microsoft/OmniParser-v2.0")
WEIGHTS_DIR = os.getenv("OMNIPARSER_WEIGHTS_DIR", "weights")

ICON_DETECT_PATH = os.path.join(WEIGHTS_DIR, "icon_detect", "model.pt")
ICON_CAPTION_DIR = os.path.join(WEIGHTS_DIR, "icon_caption")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def caption_dir() -> str:
    """Path of the Florence-2 caption weights.

    The upstream README tells you to rename `icon_caption` to
    `icon_caption_florence`, so accept either name.
    """
    legacy = os.path.join(WEIGHTS_DIR, "icon_caption_florence")
    if not os.path.isdir(ICON_CAPTION_DIR) and os.path.isdir(legacy):
        return legacy
    return ICON_CAPTION_DIR


def ensure_weights() -> None:
    """Download the V2 weights unless they are already on disk."""
    if os.path.isfile(ICON_DETECT_PATH) and os.path.isdir(caption_dir()):
        return

    from huggingface_hub import snapshot_download

    print(f"downloading {WEIGHTS_REPO} -> {WEIGHTS_DIR}")
    snapshot_download(repo_id=WEIGHTS_REPO, local_dir=WEIGHTS_DIR)


ensure_weights()

yolo_model = get_yolo_model(model_path=ICON_DETECT_PATH)
yolo_model.to(DEVICE)

caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path=caption_dir(),
    device=DEVICE,
)
print(f"finish loading model on {DEVICE}!!!")

app = FastAPI(title="OmniParser V2 API", version="2.0.0")


class ParsedElement(BaseModel):
    type: str = Field("icon", description="Either 'icon' or 'text'")
    bbox: List[float] = Field(
        default_factory=list, description="[x1, y1, x2, y2] as ratios of the image size"
    )
    interactivity: bool = Field(False, description="Whether the element looks clickable")
    content: Optional[str] = Field(None, description="OCR text or generated caption")
    source: Optional[str] = Field(None, description="Which model produced the element")


class ProcessResponse(BaseModel):
    image: str = Field(..., description="Base64 encoded PNG with the boxes drawn on")
    parsed_content_list: List[ParsedElement]
    label_coordinates: Dict[str, List[float]] = Field(
        ..., description="Box id -> [x, y, w, h] as ratios of the image size"
    )


def to_float_list(value: Any) -> List[float]:
    """Coerce numpy arrays / tensors coming out of the models into plain floats."""
    return [float(v) for v in value]


@torch.inference_mode()
def process(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int,
) -> ProcessResponse:
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_input,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=use_paddleocr,
    )
    text, ocr_bbox = ocr_bbox_rslt

    labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_input,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
    print("finish processing")

    # `labeled_img` is already a base64 encoded PNG, no need to round-trip it.
    return ProcessResponse(
        image=labeled_img,
        parsed_content_list=[
            ParsedElement(
                type=element.get("type", "icon"),
                bbox=to_float_list(element.get("bbox", [])),
                interactivity=bool(element.get("interactivity", False)),
                content=element.get("content"),
                source=element.get("source"),
            )
            for element in parsed_content_list
        ],
        label_coordinates={
            str(key): to_float_list(value) for key, value in label_coordinates.items()
        },
    )


@app.post("/process_image", response_model=ProcessResponse)
async def process_image(
    image_file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = True,
    imgsz: int = 640,
):
    try:
        contents = await image_file.read()
        image_input = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return process(image_input, box_threshold, iou_threshold, use_paddleocr, imgsz)


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "model": WEIGHTS_REPO}
