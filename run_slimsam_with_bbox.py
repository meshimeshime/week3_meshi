import os
import csv
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

# -----------------------------
# 1. 모델 준비
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Zigeng/SlimSAM-uniform-50"
model = SamModel.from_pretrained(model_id).to(device)
processor = SamProcessor.from_pretrained(model_id)


# -----------------------------
# 2. SlimSAM 실행 함수
# -----------------------------
def run_slimsam_with_bbox(img_path, bbox_orig):
    print(f"\n[INFO] Processing {img_path}")

    # 원본 이미지 읽기
    orig = cv2.imread(img_path)
    if orig is None:
        raise RuntimeError(f"[ERROR] cv2.imread failed for {img_path}")
    h, w = orig.shape[:2]

    # PIL 이미지 준비
    image = Image.open(img_path).convert("RGB")

    # CSV bbox (원본 좌표 그대로 사용)
    x1, y1, x2, y2 = map(int, bbox_orig)
    input_boxes = [[[float(x1), float(y1), float(x2), float(y2)]]]

    # SlimSAM input
    inputs = processor(
        image,
        input_points=None,
        input_boxes=input_boxes,
        return_tensors="pt"
    ).to(device)

    # 모델 실행
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    # 마스크 처리
    mask = masks[0][0].cpu().numpy().astype(np.float32)
    if mask.ndim == 3 and mask.shape[0] == 3:
        mask = mask[0]

    thresh = 0.5 if mask.max() > 0.6 else 0.05
    mask_bin = (mask > thresh).astype(np.uint8) * 255

    if mask_bin.shape != (h, w):
        mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    # -----------------------------
    # 저장
    # -----------------------------
    base = os.path.splitext(img_path)[0]
    mask_out = base + "_slimsam_mask.png"
    overlay_out = base + "_slimsam_overlay.png"
    bbox_debug = base + "_debug_bbox.png"

    cv2.imwrite(mask_out, mask_bin)
    print(f"[SUCCESS] Saved mask: {mask_out}")

    # 초록색 박스 (원본 bbox)
    debug_img = orig.copy()
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(bbox_debug, debug_img)
    print(f"[DEBUG] Saved bbox debug: {bbox_debug}")

    # overlay 저장
    overlay = orig.copy()
    overlay[mask_bin > 0] = (0, 0, 255)
    blended = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
    cv2.imwrite(overlay_out, blended)
    print(f"[SUCCESS] Saved overlay: {overlay_out}")


# -----------------------------
# 3. 메인 실행부
# -----------------------------
if __name__ == "__main__":
    csv_path = "bboxes.csv"
    dataset_root = "Conjunctival Images for Anemia Detection"

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["filename"].replace("\\", "/")
            img_path = os.path.join(dataset_root, rel)

            if not os.path.exists(img_path):
                print(f"⚠️ Skip (not found): {img_path}")
                continue

            try:
                x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
                bbox_orig = [x1, y1, x2, y2]
                run_slimsam_with_bbox(img_path, bbox_orig)
            except Exception as e:
                print(f"❌ Failed {rel}: {e}")