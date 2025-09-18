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
# 2. 좌표 변환 함수 (원본 → SlimSAM target_size)
# -----------------------------
def resize_boxes_for_slimsam(boxes, orig_size, target_size):
    """
    boxes: (N,4) [x1,y1,x2,y2] (원본 좌표계)
    orig_size: (H,W) 원본 이미지 크기
    target_size: (H,W) SlimSAM processor 내부 리사이즈 크기
    """
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size

    scale_x = tgt_w / orig_w
    scale_y = tgt_h / orig_h

    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes


# -----------------------------
# 3. SlimSAM 실행 함수
# -----------------------------
def run_slimsam_with_bbox(img_path, bbox_orig):
    print(f"\n[INFO] Processing {img_path}")

    # 원본 이미지 읽기
    orig = cv2.imread(img_path)
    if orig is None:
        raise RuntimeError(f"[ERROR] cv2.imread failed for {img_path}")
    h, w = orig.shape[:2]
    print(f"[DEBUG] Original image shape: {orig.shape}")

    # PIL 이미지 준비
    image = Image.open(img_path).convert("RGB")

    # SlimSAM target_size 얻기
    tmp_inputs = processor(image, return_tensors="pt").to(device)
    target_size = tmp_inputs["reshaped_input_sizes"][0].tolist()  # e.g. [1024,1024]

    # CSV bbox (원본 좌표 → SlimSAM 좌표 변환)
    x1, y1, x2, y2 = map(int, bbox_orig)
   
    bbox_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    resized_boxes = resize_boxes_for_slimsam(bbox_np, (h, w), target_size)

    # SlimSAM input
    input_boxes = [[[float(x) for x in resized_boxes[0]]]]
    inputs = processor(
        image,
        input_points=None, # [[[x1, y1],[x2, y2]]]
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
        print(f"[DEBUG] Mask has 3 channels, taking first channel. shape={mask.shape}")
        mask = mask[0]

    print(f"[DEBUG] raw mask stats: min={mask.min()}, max={mask.max()}, mean={mask.mean()}")

    thresh = 0.5 if mask.max() > 0.6 else 0.05
    mask_bin = (mask > thresh).astype(np.uint8) * 255

    if mask_bin.shape != (h, w):
        print(f"[DEBUG] Resizing mask from {mask_bin.shape} -> ({h}, {w})")
        mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    # -----------------------------
    # 저장
    # -----------------------------
    base = os.path.splitext(img_path)[0]
    mask_out = base + "_slimsam_mask.png"
    overlay_out = base + "_slimsam_overlay.png"
    bbox_debug = base + "_debug_bbox.png"
    inputbox_debug = base + "_slimsam_inputbox.png"

    cv2.imwrite(mask_out, mask_bin)
    print(f"[SUCCESS] Saved mask: {mask_out}")

    # 초록색 박스 (원본 bbox)
    debug_img = orig.copy()
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(bbox_debug, debug_img)
    print(f"[DEBUG] Saved bbox debug: {bbox_debug}")

    # 파란색 박스 (SlimSAM inputbox, 변환된 좌표계)
    inputbox_img = orig.copy()
    rx1, ry1, rx2, ry2 = resized_boxes[0]
    # 변환된 좌표를 다시 원본 좌표계에 맞춰 그림
    # (역변환: target_size -> 원본 size)
    rx1, rx2 = int(rx1 * w / target_size[1]), int(rx2 * w / target_size[1])
    ry1, ry2 = int(ry1 * h / target_size[0]), int(ry2 * h / target_size[0])
    cv2.rectangle(inputbox_img, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
    cv2.imwrite(inputbox_debug, inputbox_img)
    print(f"[DEBUG] Saved SlimSAM inputbox: {inputbox_debug}")

    # overlay 저장
    overlay = orig.copy()
    overlay[mask_bin > 0] = (0, 0, 255)
    blended = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
    cv2.imwrite(overlay_out, blended)
    print(f"[SUCCESS] Saved overlay: {overlay_out}")


# -----------------------------
# 4. 메인 실행부
# -----------------------------
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SlimSAM with bounding boxes from CSV")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="bboxes.csv",
        help="Path to CSV file containing bounding boxes",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="Conjunctival Images for Anemia Detection",
        help="Root directory of the dataset",
    )

    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["filename"].replace("\\", "/")
            img_path = os.path.join(args.dataset_root, rel)

            if not os.path.exists(img_path):
                print(f"⚠️ Skip (not found): {img_path}")
                continue

            try:
                x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
                bbox_orig = [x1, y1, x2, y2]
                run_slimsam_with_bbox(img_path, bbox_orig)
            except Exception as e:
                print(f"❌ Failed {rel}: {e}")
