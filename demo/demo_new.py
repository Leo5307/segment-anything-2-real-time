import cv2
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor

# 初始化模型
sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# 打開攝像頭
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if_init = False
tracking_i = 0
warmup_frames = 10
next_obj_id = 1  # 下個要新增物件的 ID

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    # 丟掉前幾幀避免黑畫面
    if not if_init and warmup_frames > 0:
        warmup_frames -= 1
        continue

    if not if_init:
        # 第一幀初始化
        predictor.load_first_frame(frame_rgb)

        # 使用者框選第一個物件
        bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Object")
        x, y, w, h = bbox
        bbox_np = np.array([[x, y], [x + w, y + h]], dtype=np.float32)

        # 新增追蹤物件
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=0, obj_id=next_obj_id, bbox=bbox_np
        )
        next_obj_id += 1
        if_init = True

    else:
        # 後續幀追蹤
        out_obj_ids, out_mask_logits = predictor.track(frame_rgb)
        tracking_i += 1

        # 每 20 幀更新 conditioning frame 防止 mask 消失
        if tracking_i % 20 == 0:
            predictor.add_conditioning_frame(frame_rgb)

        # 可視化 mask
        all_mask = np.zeros((height, width, 3), dtype=np.uint8)
        all_mask[..., 1] = 255
        for i in range(len(out_obj_ids)):
            mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            hue = (i + 3) / (len(out_obj_ids) + 3) * 255
            all_mask[mask[..., 0] == 255, 0] = hue
            all_mask[mask[..., 0] == 255, 2] = 255

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
        frame_rgb = cv2.addWeighted(frame_rgb, 1, all_mask, 0.5, 0)

    # 顯示畫面
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Tracking", frame_bgr)

    key = cv2.waitKey(1) & 0xFF

    # 按 'a' 新增物件
    if key == ord("a"):
        # 暫停畫面讓使用者框選
        bbox = cv2.selectROI("Add Object", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Add Object")
        x, y, w, h = bbox
        bbox_np = np.array([[x, y], [x + w, y + h]], dtype=np.float32)

        # 運行中新增追蹤物件
        predictor.add_new_prompt_during_track(
            bbox=bbox_np,
            obj_id=next_obj_id,
            if_new_target=True,
            clear_old_points=False
        )
        next_obj_id += 1

    # 按 'q' 離開
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
