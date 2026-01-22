import os
import time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding


def calculate_metrics(y_true, y_prob):
    """Calculate evaluation metrics (threshold at 0.5)."""
    # y_true: (1,1,H,W) float 0/1
    # y_prob: (1,1,H,W) float in [0,1]
    y_true = y_true.detach().cpu().numpy().astype(np.uint8).reshape(-1)
    y_pred = (y_prob.detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask):
    """Convert grayscale mask to RGB."""
    mask = np.expand_dims(mask, axis=-1)          # (H, W, 1)
    mask = np.concatenate([mask, mask, mask], -1) # (H, W, 3)
    return mask


if __name__ == "__main__":
    """Seeding"""
    seeding(42)

    """Folders"""
    results_dir = os.path.join("results")
    create_dir(results_dir)

    """Load dataset"""
    test_x = sorted(glob(os.path.join("Drive", "test", "images", "*")))
    test_y = sorted(glob(os.path.join("Drive", "test", "mask", "*")))

    assert len(test_x) > 0, "No test images found. Check your test images path."
    assert len(test_y) > 0, "No test masks found. Check your test masks path."
    assert len(test_x) == len(test_y), "Mismatch between number of test images and masks."

    """Hyperparameters"""
    H, W = 512, 512
    size = (W, H)  # cv2.resize uses (width, height)
    checkpoint_path = os.path.join("files", "checkpoint.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    """Load the checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_unet()
    model = model.to(device)

    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    processed = 0

    for i, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        try:
            """Extract the name"""
            name = os.path.splitext(os.path.basename(x_path))[0]

            """Reading image"""
            image = cv2.imread(x_path, cv2.IMREAD_COLOR)  # (H, W, 3) BGR
            if image is None:
                print(f"Failed to read image: {x_path}")
                continue

            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            x_input = np.transpose(image, (2, 0, 1)) / 255.0  # (3,H,W)
            x_input = np.expand_dims(x_input, axis=0).astype(np.float32)  # (1,3,H,W)
            x_input = torch.from_numpy(x_input).to(device)

            """Reading mask (GT)"""
            mask = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
            if mask is None:
                print(f"Failed to read mask: {y_path}")
                continue

            # 关键：mask 用最近邻，避免插值出灰度
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

            # 关键：严格二值化成0/1
            mask_bin = (mask > 127).astype(np.float32)  # (H,W) 0/1
            y_target = torch.from_numpy(mask_bin[None, None, ...]).to(device)  # (1,1,H,W)

            with torch.no_grad():
                """Prediction and time"""
                start_time = time.time()

                preds = model(x_input)      # Deep Supervision: [y1,y2,y3,y4]
                logits = preds[-1]          # final output
                prob = torch.sigmoid(logits)

                total_time = time.time() - start_time
                time_taken.append(total_time)

                """Metrics"""
                score = calculate_metrics(y_target, prob)
                metrics_score = list(map(add, metrics_score, score))

                """Post-process prediction for saving"""
                prob_np = prob[0, 0].detach().cpu().numpy()       # (H,W) float
                pred_bin = (prob_np > 0.5).astype(np.uint8)       # (H,W) 0/1

            """Saving visualization (image | GT | Pred)"""
            # 显示用：把0/1转成0/255
            ori_mask = mask_parse((mask_bin * 255).astype(np.uint8))
            pred_mask = mask_parse((pred_bin * 255).astype(np.uint8))

            line = (np.ones((H, 10, 3)) * 128).astype(np.uint8)
            combined_image = np.concatenate([image, line, ori_mask, line, pred_mask], axis=1)

            save_path = os.path.join(results_dir, f"{name}.png")
            ok = cv2.imwrite(save_path, pred_mask)
            if not ok:
                print(f"Failed to save result: {save_path}")

            processed += 1

        except Exception as e:
            print(f"Error processing {x_path}: {e}")

    if processed == 0:
        raise RuntimeError("No samples were processed. Please check your test paths and files.")

    """Final metrics"""
    jaccard = metrics_score[0] / processed
    f1 = metrics_score[1] / processed
    recall = metrics_score[2] / processed
    precision = metrics_score[3] / processed
    acc = metrics_score[4] / processed

    print(f"Processed: {processed}/{len(test_x)}")
    print(f"Jaccard: {jaccard:.4f} - F1: {f1:.4f} - Recall: {recall:.4f} - Precision: {precision:.4f} - Acc: {acc:.4f}")

    fps = 1.0 / np.mean(time_taken)
    print("FPS:", fps)
