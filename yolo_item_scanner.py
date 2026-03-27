import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

try:
    import cv2
except ImportError:
    print("OpenCV fehlt. Installiere es mit: pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics fehlt. Installiere es mit: pip install ultralytics")
    sys.exit(1)


WINDOW_NAME = "YOLO Item Scanner"
WEBHOOK_BASE_URL = "https://hook.eu1.make.celonis.com/79j69cp1aesv98zwozwgpduz86hk71rr?ItemCode="
DATASET_URL = "https://hook.eu1.make.celonis.com/2dkavnxbe1o4rns7k75r1ej9yqflfa7x"
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
CUSTOM_MODEL_PATH = MODELS_DIR / "custom-items-cls.pt"
DEFAULT_MODEL_NAME = "yolo11n-cls.pt"
MAPPING_PATH = ROOT / "classification_item_mapping.json"
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAMERA_FPS = 30
ROI_WIDTH_RATIO = 0.58
ROI_HEIGHT_RATIO = 0.70
DETECTION_INTERVAL_SECONDS = 0.30
CONFIDENCE_THRESHOLD = 0.55
STABLE_FRAMES = 2
JASSKARTEN_STABLE_FRAMES = 5
JASSKARTEN_MARGIN_THRESHOLD = 0.22
JASSKARTEN_VISUAL_SCORE_THRESHOLD = 0.26
JASSKARTEN_HIGH_CONFIDENCE = 0.96
SCHOKOLADE_OVERRIDE_CONFIDENCE = 0.72
SCHOKOLADE_VISUAL_SCORE_THRESHOLD = 0.34
RESULT_HOLD_SECONDS = 6.0
MAX_CANDIDATES = 6
MIN_OBJECT_AREA_RATIO = 0.015
MAX_OBJECT_AREA_RATIO = 0.55
MIN_OBJECT_SIDE = 70
MAX_OBJECT_ASPECT = 3.6
REGION_IOU_THRESHOLD = 0.34
PREDICTION_IOU_THRESHOLD = 0.55

COLOR_TEXT = (245, 247, 250)
COLOR_MUTED = (180, 188, 196)
COLOR_PANEL = (24, 28, 34)
COLOR_PANEL_BORDER = (92, 102, 112)
COLOR_ROI_IDLE = (110, 130, 150)
COLOR_ROI_ACTIVE = (58, 212, 166)
COLOR_WARNING = (0, 181, 255)
COLOR_SUCCESS = (58, 212, 166)
PANEL_ALPHA = 0.82


def draw_text(frame, text, position, color, scale=0.65, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(frame, top_left, bottom_right, fill_color=COLOR_PANEL, border_color=COLOR_PANEL_BORDER, alpha=PANEL_ALPHA):
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)


def draw_status_chip(frame, text, position, color):
    x, y = position
    width = 180
    height = 38
    draw_panel(frame, (x, y), (x + width, y + height), fill_color=(32, 36, 42), border_color=color, alpha=0.88)
    cv2.circle(frame, (x + 18, y + height // 2), 5, color, -1)
    draw_text(frame, text, (x + 34, y + 26), COLOR_TEXT, scale=0.62, thickness=2)


def draw_info_card(frame, title, lines, top_left, width):
    line_height = 24
    height = 54 + len(lines) * line_height
    x1, y1 = top_left
    x2, y2 = x1 + width, y1 + height
    draw_panel(frame, (x1, y1), (x2, y2))
    draw_text(frame, title, (x1 + 16, y1 + 28), COLOR_TEXT, scale=0.7, thickness=2)
    y = y1 + 55
    for line in lines:
        draw_text(frame, line, (x1 + 16, y), COLOR_MUTED, scale=0.57, thickness=1)
        y += line_height
    return height


def draw_focus_hint(frame, roi, trusted):
    x1, y1, x2, y2 = roi
    label = "Artikel mittig in die Box halten"
    color = COLOR_ROI_ACTIVE if trusted else COLOR_MUTED
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y2 + 26
    draw_text(frame, label, (text_x, text_y), color, scale=0.55, thickness=1)


def build_focus_roi(frame):
    height, width = frame.shape[:2]
    roi_width = int(width * ROI_WIDTH_RATIO)
    roi_height = int(height * ROI_HEIGHT_RATIO)
    x1 = max(0, (width - roi_width) // 2)
    y1 = max(0, (height - roi_height) // 2)
    x2 = x1 + roi_width
    y2 = y1 + roi_height
    return x1, y1, x2, y2


def draw_focus_box(frame, roi):
    x1, y1, x2, y2 = roi
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], y1), (8, 10, 14), -1)
    cv2.rectangle(overlay, (0, y2), (frame.shape[1], frame.shape[0]), (8, 10, 14), -1)
    cv2.rectangle(overlay, (0, y1), (x1, y2), (8, 10, 14), -1)
    cv2.rectangle(overlay, (x2, y1), (frame.shape[1], y2), (8, 10, 14), -1)
    cv2.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)


def draw_focus_frame(frame, roi, color):
    x1, y1, x2, y2 = roi
    corner = 26
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, thickness)


def crop_center(frame, scale=0.82):
    height, width = frame.shape[:2]
    crop_width = max(1, int(width * scale))
    crop_height = max(1, int(height * scale))
    x1 = max(0, (width - crop_width) // 2)
    y1 = max(0, (height - crop_height) // 2)
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    return frame[y1:y2, x1:x2]


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def enhance_roi_for_classification(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(enhanced, 1.12, cv2.GaussianBlur(enhanced, (0, 0), 1.0), -0.12, 0)


def analyze_jasskarten_visual_cues(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total_pixels = max(1, frame.shape[0] * frame.shape[1])

    red_mask_1 = cv2.inRange(hsv, (0, 70, 60), (12, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (168, 70, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    blue_mask = cv2.inRange(hsv, (88, 40, 40), (132, 255, 255))
    white_mask = cv2.inRange(hsv, (0, 0, 130), (180, 70, 255))

    red_ratio = cv2.countNonZero(red_mask) / total_pixels
    blue_ratio = cv2.countNonZero(blue_mask) / total_pixels
    white_ratio = cv2.countNonZero(white_mask) / total_pixels

    red_white_score = 0.0
    if 0.004 <= red_ratio <= 0.12 and white_ratio >= 0.20:
        red_component = clamp((red_ratio - 0.004) / 0.03, 0.0, 1.0)
        white_component = clamp((white_ratio - 0.20) / 0.35, 0.0, 1.0)
        red_white_score = 0.55 * red_component + 0.45 * white_component

    blue_pattern_score = 0.0
    if 0.015 <= blue_ratio <= 0.35:
        blue_pattern_score = clamp((blue_ratio - 0.015) / 0.12, 0.0, 1.0)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangle_score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < total_pixels * 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) <= 0:
            continue

        aspect_ratio = max(w / h, h / w)
        rect_area = max(1, w * h)
        solidity = area / rect_area
        area_ratio = area / total_pixels

        if 4 <= len(approx) <= 8 and 1.2 <= aspect_ratio <= 2.6 and solidity >= 0.55:
            candidate = (
                0.45 * clamp((area_ratio - 0.08) / 0.30, 0.0, 1.0)
                + 0.35 * clamp((solidity - 0.55) / 0.30, 0.0, 1.0)
                + 0.20 * clamp((aspect_ratio - 1.2) / 0.9, 0.0, 1.0)
            )
            rectangle_score = max(rectangle_score, candidate)

    visual_score = 0.45 * rectangle_score + 0.35 * red_white_score + 0.20 * blue_pattern_score
    return {
        "visual_score": visual_score,
        "rectangle_score": rectangle_score,
        "red_ratio": red_ratio,
        "blue_ratio": blue_ratio,
        "white_ratio": white_ratio,
    }


def analyze_schokolade_visual_cues(frame):
    height, width = frame.shape[:2]
    total_pixels = max(1, height * width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    qr_detector = cv2.QRCodeDetector()
    qr_text, qr_points, _ = qr_detector.detectAndDecode(frame)
    qr_present = bool(qr_text) or qr_points is not None

    white_mask = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    white_ratio = cv2.countNonZero(white_mask) / total_pixels

    dark_ratio = cv2.countNonZero(cv2.inRange(gray, 0, 90)) / total_pixels

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sticker_score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < total_pixels * 0.015:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) <= 0:
            continue

        aspect_ratio = max(w / h, h / w)
        area_ratio = area / total_pixels
        if aspect_ratio > 1.45:
            continue

        centered_x = abs((x + w / 2) - (width / 2)) / max(1, width / 2)
        centered_y = abs((y + h / 2) - (height / 2)) / max(1, height / 2)
        centered_score = 1.0 - clamp((centered_x + centered_y) / 1.2, 0.0, 1.0)
        candidate = (
            0.45 * clamp((area_ratio - 0.015) / 0.08, 0.0, 1.0)
            + 0.30 * clamp((1.45 - aspect_ratio) / 0.65, 0.0, 1.0)
            + 0.25 * centered_score
        )
        sticker_score = max(sticker_score, candidate)

    qr_score = 1.0 if qr_present else 0.0
    dark_score = clamp((dark_ratio - 0.28) / 0.32, 0.0, 1.0)
    white_score = clamp((white_ratio - 0.03) / 0.12, 0.0, 1.0)
    visual_score = 0.45 * qr_score + 0.30 * sticker_score + 0.15 * dark_score + 0.10 * white_score

    return {
        "visual_score": visual_score,
        "qr_present": qr_present,
        "sticker_score": sticker_score,
        "dark_ratio": dark_ratio,
        "white_ratio": white_ratio,
    }


def refine_prediction(prediction, frame):
    refined = dict(prediction)
    refined["visual_score"] = None
    refined["schokolade_visual_score"] = None

    schokolade_cues = analyze_schokolade_visual_cues(frame)
    refined["schokolade_visual_score"] = schokolade_cues["visual_score"]

    if (
        refined["label"] in {"Jasskarten", "Charger"}
        and prediction["confidence"] < (0.99 if refined["label"] == "Jasskarten" else 0.999)
        and schokolade_cues["qr_present"]
        and schokolade_cues["visual_score"] >= SCHOKOLADE_VISUAL_SCORE_THRESHOLD
    ):
        refined["label"] = "Schokolade"
        refined["confidence"] = max(
            SCHOKOLADE_OVERRIDE_CONFIDENCE,
            refined["runner_up_confidence"] if refined.get("runner_up_label") == "Schokolade" else 0.0,
        )
        refined["runner_up_label"] = prediction["label"]
        refined["runner_up_confidence"] = min(prediction["confidence"], refined["confidence"] - 0.06)
        refined["margin"] = refined["confidence"] - refined["runner_up_confidence"]
        return refined

    if refined["label"] != "Jasskarten":
        return refined

    cues = analyze_jasskarten_visual_cues(frame)
    refined["visual_score"] = cues["visual_score"]

    if refined["confidence"] >= JASSKARTEN_HIGH_CONFIDENCE:
        return refined

    if (
        refined["margin"] < JASSKARTEN_MARGIN_THRESHOLD
        or cues["visual_score"] < JASSKARTEN_VISUAL_SCORE_THRESHOLD
        or cues["rectangle_score"] < 0.20
    ):
        refined["blocked_reason"] = "karten-check"
        return None

    return refined


def fetch_text(url):
    request = urllib.request.Request(url, method="GET")
    contexts = [ssl.create_default_context(), ssl._create_unverified_context()]

    for context in contexts:
        try:
            with urllib.request.urlopen(request, context=context, timeout=30) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except ssl.SSLError:
            continue

    with urllib.request.urlopen(request, context=ssl._create_unverified_context(), timeout=30) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def fetch_dataset():
    status, body = fetch_text(DATASET_URL)
    parsed = json.loads(body)
    items = parsed.get("value", [])
    return status, {item.get("ItemCode"): item for item in items if item.get("ItemCode")}


def load_mapping():
    if not MAPPING_PATH.exists():
        return {}
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def build_hook_result(item_code, status, body, insecure_fallback=False):
    stripped = body.strip()
    preview = stripped.replace("\n", " ")[:140] if stripped else "(leere Antwort)"
    exists = bool(stripped) and status == 200
    return {
        "item_code": item_code,
        "status": status,
        "body": body,
        "preview": preview,
        "exists": exists,
        "insecure_fallback": insecure_fallback,
    }


def query_webhook(item_code):
    url = WEBHOOK_BASE_URL + urllib.parse.quote(item_code)
    request = urllib.request.Request(url, method="GET")
    contexts = [ssl.create_default_context(), ssl._create_unverified_context()]

    for index, context in enumerate(contexts):
        try:
            with urllib.request.urlopen(request, context=context, timeout=20) as response:
                body = response.read().decode("utf-8", errors="replace")
                return build_hook_result(item_code, response.status, body, insecure_fallback=(index == 1))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return build_hook_result(item_code, exc.code, body, insecure_fallback=(index == 1))
        except ssl.SSLError:
            continue
        except Exception as exc:
            return {
                "item_code": item_code,
                "status": None,
                "body": str(exc),
                "preview": str(exc),
                "exists": False,
                "insecure_fallback": index == 1,
            }

    return {
        "item_code": item_code,
        "status": None,
        "body": "SSL-Verbindung fehlgeschlagen.",
        "preview": "SSL-Verbindung fehlgeschlagen.",
        "exists": False,
        "insecure_fallback": False,
    }


def summarize_hook_result(result):
    if result["status"] is None:
        return f"Hook Fehler: {result['preview']}"
    if result["status"] != 200:
        return f"Hook HTTP {result['status']}: {result['preview']}"
    if result["exists"]:
        return "Item im Hook gefunden"
    return f"Hook OK, aber nichts gefunden: {result['preview']}"


def load_model():
    model_source = str(CUSTOM_MODEL_PATH) if CUSTOM_MODEL_PATH.exists() else DEFAULT_MODEL_NAME
    using_custom_model = CUSTOM_MODEL_PATH.exists()
    return YOLO(model_source), model_source, using_custom_model


def classify_roi(model, frame):
    views = [
        frame,
        crop_center(frame, 0.84),
        enhance_roi_for_classification(frame),
    ]
    aggregated_scores = None
    names = None
    used_views = 0

    for view in views:
        result = model.predict(
            source=view,
            imgsz=320,
            conf=CONFIDENCE_THRESHOLD,
            device="cpu",
            verbose=False,
        )[0]

        if result.probs is None:
            continue

        scores = [float(score) for score in result.probs.data.tolist()]
        if aggregated_scores is None:
            aggregated_scores = [0.0] * len(scores)

        for index, score in enumerate(scores):
            aggregated_scores[index] += score

        names = result.names
        used_views += 1

    if aggregated_scores is None or used_views == 0 or names is None:
        return None

    averaged_scores = [score / used_views for score in aggregated_scores]
    ranked_indices = sorted(
        range(len(averaged_scores)),
        key=lambda index: averaged_scores[index],
        reverse=True,
    )
    best_index = ranked_indices[0]
    runner_up_index = ranked_indices[1] if len(ranked_indices) > 1 else best_index
    label = str(names[best_index])
    confidence = averaged_scores[best_index]
    runner_up_confidence = averaged_scores[runner_up_index]
    prediction = {
        "label": label,
        "confidence": confidence,
        "runner_up_label": str(names[runner_up_index]),
        "runner_up_confidence": runner_up_confidence,
        "margin": confidence - runner_up_confidence,
    }
    return refine_prediction(prediction, frame)


def try_open_camera():
    for camera_index in (0, 1, 2):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap, camera_index
        cap.release()

    return cv2.VideoCapture(0), 0


def configure_camera(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)


def rect_iou(rect_a, rect_b):
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def expand_rect(rect, frame_shape, scale=0.08):
    x, y, w, h = rect
    height, width = frame_shape[:2]
    pad_x = int(w * scale)
    pad_y = int(h * scale)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    return x1, y1, x2 - x1, y2 - y1


def dedupe_regions(regions):
    kept = []
    for region in sorted(regions, key=lambda item: item["score"], reverse=True):
        if any(rect_iou(region["rect"], other["rect"]) > REGION_IOU_THRESHOLD for other in kept):
            continue
        kept.append(region)
    return kept[:MAX_CANDIDATES]


def region_detail_score(frame, rect):
    x, y, w, h = rect
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_ratio = cv2.countNonZero(edges) / max(1, crop.shape[0] * crop.shape[1])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat_ratio = cv2.countNonZero(cv2.inRange(hsv, (0, 30, 0), (180, 255, 255))) / max(1, crop.shape[0] * crop.shape[1])
    return edge_ratio * 0.65 + sat_ratio * 0.35


def split_wide_regions(regions, frame_shape):
    height, width = frame_shape[:2]
    split_regions = []
    for region in regions:
        x, y, w, h = region["rect"]
        if w > h * 1.10 or w > width * 0.42:
            half_w = max(90, int(w * 0.54))
            left_rect = expand_rect((x, y, half_w, h), frame_shape, 0.03)
            right_rect = expand_rect((x + w - half_w, y, half_w, h), frame_shape, 0.03)
            split_regions.append({"rect": left_rect, "score": region["score"] * 0.92})
            split_regions.append({"rect": right_rect, "score": region["score"] * 0.92})
    return split_regions


def supplemental_grid_regions(frame):
    height, width = frame.shape[:2]
    base_regions = [
        (int(width * 0.03), int(height * 0.12), int(width * 0.42), int(height * 0.72)),
        (int(width * 0.29), int(height * 0.12), int(width * 0.42), int(height * 0.72)),
        (int(width * 0.55), int(height * 0.12), int(width * 0.42), int(height * 0.72)),
    ]

    regions = []
    for rect in base_regions:
        score = region_detail_score(frame, rect)
        if score >= 0.06:
            regions.append({"rect": rect, "score": score})
    return regions


def propose_object_regions(frame):
    enhanced = enhance_roi_for_classification(frame)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 55, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = max(1, frame.shape[0] * frame.shape[1])
    regions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / frame_area
        if not (MIN_OBJECT_AREA_RATIO <= area_ratio <= MAX_OBJECT_AREA_RATIO):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < MIN_OBJECT_SIDE:
            continue

        aspect_ratio = max(w / max(1, h), h / max(1, w))
        if aspect_ratio > MAX_OBJECT_ASPECT:
            continue

        rect = expand_rect((x, y, w, h), frame.shape)
        center_x = x + w / 2
        center_y = y + h / 2
        center_bias = 1.0 - clamp(
            abs(center_x - frame.shape[1] / 2) / max(1, frame.shape[1] / 2)
            + abs(center_y - frame.shape[0] / 2) / max(1, frame.shape[0] / 2),
            0.0,
            1.4,
        ) / 1.4
        regions.append({"rect": rect, "score": area_ratio * 0.7 + center_bias * 0.3})

    regions.extend(split_wide_regions(regions, frame.shape))
    regions.extend(supplemental_grid_regions(frame))
    regions = dedupe_regions(regions)
    if not regions:
        height, width = frame.shape[:2]
        fallback = (int(width * 0.18), int(height * 0.16), int(width * 0.64), int(height * 0.68))
        regions = [{"rect": fallback, "score": 0.0}]
    return regions


def classify_region(model, frame, rect):
    x, y, w, h = rect
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return None

    prediction = classify_roi(model, crop)
    if prediction is None or prediction["confidence"] < CONFIDENCE_THRESHOLD:
        return None

    prediction["box"] = rect
    return prediction


def dedupe_predictions(predictions):
    def contains_most(inner_rect, outer_rect):
        ix, iy, iw, ih = inner_rect
        ox, oy, ow, oh = outer_rect
        inter_x1 = max(ix, ox)
        inter_y1 = max(iy, oy)
        inter_x2 = min(ix + iw, ox + ow)
        inter_y2 = min(iy + ih, oy + oh)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        return intersection / max(1, iw * ih)

    kept = []
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        if any(
            (
                rect_iou(prediction["box"], other["box"]) > PREDICTION_IOU_THRESHOLD
                or contains_most(prediction["box"], other["box"]) > 0.72
            )
            for other in kept
        ):
            continue
        kept.append(prediction)
    return kept


def detect_multiple_items(model, frame):
    predictions = []
    for region in propose_object_regions(frame):
        prediction = classify_region(model, frame, region["rect"])
        if prediction is None:
            continue
        predictions.append(prediction)
    return dedupe_predictions(predictions)


def counts_signature(counts):
    return tuple(sorted((label, count) for label, count in counts.items()))


def required_frames_for_counts(counts):
    if counts.get("Jasskarten", 0) > 0:
        return JASSKARTEN_STABLE_FRAMES
    return STABLE_FRAMES


def yolo_item_scanner():
    model, model_source, using_custom_model = load_model()
    mapping = load_mapping()

    try:
        dataset_status, dataset_items = fetch_dataset()
        print(f"Datensatz geladen: {len(dataset_items)} Artikel (HTTP {dataset_status})")
    except Exception as exc:
        dataset_status = None
        dataset_items = {}
        print(f"Datensatz konnte nicht geladen werden: {exc}")

    cap, camera_index = try_open_camera()
    configure_camera(cap)

    if not cap.isOpened():
        print("Kamera konnte nicht geoeffnet werden. Bitte Berechtigungen und Treiber pruefen.")
        return

    stable_signature = None
    stable_count = 0
    current_predictions = []
    current_counts = Counter()
    confirmed_counts = Counter()
    last_detection_time = 0.0
    last_checked_signature = None
    last_result_time = 0.0
    last_hook_results = {}
    last_items = {}

    print(f"Kamera {camera_index} ist aktiv.")
    print("YOLO-Klassifikationsscanner startet. Druecke q zum Beenden.")
    print(f"Modell: {model_source}")
    if not using_custom_model:
        print("Hinweis: Es wird noch das generische Klassifikationsmodell verwendet. Fuer eure Artikel zuerst train_yolo_local.py laufen lassen.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

    while True:
        success, frame = cap.read()
        if not success:
            print("Kamerabild konnte nicht gelesen werden.")
            break

        now = time.time()
        roi = build_focus_roi(frame)
        draw_focus_box(frame, roi)

        if (now - last_detection_time) >= DETECTION_INTERVAL_SECONDS:
            last_detection_time = now
            x1, y1, x2, y2 = roi
            roi_frame = frame[y1:y2, x1:x2]
            predictions = detect_multiple_items(model, roi_frame)
            current_predictions = predictions
            current_counts = Counter(prediction["label"] for prediction in predictions)
            signature = counts_signature(current_counts)

            if not signature:
                current_counts = Counter()
                stable_signature = None
                stable_count = 0
            else:
                if signature == stable_signature:
                    stable_count += 1
                else:
                    stable_signature = signature
                    stable_count = 1

                required_frames = required_frames_for_counts(current_counts)
                if stable_count >= required_frames:
                    confirmed_counts = Counter(current_counts)
                    if signature != last_checked_signature:
                        last_checked_signature = signature
                        last_result_time = now
                        last_hook_results = {}
                        last_items = {}
                        print("Erkannt:", ", ".join(f"{label} x{count}" for label, count in signature))

                        for label, count in signature:
                            mapping_entry = mapping.get(label, {})
                            item_code = mapping_entry.get("item_code")
                            if not item_code:
                                continue

                            hook_result = query_webhook(item_code)
                            item = dataset_items.get(item_code)
                            last_hook_results[label] = hook_result
                            last_items[label] = item
                            print(f"{label} x{count} -> ItemCode={item_code}")
                            if item:
                                print(f"Artikel: {item.get('ItemName')} ({item_code})")
                            print(summarize_hook_result(hook_result))

        trusted_prediction = len(current_predictions) > 0
        roi_color = COLOR_ROI_ACTIVE if trusted_prediction else COLOR_ROI_IDLE
        draw_focus_frame(frame, roi, roi_color)
        draw_focus_hint(frame, roi, trusted_prediction)

        for prediction in current_predictions:
            box_x, box_y, box_w, box_h = prediction["box"]
            abs_x = box_x + roi[0]
            abs_y = box_y + roi[1]
            cv2.rectangle(frame, (abs_x, abs_y), (abs_x + box_w, abs_y + box_h), COLOR_SUCCESS, 2)
            label_text = f"{prediction['label']} {prediction['confidence']:.2f}"
            draw_text(frame, label_text, (abs_x + 6, max(24, abs_y - 10)), COLOR_SUCCESS, scale=0.5, thickness=1)

        current_required_frames = required_frames_for_counts(current_counts) if current_predictions else STABLE_FRAMES

        if confirmed_counts and stable_count >= current_required_frames:
            status_text = "Stabil erkannt"
            status_color = COLOR_SUCCESS
        elif current_predictions:
            status_text = "Noch unsicher"
            status_color = COLOR_WARNING
        else:
            status_text = "Bereit"
            status_color = COLOR_ROI_IDLE

        header_lines = [
            "Mehrere Artikel in den Rahmen legen",
            "Q beendet den Scanner",
            f"Datensatz: {len(dataset_items)} Artikel" if dataset_status is not None else "Datensatz nicht verfuegbar",
            "Custom-Modell aktiv" if using_custom_model else "Generisches Modell aktiv",
        ]
        draw_info_card(frame, "YOLO Item Scanner", header_lines, (18, 18), 300)
        draw_status_chip(frame, status_text, (frame.shape[1] - 205, 18), status_color)

        prediction_lines = []
        if not current_predictions:
            prediction_lines.append("Noch keine sichere Vorhersage")
            prediction_lines.append("Artikel ruhig im Rahmen halten")
        else:
            for label, count in current_counts.most_common(4):
                prediction_lines.append(f"{label} x{count}")
            if len(current_predictions) > 4:
                prediction_lines.append(f"Weitere Treffer: {len(current_predictions) - 4}")
            prediction_lines.append(f"Stabilitaet: {stable_count}/{current_required_frames}")
        prediction_card_height = 54 + len(prediction_lines) * 24
        prediction_y = max(18, frame.shape[0] - prediction_card_height - 18)
        draw_info_card(frame, "Erkennung", prediction_lines, (18, prediction_y), 320)

        item_lines = []
        if confirmed_counts:
            for label, count in confirmed_counts.most_common(4):
                mapping_entry = mapping.get(label)
                if mapping_entry and mapping_entry.get("item_code"):
                    item_lines.append(f"{label} x{count} -> {mapping_entry['item_code']}")
                    if label in last_items and last_items[label]:
                        item_lines.append(f"{last_items[label].get('ItemName')}")
                elif mapping_entry:
                    item_lines.append(f"{label} x{count} -> ERP-Zuordnung fehlt")
                else:
                    item_lines.append(f"{label} x{count} -> kein Mapping")
        else:
            item_lines.append("Noch kein bestaetigter Scan")

        if last_hook_results and (now - last_result_time) <= RESULT_HOLD_SECONDS:
            for label, hook_result in list(last_hook_results.items())[:3]:
                item_lines.append(f"{label}: {summarize_hook_result(hook_result)}")

        item_card_height = 54 + len(item_lines) * 24
        item_y = max(18, frame.shape[0] - item_card_height - 18)
        draw_info_card(frame, "Datenbank", item_lines, (frame.shape[1] - 338, item_y), 320)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_item_scanner()
