import sys
import time

try:
    import cv2
except ImportError:
    print("OpenCV fehlt. Installiere es mit: pip install opencv-python")
    sys.exit(1)

try:
    from pyzbar.pyzbar import decode as decode_barcodes
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
except Exception:
    decode_barcodes = None
    ZBarSymbol = None
    PYZBAR_AVAILABLE = False


WINDOW_NAME = "Barcode Scanner"
BARCODE_DETECTOR_AVAILABLE = hasattr(cv2, "barcode_BarcodeDetector")
RESULT_HOLD_SECONDS = 6.0
MIN_CONFIRMATION_FRAMES = 2
TRACK_EXPIRY_SECONDS = 2.5
SCAN_COOLDOWN_SECONDS = 1.5
DETECTION_INTERVAL_SECONDS = 0.12
ROI_SCALE = 1.6
ACTIVE_LOCK_SECONDS = 1.2
ALLOWED_BARCODE_TYPES = {"EAN_13", "EAN13", "EAN_8", "EAN8", "UPC_A", "UPCA", "UPC_E", "UPCE", "QR"}
SUPPORTED_BARCODE_SYMBOLS = [
    ZBarSymbol.EAN13,
    ZBarSymbol.EAN8,
    ZBarSymbol.UPCA,
    ZBarSymbol.UPCE,
] if PYZBAR_AVAILABLE else []


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    sharpened = cv2.addWeighted(equalized, 1.6, blurred, -0.6, 0)
    return gray, sharpened


def draw_text(frame, text, position, color):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_banner(frame, text):
    cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 60), (30, 30, 30), -1)
    cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 60), (255, 255, 255), 1)
    draw_text(frame, text, (20, 42), (255, 255, 255))


def clean_code_text(text):
    return "".join(character for character in text.strip() if character.isprintable())


def normalize_code_type(code_type):
    return (code_type or "BARCODE").strip().upper()


def configure_barcode_detector(detector):
    if detector is None:
        return None

    detector.setDownsamplingThreshold(1024.0)
    detector.setGradientThreshold(48.0)
    detector.setDetectorScales([0.01, 0.02, 0.04, 0.08])
    return detector


def is_valid_ean_checksum(code):
    digits = [int(character) for character in code]
    checksum = digits.pop()
    digits.reverse()
    weighted_sum = 0
    for index, digit in enumerate(digits):
        weighted_sum += digit * (3 if index % 2 == 0 else 1)
    calculated = (10 - (weighted_sum % 10)) % 10
    return checksum == calculated


def is_plausible_code(text, code_type):
    normalized_type = normalize_code_type(code_type)
    if not text or len(text) < 4:
        return False

    if normalized_type in {"EAN_13", "EAN13"}:
        return text.isdigit() and len(text) == 13 and is_valid_ean_checksum(text)
    if normalized_type in {"EAN_8", "EAN8"}:
        return text.isdigit() and len(text) == 8 and is_valid_ean_checksum(text)
    if normalized_type in {"UPC_A", "UPCA"}:
        return text.isdigit() and len(text) == 12 and is_valid_ean_checksum("0" + text)
    if normalized_type in {"UPC_E", "UPCE"}:
        return text.isdigit() and len(text) in {6, 7, 8}
    if normalized_type == "QR":
        return len(text) >= 3
    return False


def normalize_points(points):
    if points is None or len(points) == 0:
        return []

    normalized = []
    for polygon in points:
        normalized.append([tuple(map(int, point)) for point in polygon])
    return normalized


def build_focus_roi(frame):
    height, width = frame.shape[:2]
    roi_width = int(width * 0.7)
    roi_height = int(height * 0.45)
    x1 = max(0, (width - roi_width) // 2)
    y1 = max(0, (height - roi_height) // 2)
    x2 = x1 + roi_width
    y2 = y1 + roi_height
    return x1, y1, x2, y2


def crop_with_offset(image, roi):
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2], (x1, y1)


def offset_polygon(polygon, offset):
    offset_x, offset_y = offset
    return [(x + offset_x, y + offset_y) for x, y in polygon]


def detect_with_opencv(detector, images):
    if detector is None:
        return []

    for image in images:
        found, decoded_info, decoded_types, points = detector.detectAndDecodeWithType(image)
        if not found or points is None or len(points) == 0:
            continue

        polygons = normalize_points(points)
        results = []
        for index, polygon in enumerate(polygons):
            text = ""
            code_type = "BARCODE"

            if index < len(decoded_info) and decoded_info[index]:
                text = decoded_info[index].strip()
            if index < len(decoded_types) and decoded_types[index]:
                code_type = decoded_types[index]

            normalized_type = normalize_code_type(code_type)
            if text and polygon and normalized_type in ALLOWED_BARCODE_TYPES:
                results.append({"text": text, "type": normalized_type, "polygon": polygon})

        if results:
            return results

    return []


def detect_with_pyzbar(images):
    if not PYZBAR_AVAILABLE:
        return []

    for image in images:
        results = decode_barcodes(image, symbols=SUPPORTED_BARCODE_SYMBOLS)
        if not results:
            continue

        converted = []
        for barcode in results:
            text = barcode.data.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            normalized_type = normalize_code_type(barcode.type)
            if normalized_type not in ALLOWED_BARCODE_TYPES:
                continue

            x, y, w, h = barcode.rect
            polygon = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
            ]
            converted.append({"text": text, "type": normalized_type, "polygon": polygon})

        if converted:
            return converted

    return []


def draw_polygon(frame, polygon, color):
    if not polygon:
        return

    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        cv2.line(frame, start, end, color, 2)


def polygon_area(polygon):
    if not polygon or len(polygon) < 3:
        return 0.0

    total = 0.0
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygon_center(polygon):
    if not polygon:
        return None

    x = sum(point[0] for point in polygon) / len(polygon)
    y = sum(point[1] for point in polygon) / len(polygon)
    return x, y


def required_confirmation_frames(detection):
    normalized_type = normalize_code_type(detection["type"])
    text = detection["text"]
    area = polygon_area(detection["polygon"])

    if normalized_type in {"EAN_13", "EAN13"} and text.isdigit() and len(text) == 13 and is_valid_ean_checksum(text):
        return 1
    if normalized_type in {"EAN_8", "EAN8"} and text.isdigit() and len(text) == 8 and is_valid_ean_checksum(text):
        return 1
    if normalized_type in {"UPC_A", "UPCA"} and text.isdigit() and len(text) == 12 and is_valid_ean_checksum("0" + text):
        return 1
    if area >= 25000:
        return 1
    return MIN_CONFIRMATION_FRAMES


def draw_focus_box(frame, roi):
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)


def deduplicate_results(results):
    unique = {}
    for result in results:
        key = (result["type"], result["text"])
        unique[key] = result
    return list(unique.values())


def pick_best_detection(detections, frame_shape, roi):
    if not detections:
        return None

    frame_height, frame_width = frame_shape[:2]
    frame_center = (frame_width / 2.0, frame_height / 2.0)
    x1, y1, x2, y2 = roi

    def detection_score(detection):
        polygon = detection["polygon"]
        center = polygon_center(polygon)
        area = polygon_area(polygon)

        if center is None:
            return (-1, 0.0, float("-inf"))

        inside_roi = x1 <= center[0] <= x2 and y1 <= center[1] <= y2
        distance = ((center[0] - frame_center[0]) ** 2 + (center[1] - frame_center[1]) ** 2) ** 0.5
        return (1 if inside_roi else 0, area, -distance)

    return max(detections, key=detection_score)


def cleanup_candidate_history(candidate_history, now):
    expired_keys = [
        key for key, value in candidate_history.items()
        if now - value["last_seen"] > TRACK_EXPIRY_SECONDS
    ]
    for key in expired_keys:
        del candidate_history[key]


def discard_candidate(candidate_history, detection):
    if detection is None:
        return

    key = (normalize_code_type(detection["type"]), clean_code_text(detection["text"]))
    if key in candidate_history:
        del candidate_history[key]


def confirm_detection(candidate_history, detection, now):
    if detection is None:
        cleanup_candidate_history(candidate_history, now)
        return None

    text = clean_code_text(detection["text"])
    code_type = normalize_code_type(detection["type"])

    if not is_plausible_code(text, code_type):
        cleanup_candidate_history(candidate_history, now)
        return None

    key = (code_type, text)
    current = candidate_history.get(key)
    if current is None:
        current = {
            "count": 1,
            "last_seen": now,
            "polygon": detection["polygon"],
        }
        candidate_history[key] = current
        cleanup_candidate_history(candidate_history, now)
        if current["count"] >= required_confirmation_frames({
            "text": text,
            "type": code_type,
            "polygon": current["polygon"],
        }):
            return {
                "text": text,
                "type": code_type,
                "polygon": current["polygon"],
            }
        return None

    current["count"] += 1
    current["last_seen"] = now
    current["polygon"] = detection["polygon"]
    cleanup_candidate_history(candidate_history, now)

    if current["count"] >= required_confirmation_frames({
        "text": text,
        "type": code_type,
        "polygon": current["polygon"],
    }):
        return {
            "text": text,
            "type": code_type,
            "polygon": current["polygon"],
        }

    return None


def detect_barcodes(frame, detector):
    gray, sharpened = preprocess_frame(frame)
    roi = build_focus_roi(frame)

    roi_frame, offset = crop_with_offset(frame, roi)
    roi_gray, _ = crop_with_offset(gray, roi)
    roi_sharpened, _ = crop_with_offset(sharpened, roi)
    zoomed_roi = cv2.resize(roi_sharpened, None, fx=ROI_SCALE, fy=ROI_SCALE, interpolation=cv2.INTER_CUBIC)

    search_images = (
        (roi_frame, offset),
        (roi_gray, offset),
        (roi_sharpened, offset),
        (frame, (0, 0)),
        (gray, (0, 0)),
    )

    results = []
    if detector is not None:
        opencv_results = []
        for image, current_offset in search_images:
            detected = detect_with_opencv(detector, (image,))
            for result in detected:
                opencv_results.append({
                    "text": result["text"],
                    "type": result["type"],
                    "polygon": offset_polygon(result["polygon"], current_offset),
                })

        zoomed_results = detect_with_opencv(detector, (zoomed_roi,))
        for result in zoomed_results:
            scaled_polygon = [(int(x / ROI_SCALE), int(y / ROI_SCALE)) for x, y in result["polygon"]]
            opencv_results.append({
                "text": result["text"],
                "type": result["type"],
                "polygon": offset_polygon(scaled_polygon, offset),
            })

        results.extend(opencv_results)

    if not results:
        pyzbar_results = []
        for image, current_offset in search_images:
            detected = detect_with_pyzbar((image,))
            for result in detected:
                pyzbar_results.append({
                    "text": result["text"],
                    "type": result["type"],
                    "polygon": offset_polygon(result["polygon"], current_offset),
                })
        results.extend(pyzbar_results)

    return deduplicate_results(results), roi, gray


def try_open_camera():
    for camera_index in (0, 1, 2):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap, camera_index
        cap.release()

    return cv2.VideoCapture(0), 0


def camera_barcode_scanner():
    cap, camera_index = try_open_camera()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    if not cap.isOpened():
        print("Kamera konnte nicht geoeffnet werden. Bitte Berechtigungen und Treiber pruefen.")
        return

    barcode_detector = configure_barcode_detector(cv2.barcode_BarcodeDetector() if BARCODE_DETECTOR_AVAILABLE else None)
    qr_detector = cv2.QRCodeDetector()
    scanned_codes = set()
    candidate_history = {}
    active_detection = None
    last_scanned_key = None
    last_scanned_time = 0.0
    last_result = None
    last_result_time = 0.0
    last_detection_time = 0.0

    print(f"Kamera {camera_index} ist aktiv.")
    print("Scanner startet. Druecke q zum Beenden.")
    print("Tipp: Barcode gerade halten, nah an die Kamera bringen und gut ausleuchten.")
    print("Tipp: Den Barcode zuerst mittig und ruhig halten, dann 1 bis 2 Sekunden warten.")
    if not BARCODE_DETECTOR_AVAILABLE:
        print("OpenCV hat keinen eingebauten Barcode-Detector. Nur QR-Codes sind aktiv.")
    elif not PYZBAR_AVAILABLE:
        print("OpenCV-Barcode-Detector ist aktiv. pyzbar-Fallback ist auf diesem Rechner nicht verfuegbar.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Kamerabild konnte nicht gelesen werden.")
            break

        now = time.time()
        roi = build_focus_roi(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stable_barcode = None
        draw_focus_box(frame, roi)

        if (now - last_detection_time) >= DETECTION_INTERVAL_SECONDS:
            last_detection_time = now
            barcode_results, roi, gray = detect_barcodes(frame, barcode_detector)
            best_barcode = pick_best_detection(barcode_results, frame.shape, roi)
            stable_barcode = confirm_detection(candidate_history, best_barcode, now)

            if stable_barcode:
                raw_text = stable_barcode["text"]
                barcode_type = stable_barcode["type"]
                polygon = stable_barcode["polygon"]
                barcode_key = (barcode_type, raw_text)

                if active_detection and (now - active_detection["last_seen"]) <= ACTIVE_LOCK_SECONDS:
                    if barcode_key != active_detection["key"]:
                        discard_candidate(candidate_history, stable_barcode)
                        stable_barcode = None
                    else:
                        active_detection["last_seen"] = now
                        active_detection["polygon"] = polygon
                elif stable_barcode:
                    active_detection = {
                        "key": barcode_key,
                        "text": raw_text,
                        "type": barcode_type,
                        "polygon": polygon,
                        "last_seen": now,
                    }

                if stable_barcode:
                    if barcode_key != last_scanned_key or (now - last_scanned_time) > SCAN_COOLDOWN_SECONDS:
                        if raw_text not in scanned_codes:
                            scanned_codes.add(raw_text)
                        print(f"Barcode erkannt [{barcode_type}]: {raw_text}")
                        last_scanned_key = barcode_key
                        last_scanned_time = now

                    last_result = f"{barcode_type}: {raw_text}"
                    last_result_time = now

        if active_detection and (now - active_detection["last_seen"]) > ACTIVE_LOCK_SECONDS:
            active_detection = None

        if active_detection:
            polygon = active_detection["polygon"]
            draw_polygon(frame, polygon, (0, 180, 255))
            label_position = polygon[0] if polygon else (30, 80)
            draw_text(
                frame,
                f"{active_detection['type']}: {active_detection['text']}",
                (label_position[0], max(30, label_position[1] - 10)),
                (0, 180, 255),
            )

        qr_text, qr_points, _ = qr_detector.detectAndDecode(gray)
        if qr_text:
            qr_polygon = [tuple(map(int, point)) for point in qr_points[0]] if qr_points is not None else []
            stable_qr = confirm_detection(
                candidate_history,
                {"text": qr_text, "type": "QR", "polygon": qr_polygon},
                now,
            )
            if stable_qr:
                qr_key = ("QR", stable_qr["text"])
                if active_detection and (now - active_detection["last_seen"]) <= ACTIVE_LOCK_SECONDS and qr_key != active_detection["key"]:
                    discard_candidate(candidate_history, stable_qr)
                else:
                    active_detection = {
                        "key": qr_key,
                        "text": stable_qr["text"],
                        "type": "QR",
                        "polygon": qr_polygon,
                        "last_seen": now,
                    }

                    if qr_key != last_scanned_key or (now - last_scanned_time) > SCAN_COOLDOWN_SECONDS:
                        if stable_qr["text"] not in scanned_codes:
                            scanned_codes.add(stable_qr["text"])
                        print(f"QR-Code erkannt: {stable_qr['text']}")
                        last_scanned_key = qr_key
                        last_scanned_time = now

                    last_result = f"QR: {stable_qr['text']}"
                    last_result_time = now

        if last_result and (time.time() - last_result_time) <= RESULT_HOLD_SECONDS:
            draw_banner(frame, f"Letzter Treffer: {last_result}")

        status_parts = []
        if BARCODE_DETECTOR_AVAILABLE:
            status_parts.append("opencv-barcode aktiv")
        if PYZBAR_AVAILABLE:
            status_parts.append("pyzbar fallback aktiv")
        if not status_parts:
            status_parts.append("nur QR-Modus")
        status_parts.append(f"bestaetigt nach {MIN_CONFIRMATION_FRAMES} Frames")

        draw_text(frame, f"Status: {', '.join(status_parts)}", (20, frame.shape[0] - 20), (255, 255, 0))
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_barcode_scanner()
