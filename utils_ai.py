import os
import cv2
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import sqlite3
from reportlab.pdfgen import canvas
from openpyxl import Workbook

model = YOLO("yolov8x-oiv7.pt")

def is_furniture(cls_name):
    return cls_name.lower() in {"chair", "dining table", "table"}

def is_point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def annotate_frame(frame, text, position, color=(255, 0, 0)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def create_output_dir(base="static"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(base, now)
    os.makedirs(path, exist_ok=True)
    return path

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")

    results = model.track(img, persist=True, conf=0.3, tracker="bytetrack.yaml")

    if not results or results[0].boxes is None:
        raise ValueError("Объекты не найдены")

    people = []
    furniture = []
    sitting_ids = set()

    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if cls_name.lower() == "person" and box.id is not None:
            person_id = int(box.id[0])
            bottom_center = ((x1 + x2) // 2, y2)
            people.append((person_id, bottom_center, (x1, y1)))
        elif is_furniture(cls_name):
            furniture.append([x1, y1, x2, y2])

    annotated = img.copy()

    for person_id, point, (x, y) in people:
        for furn in furniture:
            if is_point_inside_bbox(point, furn):
                sitting_ids.add(person_id)
                x1, y1 = x, y
                x2, y2 = point[0] * 2 - x1, point[1]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                annotate_frame(annotated, f"id:{person_id} [person]", (x1, y1 - 10), (0, 255, 0))
                break

    output_dir = create_output_dir()
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, annotated)

    return {"unique_people_sitting": len(sitting_ids), "output_path": output_path}

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Ошибка чтения видео")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = create_output_dir()
    output_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"result_{output_filename}")


    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    time_on_furniture = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.3, tracker="bytetrack.yaml")

        if not results or results[0].boxes is None:
            out.write(frame)
            continue

        people = []
        furniture = []

        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls_name.lower() == "person" and box.id is not None:
                person_id = int(box.id[0])
                bottom_center = ((x1 + x2) // 2, y2)
                people.append((person_id, bottom_center, (x1, y1)))
            elif is_furniture(cls_name):
                furniture.append([x1, y1, x2, y2])

        sitting_now = set()
        for person_id, point, (x, y) in people:
            for furn in furniture:
                if is_point_inside_bbox(point, furn):
                    time_on_furniture[person_id] += 1
                    sitting_now.add(person_id)
                    break

        annotated = frame.copy()
        for person_id, point, (x, y) in people:
            if person_id in time_on_furniture:
                x1, y1 = x, y
                x2, y2 = point[0] * 2 - x1, point[1]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                annotate_frame(annotated, f"id:{person_id} [person]", (x1, y1 - 10), (0, 255, 0))

        out.write(annotated)

    cap.release()
    out.release()

    threshold_frames = int(fps * 2)
    sitting_people = {pid for pid, count in time_on_furniture.items() if count >= threshold_frames}

    return {"unique_people_sitting": len(sitting_people), "output_path": output_path}

def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            type TEXT,
            count INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_to_history(filename, filetype, count):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO requests (timestamp, filename, type, count)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().isoformat(), filename, filetype, count))
    conn.commit()
    conn.close()

def generate_report(report_type, result_data, report_dir="reports"):
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_type}"
    report_path = os.path.join(report_dir, filename)

    if report_type == "pdf":
        c = canvas.Canvas(report_path)
        c.setFont("Helvetica", 14)
        c.drawString(50, 800, "Analysis Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 770, f"File: {result_data['filename']}")
        c.drawString(50, 750, f"People sitting: {result_data['count']}")
        c.drawString(50, 730, f"Generated at: {timestamp}")
        c.save()
    elif report_type == "xlsx":
        wb = Workbook()
        ws = wb.active
        ws.title = "Report"
        ws.append(["File", "People Sitting", "Timestamp"])
        ws.append([result_data["filename"], result_data["count"], timestamp])
        wb.save(report_path)

    return report_path

def export_history_to_xlsx(output_path="reports/history_export.xlsx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, filename, type, count FROM requests")
    rows = cursor.fetchall()
    conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "History"

    ws.append(["Timestamp", "Filename", "Type", "Count"])
    for row in rows:
        ws.append(row)

    wb.save(output_path)
    return output_path