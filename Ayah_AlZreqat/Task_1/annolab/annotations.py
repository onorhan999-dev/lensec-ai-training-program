
import cv2


# =====================================================
# TASK 1 - Part A
# Helper Function: Draw Bounding Box
#
# تستخدم في:
# Draw a box per object
#
# ترسم مستطيلاً حول الشخص باستخدام:
# x, y = موقع الصندوق
# w, h = العرض والارتفاع
# =====================================================

def draw_bbox(image, x, y, w, h, color=(0, 255, 0), thickness=2):

    x1 = int(x)
    y1 = int(y)

    x2 = int(x + w)
    y2 = int(y + h)

    # رسم المستطيل على الصورة
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    return image


# =====================================================
# TASK 1 - Helper: Calculate IoU
#
# تحسب Intersection over Union
# أي نسبة التداخل بين صندوقين.
#
# ليست جزءاً مباشراً من متطلبات
# Draw Annotations on the Video في Part A.
# =====================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# =====================================================
# TASK 1 - Part A - Requirement 1
#
# Read the MOT17 gt.txt file.
#
# المطلوب:
# قراءة ملف gt.txt واستخراج الأعمدة التسعة:
# frame, track id, left, top,
# width, height, flag, class, visibility
#
# INPUT: gt.txt
# OUTPUT: annotations Dictionary
# =====================================================

def load_mot_annotations(file_path):

    # إنشاء Dictionary لتخزين البيانات
    annotations = {}

    # فتح ملف gt.txt
    with open(file_path, "r") as file:

        # قراءة كل سطر
        for line in file:

            # تقسيم السطر عند الفواصل
            values = line.strip().split(",")

            # رقم الإطار
            frame_id = int(values[0])

            # رقم الشخص
            track_id = int(values[1])

            # إحداثيات وأبعاد Bounding Box
            x = float(values[2])
            y = float(values[3])
            w = float(values[4])
            h = float(values[5])

            # قراءة Flag وClass وVisibility
            flag = int(values[6])
            class_id = int(values[7])
            visibility = float(values[8])


            # =========================================
            # TASK 1 - Part A - Requirement 2
            #
            # Keep only rows where:
            # flag = 1 AND class = 1
            #
            # المطلوب:
            # الاحتفاظ فقط بالأشخاص المطابقين
            # للشرطين واستبعاد باقي الصفوف.
            # =========================================

            if flag != 1 or class_id != 1:
                continue


            # =========================================
            # TASK 1 - Part A - Requirement 1
            #
            # تخزين الأشخاص حسب رقم الإطار.
            # =========================================

            if frame_id not in annotations:
                annotations[frame_id] = []

            annotations[frame_id].append({
                "track_id": track_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

    # إرجاع البيانات بعد قراءتها وفلترتها
    return annotations


# =====================================================
# TASK 1 - Part A - Requirement 3
#
# Draw a box per object with a stable color
# for each track id, and the label ID 12.
#
# المطلوب:
# رسم مربع حول كل شخص.
# إعطاء لون ثابت لكل Track ID.
# كتابة رقم الشخص فوق المربع.
#
# TASK 1 - Part A - Requirement 4
#
# Draw a short trail of the last 30 positions.
#
# المطلوب:
# رسم مسار حركة الشخص باستخدام آخر 30 موقع.
# =====================================================

def draw_mot_annotations(image, objects, track_history=None):

    # Dictionary لتخزين مسارات الأشخاص
    if track_history is None:
        track_history = {}

    # المرور على كل شخص في الإطار الحالي
    for obj in objects:

        track_id = obj["track_id"]

        x = int(obj["x"])
        y = int(obj["y"])
        w = int(obj["w"])
        h = int(obj["h"])


        # =========================================
        # Requirement 3: Stable Color
        #
        # توليد لون ثابت بناءً على Track ID.
        # نفس الشخص يحصل على نفس اللون.
        # =========================================

        color = (
            (track_id * 37) % 256,
            (track_id * 67) % 256,
            (track_id * 97) % 256
        )


        # =========================================
        # Requirement 3: Draw Bounding Box
        #
        # استدعاء دالة draw_bbox
        # لرسم المربع حول الشخص.
        # =========================================

        draw_bbox(image, x, y, w, h, color=color)


        # =========================================
        # Requirement 3: Draw Track ID
        #
        # كتابة ID فوق الشخص.
        # مثال: ID 12
        # =========================================

        cv2.putText(
            image,
            f"ID {track_id}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )


        # =========================================
        # Requirement 4: Calculate Position
        #
        # تحديد نقطة أسفل منتصف المربع
        # لاستخدامها في رسم مسار الحركة.
        # =========================================

        center_x = x + w // 2
        center_y = y + h


        # =========================================
        # Requirement 4: Track History
        #
        # إنشاء قائمة للمواقع إذا كان
        # هذا الشخص يظهر لأول مرة.
        # =========================================

        if track_id not in track_history:
            track_history[track_id] = []

        # إضافة الموقع الحالي للشخص
        track_history[track_id].append(
            (center_x, center_y)
        )


        # =========================================
        # Requirement 4: Last 30 Positions
        #
        # الاحتفاظ بآخر 30 موقع فقط.
        # =========================================

        track_history[track_id] = track_history[track_id][-30:]


        # =========================================
        # Requirement 4: Draw Movement Trail
        #
        # توصيل المواقع السابقة بخطوط
        # لرسم مسار حركة الشخص.
        # =========================================

        points = track_history[track_id]

        for i in range(1, len(points)):
            cv2.line(
                image,
                points[i - 1],
                points[i],
                color,
                2
            )

    return image
