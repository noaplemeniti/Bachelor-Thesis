import cv2
import numpy as np
import random

def get_nonpadding_bbox(image, padding_color=(127, 127, 127), tolerance=0):
        padding_color = np.array(padding_color, dtype=np.uint8)

        lower = np.clip(padding_color - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(padding_color + tolerance, 0, 255).astype(np.uint8)

        pad_mask = cv2.inRange(image, lower, upper)
        content_mask = cv2.bitwise_not(pad_mask)

        coords = cv2.findNonZero(content_mask)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        return x, y, w, h

def brush_strokes(
        image,
        stroke_length=random.randint(20, 50),
        stroke_width=random.randint(8, 32),
        stroke_count=random.randint(3, 15),
        stroke_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        padding_color=(127, 127, 127),
        tolerance=0,
        inner_margin=3,
    ):
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        bbox = get_nonpadding_bbox(
            image,
            padding_color=padding_color,
            tolerance=tolerance,
        )
        if bbox is None:
            return None, None

        x, y, bw, bh = bbox

        x_min = x + inner_margin
        y_min = y + inner_margin
        x_max = x + bw - inner_margin - 1
        y_max = y + bh - inner_margin - 1

        if x_min > x_max or y_min > y_max:
            return None, None

        for _ in range(stroke_count):
            x_start = np.random.randint(x_min, x_max + 1)
            y_start = np.random.randint(y_min, y_max + 1)

            num_segments = np.random.randint(2, 7)
            points = [(x_start, y_start)]

            angle = np.random.uniform(0, 2 * np.pi)
            curr_x, curr_y = x_start, y_start

            for _ in range(num_segments):
                angle += np.random.uniform(-np.pi / 4, np.pi / 4)
                length = np.random.uniform(1, stroke_length)

                curr_x = int(curr_x + length * np.cos(angle))
                curr_y = int(curr_y + length * np.sin(angle))

                curr_x = np.clip(curr_x, x_min, x_max)
                curr_y = np.clip(curr_y, y_min, y_max)

                points.append((curr_x, curr_y))

            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], False, color=255, thickness=stroke_width)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        safe_region = np.zeros((h, w), dtype=np.uint8)
        safe_region[y:y + bh, x:x + bw] = 255
        mask = cv2.bitwise_and(mask, safe_region)

        masked_image = image.copy()
        masked_image[mask == 255] = stroke_color

        return masked_image, mask
