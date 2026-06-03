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
        stroke_length=None,
        stroke_width=None,
        stroke_count=None,
        random_color_flag=True,
        mask_color=(0, 0, 0),
        padding_color=(127, 127, 127),
        tolerance=0,
        inner_margin=3,
    ):
    # Fix python mutable default arguments evaluation bug
    stroke_length = stroke_length if stroke_length is not None else random.randint(20, 50)
    stroke_width = stroke_width if stroke_width is not None else random.randint(8, 32)
    stroke_count = stroke_count if stroke_count is not None else random.randint(3, 15)

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

    # Generate the geometry shape using your exact loop logic
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

    # --- ADVANCED CORRUPTION LOGIC LIVES IN THE BACKFILL ENGINE ---
    masked_image = image.copy()
    
    if not random_color_flag:
        # Fallback behavior baseline
        masked_image[mask == 255] = mask_color
    else:
        strategy_roll = random.random()
        
        if strategy_roll < 0.40:
            # Strategy A: Random Solid Rainbow Color
            random_solid = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            masked_image[mask == 255] = random_solid
            
        elif strategy_roll < 0.80:
            # Strategy B: Per-pixel Digital Noise (TV Static)
            noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            masked_image = np.where(mask[..., None] == 255, noise, masked_image)
            
        else:
            # Strategy C: Chameleon Color (Match plate average background context)
            plate_crop = image[y_min:y_max, x_min:x_max]
            mean_color = np.mean(plate_crop, axis=(0, 1)).astype(int) if plate_crop.size > 0 else [127, 127, 127]
            jitter = random.randint(-25, 25)
            chameleon_color = tuple(np.clip(mean_color + jitter, 0, 255).astype(int))
            masked_image[mask == 255] = chameleon_color

    # Soft edge alpha blending to prevent the UNet edge-detection shortcut
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 0) / 255.0
    mask_blur_3d = np.expand_dims(mask_blur, axis=2)
    
    final_masked_image = (image * (1.0 - mask_blur_3d) + masked_image * mask_blur_3d).astype(np.uint8)

    return final_masked_image, mask