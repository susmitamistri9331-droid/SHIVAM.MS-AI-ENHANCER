"""Classical finishing effects; these are image-processing operations, not trained AI models."""
import cv2
import numpy as np

EFFECT_PRESETS = [
    "Clean Natural", "Crisp Detail", "Strong Sharpen", "Soft Denoise", "Anime Crisp",
    "Texture Pop", "Cinematic Contrast", "Bright Clean", "Glow Highlights", "Film Grain",
    "Warm Color", "Cool Color", "Vivid Color", "Matte Film", "Edge Clarity", "Low Noise",
]

def apply_effects(frame, denoise=2, sharpen=30, structure=15, clarity=0, glow=0,
                 saturation=100, contrast=100, brightness=0, grain=0, preset="Clean Natural"):
    # Work in uint8 BGR. The preset adjusts the base values but user controls remain adjustable.
    p = preset or "Clean Natural"
    if p == "Crisp Detail": sharpen = max(sharpen, 45); structure = max(structure, 25)
    elif p == "Strong Sharpen": sharpen = max(sharpen, 70); structure = max(structure, 35)
    elif p in ("Soft Denoise", "Low Noise"): denoise = max(denoise, 5)
    elif p == "Anime Crisp": sharpen = max(sharpen, 45); saturation = max(saturation, 105)
    elif p == "Texture Pop": structure = max(structure, 40); clarity = max(clarity, 20)
    elif p == "Cinematic Contrast": contrast = max(contrast, 112); saturation = min(saturation, 95)
    elif p == "Bright Clean": brightness = max(brightness, 8)
    elif p == "Glow Highlights": glow = max(glow, 30)
    elif p == "Film Grain": grain = max(grain, 5)
    elif p == "Warm Color":
        frame = frame.astype(np.float32); frame[:, :, 2] *= 1.04; frame[:, :, 0] *= 0.96
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif p == "Cool Color":
        frame = frame.astype(np.float32); frame[:, :, 0] *= 1.04; frame[:, :, 2] *= 0.97
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif p == "Vivid Color": saturation = max(saturation, 125)
    elif p == "Matte Film": contrast = min(contrast, 90); saturation = min(saturation, 92)
    elif p == "Edge Clarity": clarity = max(clarity, 35)

    if denoise > 0:
        # Fast bilateral filtering at low levels; NLM is stronger but more expensive.
        h = int(denoise * 1.8)
        frame = cv2.fastNlMeansDenoisingColored(frame, None, h, h, 7, 21)
    if structure > 0 or sharpen > 0 or clarity > 0:
        base = frame
        sigma = 0.8 + max(0, structure) / 45.0
        blur = cv2.GaussianBlur(base, (0, 0), sigma)
        amount = max(0, sharpen) / 100.0 + max(0, structure) / 220.0 + max(0, clarity) / 180.0
        frame = cv2.addWeighted(base, 1.0 + amount, blur, -amount, 0)
        if clarity > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray, 50, 130)
            edge = cv2.GaussianBlur(edge, (0, 0), 1.0)
            frame = cv2.addWeighted(frame, 1.0, cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR), clarity / 500.0, 0)
    if contrast != 100 or brightness != 0:
        frame = cv2.convertScaleAbs(frame, alpha=float(contrast) / 100.0, beta=float(brightness))
    if saturation != 100:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation) / 100.0, 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if glow > 0:
        lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(lum, 205, 255, cv2.THRESH_BINARY)[1]
        layer = cv2.bitwise_and(frame, frame, mask=mask)
        layer = cv2.GaussianBlur(layer, (0, 0), 7.0)
        frame = cv2.addWeighted(frame, 1.0, layer, float(glow) / 100.0, 0)
    if grain > 0:
        noise = np.random.default_rng().normal(0, float(grain) * 0.65, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame
