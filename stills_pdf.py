"""
Stills-PDF renderer — one PDF page per biomechanical metric.

Each page shows the source video frame at the metric's key moment with a
Pitchwolf-style tag overlay (label + value + optional good/bad indicator).
No skeleton overlay, no measurement glyphs — clean frame plus floating tag.

Architecture: pure server-side. Takes a video file path + a payload with
events, metrics, and zone scores. Returns PDF bytes. Designed to be called
from a FastAPI endpoint after the frontend has computed metrics; the video
is still on disk because the analyse worker no longer deletes it in finally.

Public API:
    render_stills_pdf(video_path: str, payload: dict) -> bytes

Payload shape (JSON serialisable):
    {
      "bowler": {"name": "Zac", "hand": "right", "date": "4 May 2026"},
      "events": {"bfcFrame": 194, "ffcFrame": 232, "releaseFrame": 316,
                 "backLeavesFrame": 218},  # any may be null
      "metrics": {"runupSpeedKmh": 19.8, "comDispCm": 8, ...},
      "scores": {"runupSpeedKmh": 6.0, ...}  # 0-10; >=7 = "good" tag green
    }

The 13-page layout matches the v4 prototype agreed in the previous session.
"""
from __future__ import annotations
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

logger = logging.getLogger(__name__)


# ─── Visual constants ──────────────────────────────────────────────
TAG_GOOD_RGB    = (16, 185, 129)    # emerald-500
TAG_BAD_RGB     = (249, 115, 22)    # orange-500
TAG_NEUTRAL_RGB = (71, 85, 105)     # slate-600 — for non-scored items
TAG_WHITE_RGB   = (255, 255, 255)
TEXT_DARK_RGB   = (15, 23, 42)      # slate-900
SCORE_GOOD_THRESHOLD = 7.0          # score ≥ this → green tag

ZONE_LABELS = {
    'approach': 'APPROACH ZONE',
    'impact':   'IMPACT ZONE',
    'delivery': 'DELIVERY ZONE',
}


# ─── Value formatters ──────────────────────────────────────────────
def _fmt_kmh(v):  return f'{round(v)} km/h' if v is not None else '—'
def _fmt_cm(v):   return f'{round(v)} cm' if v is not None else '—'
def _fmt_ms(v):   return f'{round(v)} ms' if v is not None else '—'
def _fmt_deg(v):  return f'{round(v)}°' if v is not None else '—'
def _fmt_rpm(v):  return f'{round(v)} RPM' if v is not None else '—'
def _fmt_below(v):
    if v is True:  return 'Below'
    if v is False: return 'Above'
    return '—'
def _fmt_static(label):
    return lambda _v: label


# ─── Page plan ─────────────────────────────────────────────────────
# Frame offsets are heuristic, picked to land on a frame that visually
# represents the metric. Tuned for v2 events on MP path (BFC = real
# touchdown, FFC = real plant). YOLO events differ — see notes below.
#
# `metric_key`   -- key in payload['metrics']; None for static-label pages
# `score_key`    -- key in payload['scores']; controls tag colour
# `formatter`    -- function: value → display string
# `frame_pick`   -- (event_key, offset_in_native_frames)
# `zone`         -- bucket label shown in top-left corner
EVENT_PLAN = [
    {'metric_key': 'runupSpeedKmh',   'score_key': 'runupSpeedKmh',
     'label': 'Run-up Speed',         'formatter': _fmt_kmh,
     'frame_pick': ('bfcFrame', -40), 'zone': 'approach'},

    {'metric_key': 'impStrideCm',     'score_key': 'impStrideCm',
     'label': 'Impulse Stride',       'formatter': _fmt_cm,
     'frame_pick': ('bfcFrame', -16), 'zone': 'approach'},

    {'metric_key': 'isContactMs',     'score_key': 'isContactMs',
     'label': 'IS Contact Time',      'formatter': _fmt_ms,
     'frame_pick': ('bfcFrame', -16), 'zone': 'approach'},

    {'metric_key': 'jumpCm',          'score_key': 'jumpCm',
     'label': 'Jump Height',          'formatter': _fmt_cm,
     'frame_pick': ('bfcFrame', -8),  'zone': 'approach'},

    {'metric_key': None,              'score_key': None,
     'label': 'Back Foot Contact',    'formatter': _fmt_static('Toe strike'),
     'frame_pick': ('bfcFrame', 0),   'zone': 'impact'},

    {'metric_key': 'bfcContactMs',    'score_key': 'bfcContactMs',
     'label': 'BFC Contact Time',     'formatter': _fmt_ms,
     'frame_pick': ('bfcFrame', 0),   'zone': 'impact'},

    {'metric_key': 'bfcCollapseDeg',  'score_key': 'bfcCollapseDeg',
     'label': 'Back-Foot Collapse',   'formatter': _fmt_deg,
     'frame_pick': ('bfcFrame', 12),  'zone': 'impact'},

    {'metric_key': 'ffcContactMs',    'score_key': 'ffcContactMs',
     'label': 'FFC Contact Time',     'formatter': _fmt_ms,
     'frame_pick': ('ffcFrame', -8),  'zone': 'delivery'},

    {'metric_key': 'delStrideCm',     'score_key': 'delStrideCm',
     'label': 'Delivery Stride',      'formatter': _fmt_cm,
     'frame_pick': ('ffcFrame', 0),   'zone': 'impact'},

    {'metric_key': 'contraLimbBelow', 'score_key': 'contraLimbBelow',
     'label': 'Contralateral Arm',    'formatter': _fmt_below,
     'frame_pick': ('ffcFrame', 0),   'zone': 'delivery'},

    {'metric_key': 'fkFlexDeg',       'score_key': 'fkFlexDeg',
     'label': 'Front Knee Flexion',   'formatter': _fmt_deg,
     'frame_pick': ('ffcFrame', 13),  'zone': 'delivery'},

    {'metric_key': 'trunkFlexDeg',    'score_key': 'trunkFlexDeg',
     'label': 'Trunk Flexion',        'formatter': _fmt_deg,
     'frame_pick': ('ffcFrame', 16),  'zone': 'impact'},

    {'metric_key': 'armRpm',          'score_key': 'armRpm',
     'label': 'Arm Speed',            'formatter': _fmt_rpm,
     'frame_pick': ('ffcFrame', 15),  'zone': 'delivery'},
]


# ─── Drawing primitives ────────────────────────────────────────────
def _find_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Pick a usable font from common system locations. Falls back to default."""
    bold_candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
    ]
    regular_candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ]
    for path in (bold_candidates if bold else regular_candidates):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_overlay(bgr_frame: np.ndarray, label: str, value_str: str,
                  zone: str, tag_rgb: Tuple[int, int, int]) -> np.ndarray:
    """Draw a Pitchwolf-style tag onto a BGR frame. Returns BGR frame.

    Layout:
      - Zone label in top-left (small, white, semi-transparent dark backing)
      - Metric tag in bottom-left:
          * white panel: metric label (e.g. "BFC Contact Time")
          * coloured panel right of it: value (e.g. "158 ms")
    """
    # Convert BGR → RGB → PIL for clean text rendering, then back at the end
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil, 'RGBA')

    W, H = pil.size

    # ─ Zone label, top-left ─
    zone_text = ZONE_LABELS.get(zone, zone.upper())
    zfont = _find_font(max(18, H // 50), bold=True)
    zbox = draw.textbbox((0, 0), zone_text, font=zfont)
    zw = zbox[2] - zbox[0]
    zh = zbox[3] - zbox[1]
    zpad = 12
    draw.rectangle(
        [(20, 20), (20 + zw + zpad * 2, 20 + zh + zpad * 2)],
        fill=(15, 23, 42, 200),
    )
    draw.text((20 + zpad, 20 + zpad - zbox[1]), zone_text,
              font=zfont, fill=TAG_WHITE_RGB)

    # ─ Metric tag, bottom-left ─
    label_font = _find_font(max(28, H // 30), bold=True)
    value_font = _find_font(max(40, H // 22), bold=True)
    label_box = draw.textbbox((0, 0), label, font=label_font)
    value_box = draw.textbbox((0, 0), value_str, font=value_font)
    label_w = label_box[2] - label_box[0]
    label_h = label_box[3] - label_box[1]
    value_w = value_box[2] - value_box[0]
    value_h = value_box[3] - value_box[1]

    pad_x = 24
    pad_y = 16
    panel_h = max(label_h, value_h) + pad_y * 2
    label_panel_w = label_w + pad_x * 2
    value_panel_w = value_w + pad_x * 2

    x0 = 32
    y1 = H - 32
    y0 = y1 - panel_h

    # White label panel
    draw.rectangle([(x0, y0), (x0 + label_panel_w, y1)], fill=TAG_WHITE_RGB)
    # Coloured value panel
    draw.rectangle(
        [(x0 + label_panel_w, y0),
         (x0 + label_panel_w + value_panel_w, y1)],
        fill=tag_rgb,
    )

    # Text
    draw.text(
        (x0 + pad_x, y0 + (panel_h - label_h) // 2 - label_box[1]),
        label, font=label_font, fill=TEXT_DARK_RGB,
    )
    draw.text(
        (x0 + label_panel_w + pad_x,
         y0 + (panel_h - value_h) // 2 - value_box[1]),
        value_str, font=value_font, fill=TAG_WHITE_RGB,
    )

    out_rgb = np.array(pil)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


# ─── Frame extraction ──────────────────────────────────────────────
def _read_frame(cap: cv2.VideoCapture, frame_idx: int,
                total_frames: int) -> Optional[np.ndarray]:
    """Seek to frame_idx and return BGR frame, or None if unreadable."""
    if frame_idx < 0:
        frame_idx = 0
    if frame_idx >= total_frames:
        frame_idx = max(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    return bgr


# ─── Main entry point ──────────────────────────────────────────────
def render_stills_pdf(video_path: str, payload: Dict[str, Any]) -> bytes:
    """Generate a stills PDF and return the bytes.

    Raises FileNotFoundError if video_path doesn't exist.
    Raises RuntimeError if cv2 can't open the video.
    Individual page failures are logged and the page is skipped — the PDF
    will still contain whatever pages did render successfully.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        events = payload.get('events') or {}
        metrics = payload.get('metrics') or {}
        scores = payload.get('scores') or {}

        buf = io.BytesIO()
        # A4 landscape so a 16:9 video frame fits naturally with margins
        page_size = landscape(A4)
        c = canvas.Canvas(buf, pagesize=page_size)
        page_w, page_h = page_size

        pages_rendered = 0
        for plan in EVENT_PLAN:
            try:
                # Determine the frame to render
                event_key, offset = plan['frame_pick']
                anchor = events.get(event_key)
                if anchor is None:
                    logger.info("Skipping page %r: event %s missing",
                                plan['label'], event_key)
                    continue
                frame_idx = int(anchor) + int(offset)

                bgr = _read_frame(cap, frame_idx, total_frames)
                if bgr is None:
                    logger.warning("Skipping page %r: could not read frame %d",
                                   plan['label'], frame_idx)
                    continue

                # Compute display value + tag colour
                value = metrics.get(plan['metric_key']) if plan['metric_key'] else None
                value_str = plan['formatter'](value)

                score = scores.get(plan['score_key']) if plan['score_key'] else None
                if score is None:
                    tag_rgb = TAG_NEUTRAL_RGB
                elif score >= SCORE_GOOD_THRESHOLD:
                    tag_rgb = TAG_GOOD_RGB
                else:
                    tag_rgb = TAG_BAD_RGB

                annotated = _draw_overlay(bgr, plan['label'], value_str,
                                          plan['zone'], tag_rgb)

                # Write to a JPEG buffer, then drop it onto the canvas
                ok, jpeg_bytes = cv2.imencode(
                    '.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    logger.warning("Skipping page %r: jpeg encode failed",
                                   plan['label'])
                    continue
                img = ImageReader(io.BytesIO(jpeg_bytes.tobytes()))

                # Fit the frame into the page with a small margin, preserving aspect
                margin = 24
                avail_w = page_w - margin * 2
                avail_h = page_h - margin * 2
                fh, fw = annotated.shape[:2]
                scale = min(avail_w / fw, avail_h / fh)
                draw_w = fw * scale
                draw_h = fh * scale
                draw_x = (page_w - draw_w) / 2
                draw_y = (page_h - draw_h) / 2
                c.drawImage(img, draw_x, draw_y,
                            width=draw_w, height=draw_h,
                            preserveAspectRatio=True, mask='auto')
                c.showPage()
                pages_rendered += 1
            except Exception as page_err:
                logger.exception("Skipping page %r: %s",
                                 plan.get('label'), page_err)
                continue

        if pages_rendered == 0:
            raise RuntimeError("No stills pages rendered — check events/payload")

        c.save()
        return buf.getvalue()
    finally:
        cap.release()
