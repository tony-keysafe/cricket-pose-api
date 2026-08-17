# CLAUDE.md — Cricket Analyze Pro

Read this fully before doing anything. It is the accumulated project knowledge from
months of sessions. The active task, if one exists, is in a separate HANDOFF_*.md file.

## What this project is

**Cricket Analyze Pro** — an AI-powered cricket bowling biomechanics web app for
coaching. Analyzes side-on slow-motion bowling videos, detects pose keypoints, computes
**13 biomechanical metrics across three zones** (Approach, Impact, Delivery), and
generates an annotated Key Moments stills PDF plus a coaching report. Goal: match, then
compete with, the commercial **Pitchwolf / PaceLab** system. A speed-gun feature
(behind-stumps footage) is secondary. Commercial roadmap: Stripe scaffolding built but
not activated, PWA conversion, eventually a custom pose model trained on cricket footage.

**Metric order (stills PDF):** Run-up Speed, Impulse Stride, IS Contact Time, Jump
Height, Back Foot Contact, BFC Contact Time, Back-Foot Collapse, FFC Contact Time,
Delivery Stride, Contralateral Arm, Front Knee Flexion, Trunk Flexion, Arm Speed.
(CoM Displacement is in the report but deliberately excluded from the stills PDF.)

## People

- **Tony** (`tony-keysafe`) — owner, developer, and a right-arm bowler. He is the
  **domain expert: his visual judgment on biomechanics is ground truth.** Reproduce his
  number before patching, never the other way round. He often leaves Claude building
  and returns later to review.
- **Sarah Hind** — collaborator (Mac Mini).
- **Zac** — primary test subject. Video `IMG_5945.mov`, 89 km/h release,
  side-on, 1920×1080, 328 frames, 30fps container × 8× slow-motion = **240fps real**.
  The Pitchwolf analysis of this exact delivery is the accuracy benchmark.

## Working agreement (strictly enforced)

1. **No guess-and-ship.** Gather evidence, trace the full pipeline end-to-end, confirm
   exactly where data breaks before writing any fix.
2. **Verify against live production output**, never local prototypes. (A stills-PDF
   frame bug was masked for weeks because "verification" used prototype output.)
3. **One small revertible commit at a time**, verified in browser before the next.
4. **No fudge factors without explicit approval.** A Front Knee correction-factor
   change was shipped and correctly reverted as a single-bowler fudge.
5. Different metrics need different detector configs — the accepted architecture is
   dual pose detection: full-frame for events/distances, crop-zoom for angle metrics.

## Stack & infrastructure

- Frontend: React/Vite → Vercel (`cricket-analyze-pro.vercel.app`), repo
  `github.com/tony-keysafe/cricket-analyze-pro` (private). Pushes auto-deploy.
- Backend: Python/FastAPI → Railway Pro (`cricket-pose-api-production.up.railway.app`),
  repo `github.com/tony-keysafe/cricket-pose-api` (private). Pushes auto-deploy.
- Auth/DB: Supabase (`doxctvmjxkdzmpwqeeox`). Local blob storage: IndexedDB.
- **Commit identity must be `tony.keysafe@gmail.com` / `tony-keysafe` or Vercel
  blocks the deploy.**

## Current production state (as of Aug 2026)

- **MediaPipe Pose Landmarker Full is the default analyzer** (commit `1aa272b`);
  `?yolo=1` is the opt-in YOLOv8n-pose ONNX fallback (17 keypoints, heel/toe always
  null — cannot do ground-contact detection). Legacy `?mp=1` flag routed to the MP
  endpoint during preview and shows an amber indicator; likely redundant now.
- **`detectEventsV2`** (heel/toe-based event detector, in
  `src/components/analysis/PoseAnalyzer.jsx`) is the promoted BFC/FFC source on the MP
  path. FFC anchors on the front **heel** since commit `263cf88` (Aug 2026); the toe
  pick is logged in `_v2_diagnostics.ffcToeFrame`. Shipped picks for Zac: BFC 194,
  back-leaves 218, FFC 218 (toe diag 232). The coupled arm-window constants
  (`armStart = ffcFrame - 2`, cap `ffcTime + 0.409`) and the stills-PDF FFC lag
  (8.3 ms, backend `4ea1ba5`) are shift-cancellations vs the old toe pick — ratified
  by Tony, production-verified 17 Aug 2026.
- **Key Moments stills PDF** (13 pages, server-side: `stills_pdf.py`, cv2/PIL/reportlab)
  is live and verified page-by-page against Tony's ground-truth contact sheet.
  Anchoring is in real-time seconds with detector-to-visual lag constants (`EVENT_PLAN`).
- **`renderVideo()` is disabled** (function kept, try/catch commented; revert notes in
  the commit message) — broken on iOS, 30–90s wasted CPU. Download Video, JSON export,
  and the Video tab are hidden on the report page (backing code kept).
- Reference artifact: report/PDF #79 is the last verified-good output.

## Ground truth — Zac (definitive calibration reference)

| Event | Visual truth (Tony) | v2 detector pick |
|---|---|---|
| BFC | 190 | 194 (+4, real kinematic lag) |
| back-leaves | 212 | 218 (+6) |
| FFC | 216 | 218 (+2, heel-anchored; sampling-resolution limit) |

Zac heel-strikes: the toe stays dorsiflexed and only plateaus at foot-flat
(~14 frames later), which was the old +16-frame FFC bug. The heel fix shipped
Aug 2026 (frontend `263cf88`, backend `4ea1ba5`) and passed Tony's full production
checklist on 17 Aug 2026.

## Hard-won technical learnings — do not relearn these

- **`sourceRealFps = container_fps × slomo_factor`** must be used for ALL timing and
  frame-offset calculations. `time_base` metadata is unreliable for distinguishing
  120 vs 240fps iPhone captures. (An `effectiveFps` vs `sourceRealFps` mixup once put
  every stills page up to 33 frames off.)
- **MediaPipe L/R labels are from the subject's perspective.** For a right-armer running
  right-to-left the mapping is inverted vs codebase assumptions. Current code hardcodes
  front = L (heel 29 / toe 31), back = R (heel 30 / toe 32) — open FIXME: auto-detect
  from motion direction, required before left-armers or reversed camera angles.
- **Person tracking must stay disabled** for side-on cricket video — horizontal
  movement across frame gets misidentified as a new person, discarding most frames.
- **75th percentile, not maximum, for angle metrics** — the max always picks the
  noisiest frame.
- **Raw (un-smoothed) keypoints for Arm Speed** — EMA at α=0.14 is far too aggressive
  for ~330 RPM wrist motion.
- Front Knee / Trunk Flexion drift vs Pitchwolf ("Stage 4b") is now split — see
  Aug 2026 findings below: Trunk is a sampling-definition mismatch, Front Knee is a
  genuine angle-pipeline difference (crop-zoom dual-pose is the lead suspect).
- Roboflow model `cricket-dataset-z2wkt-ko5pz/1` (broadcast-trained) is unreliable for
  behind-stumps net footage. Side-on ball HSV detection (hue 0–15°/165–180°,
  sat > 80) validated at exactly 89 km/h on Zac.

## Verification rituals

- Frontend before every push: `npx vite build 2>&1 | tail -5`
- Backend before every push: `python3 -c "import ast; ast.parse(open('main.py').read())"`
  (and the same for any edited .py file)
- **Version stamping** (footer + console.log in the main computation function) is the
  most reliable way to confirm which code is actually deployed.
- **Offline harness methodology:** export keypoints JSON → iterate locally → verify
  against frame data → deploy. `run_detector.mjs` scores detector picks against ground
  truth; keep it (and fixtures like `frame_data_1777885444366.json`) in the repo under
  `tools/` and extend it with each new bowler's fixture + truth frames.
- For complex multi-line JS replacements, a small Python script with exact string
  matching beats fragile editor find-replace on whitespace/Unicode mismatches.

## Roadmap

1. **Done (Aug 2026):** HANDOFF_FFC_HEEL_FIX.md executed and production-verified.
2. **Active milestone: second bowler.** Fastest path: Tony films his own right-arm
   delivery, same setup. A mirrored copy of Zac's video exercises the L/R FIXME free;
   a 120fps re-encode tests sourceRealFps handling. Gates: FFC generalization check,
   Stage 4b drift re-measure, then **Stage 6: deprecate YOLO** (after 2–3 bowlers
   validated on MP+v2).
3. Parked: Quick Start vs New Analysis redundancy; optional standing-calibration photo
   (for crouching run-ups / CoM accuracy); speed-gun integration into `/analyze` with
   crease-calibration UI, retiring the separate `/speed` page; make the analyzer choice
   a user setting instead of fragile URL flags.

## Aug 2026 findings (post FFC-heel fix, from Tony's production re-measure)

- **Stage 4b split.** Trunk Flexion drift vs Pitchwolf is a definition mismatch:
  we sample at FFC, Pitchwolf samples at release (~frame 236 on Zac); the value
  moved 13° → 7° with the anchor fix. Front Knee drift (24° vs 17°) is
  anchor-invariant — a genuine angle-pipeline difference; crop-zoom dual-pose is
  the lead suspect.
- **"Release" currently means end-of-tracking** (frame 316 on Zac), not a detected
  ball release — it feeds FFC Contact Time and the arm window. Define it properly
  before bowler #2.
- **Several scoring bands don't match their stated elite targets** — review in the
  definitions pass.

## Housekeeping

CLAUDE.md is committed at both repo roots — keep the copies in sync when it changes.
The offline harness and reference fixtures live under `tools/` in cricket-analyze-pro;
extend them with each new bowler's fixture + truth frames.
Never commit tokens or credentials; never echo secrets into chat or logs.
