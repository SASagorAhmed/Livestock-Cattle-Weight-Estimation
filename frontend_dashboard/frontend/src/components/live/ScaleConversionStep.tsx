import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { fileUrl } from '../../api';
import type {
  ImagePoint,
  MeasureStageResponse,
  PoseStageResponse,
  ScaleMode,
} from '../../types';
import LiveProcessingLayout from './LiveProcessingLayout';
import ReferencePointMagnifier from './ReferencePointMagnifier';
import {
  suggestAnatomicalLines,
  summarizeAvailablePoints,
  type AnatomicalLineId,
} from './suggestAnatomicalEndpoints';
import {
  usePointHistory,
  type SelectedPointId,
} from './usePointHistory';

const MIN_DISTANCE_PX = 2;

interface Props {
  measure: MeasureStageResponse | null;
  pose: PoseStageResponse | null;
  busy: boolean;
  onSkip: () => void;
  onApply: (payload: {
    reference_px: number;
    reference_cm: number;
    point_a: ImagePoint;
    point_b: ImagePoint;
  }) => void;
}

function validateCm(raw: string): { ok: boolean; value: number | null; error: string } {
  if (raw.trim() === '') {
    return { ok: false, value: null, error: 'Enter the known object length in centimetres.' };
  }
  const n = Number(raw);
  if (!Number.isFinite(n) || Number.isNaN(n)) {
    return { ok: false, value: null, error: 'Enter a valid number.' };
  }
  if (n <= 0) {
    return { ok: false, value: null, error: 'Length must be greater than zero.' };
  }
  return { ok: true, value: n, error: '' };
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

function clampPoint(p: ImagePoint, w: number, h: number): ImagePoint {
  return {
    x: Math.round(clamp(p.x, 0, Math.max(0, w - 0.01)) * 100) / 100,
    y: Math.round(clamp(p.y, 0, Math.max(0, h - 0.01)) * 100) / 100,
  };
}

/** Map client pointer to original-image coords using the rendered <img> box. */
function clientToOriginal(
  clientX: number,
  clientY: number,
  img: HTMLImageElement,
): { original: ImagePoint; displayX: number; displayY: number } | null {
  const rect = img.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0 || img.naturalWidth <= 0) return null;

  const displayX = clamp(clientX - rect.left, 0, rect.width);
  const displayY = clamp(clientY - rect.top, 0, rect.height);
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const original = clampPoint(
    { x: displayX * scaleX, y: displayY * scaleY },
    img.naturalWidth,
    img.naturalHeight,
  );
  return { original, displayX, displayY };
}

const LINEAR_KEYS = [
  'body_length',
  'body_height',
  'chest_depth_proxy',
  'left_front_leg_length',
  'right_front_leg_length',
  'left_back_leg_length',
  'right_back_leg_length',
  'torso_diagonal',
  'shoulder_width',
  'hip_width',
] as const;

export default function ScaleConversionStep({ measure, pose, busy, onSkip, onApply }: Props) {
  const [mode, setMode] = useState<ScaleMode>('UNSELECTED');
  const [pointA, setPointA] = useState<ImagePoint | null>(null);
  const [pointB, setPointB] = useState<ImagePoint | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<SelectedPointId>(null);
  const [suggestNote, setSuggestNote] = useState('');
  const [activeSuggestId, setActiveSuggestId] = useState<AnatomicalLineId | null>(null);
  const [draggingPoint, setDraggingPoint] = useState<SelectedPointId>(null);
  const [hoveredPoint, setHoveredPoint] = useState<SelectedPointId>(null);
  const [cmRaw, setCmRaw] = useState('');
  const [pointError, setPointError] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [showTech, setShowTech] = useState(false);
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });
  const [imageReady, setImageReady] = useState(false);
  const [zoomActive, setZoomActive] = useState(false);
  const [zoomDisplay, setZoomDisplay] = useState({ x: 0, y: 0 });
  const [zoomImage, setZoomImage] = useState<HTMLImageElement | null>(null);

  const imgRef = useRef<HTMLImageElement>(null);
  const draggingRef = useRef<SelectedPointId>(null);
  const pointARef = useRef(pointA);
  const pointBRef = useRef(pointB);
  const selectedRef = useRef(selectedPoint);
  pointARef.current = pointA;
  pointBRef.current = pointB;
  selectedRef.current = selectedPoint;

  const { canUndo, canRedo, pushSnapshot, undo, redo, clearHistory } = usePointHistory();

  const imageSrc = fileUrl(measure?.files?.['original_image.jpg']);
  const pxVals = measure?.measurements?.measurements_px || {};
  const seg = measure?.measurements?.segmentation;
  const poseKeypoints = pose?.selected_detection?.keypoints;
  const anatomicalSuggestions = useMemo(
    () => suggestAnatomicalLines(poseKeypoints),
    [poseKeypoints],
  );
  const anatomicalAvailability = useMemo(
    () => summarizeAvailablePoints(poseKeypoints),
    [poseKeypoints],
  );

  const snapshot = useCallback(() => ({
    pointA: pointARef.current,
    pointB: pointBRef.current,
    selectedPoint: selectedRef.current,
  }), []);

  const applySuggestion = useCallback((id: AnatomicalLineId) => {
    if (busy || confirmed) return;
    const suggestion = anatomicalSuggestions.find((s) => s.id === id);
    if (!suggestion) {
      setPointError('Pose keypoints for this suggestion are not available.');
      return;
    }
    const a = clampPoint(
      suggestion.pointA.point,
      naturalSize.width || 1,
      naturalSize.height || 1,
    );
    const b = clampPoint(
      suggestion.pointB.point,
      naturalSize.width || 1,
      naturalSize.height || 1,
    );
    if (Math.hypot(b.x - a.x, b.y - a.y) < MIN_DISTANCE_PX) {
      setPointError('Suggested endpoints are too close. Adjust manually.');
      return;
    }
    pushSnapshot(snapshot());
    setPointA(a);
    setPointB(b);
    setSelectedPoint('B');
    setConfirmed(false);
    setActiveSuggestId(id);
    setSuggestNote(
      [
        suggestion.description,
        `${suggestion.pointA.note} · ${suggestion.pointB.note}`,
        ...suggestion.warnings,
      ].filter(Boolean).join(' '),
    );
    setPointError('');
  }, [
    anatomicalSuggestions,
    busy,
    confirmed,
    naturalSize.height,
    naturalSize.width,
    pushSnapshot,
    snapshot,
  ]);

  const applySnapshot = useCallback((s: {
    pointA: ImagePoint | null;
    pointB: ImagePoint | null;
    selectedPoint: SelectedPointId;
  }) => {
    setPointA(s.pointA);
    setPointB(s.pointB);
    setSelectedPoint(s.selectedPoint);
    setConfirmed(false);
    setPointError('');
    setSuggestNote('');
    setActiveSuggestId(null);
  }, []);

  const syncDisplaySize = useCallback(() => {
    const img = imgRef.current;
    if (!img) {
      setImageReady(false);
      setZoomImage(null);
      return;
    }
    if (!img.complete || img.naturalWidth <= 0) {
      setImageReady(false);
      setZoomImage(null);
      return;
    }
    const rect = img.getBoundingClientRect();
    setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    setDisplaySize({
      width: Math.max(1, rect.width),
      height: Math.max(1, rect.height),
    });
    setZoomImage(img);
    setImageReady(true);
  }, []);

  useEffect(() => {
    if (mode !== 'WITH_REFERENCE') return undefined;
    syncDisplaySize();
    const img = imgRef.current;
    if (!img) return undefined;
    const ro = new ResizeObserver(() => syncDisplaySize());
    ro.observe(img);
    window.addEventListener('resize', syncDisplaySize);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', syncDisplaySize);
    };
  }, [syncDisplaySize, imageSrc, mode]);

  const referencePixels = useMemo(() => {
    if (!pointA || !pointB) return null;
    return Math.hypot(pointB.x - pointA.x, pointB.y - pointA.y);
  }, [pointA, pointB]);

  useEffect(() => {
    if (pointA && pointB && referencePixels != null && referencePixels < MIN_DISTANCE_PX) {
      setPointError('The selected endpoints are too close. Move one point farther away.');
    } else {
      setPointError((prev) => (prev.includes('too close') ? '' : prev));
    }
  }, [pointA, pointB, referencePixels]);

  const cmValidation = useMemo(() => validateCm(cmRaw), [cmRaw]);

  const cmPerPixel = useMemo(() => {
    if (!referencePixels || referencePixels <= 0 || !cmValidation.ok || cmValidation.value == null) {
      return null;
    }
    const v = cmValidation.value / referencePixels;
    return Number.isFinite(v) && v > 0 ? v : null;
  }, [referencePixels, cmValidation]);

  const previewConversions = useMemo(() => {
    if (cmPerPixel == null) return [];
    const rows: Array<{ name: string; px: number; cm: number; kind: 'linear' | 'area' }> = [];
    for (const key of LINEAR_KEYS) {
      const v = pxVals[key];
      if (v == null) continue;
      rows.push({ name: key, px: v, cm: v * cmPerPixel, kind: 'linear' });
    }
    if (seg?.body_perimeter_px != null) {
      rows.push({
        name: 'body_perimeter',
        px: seg.body_perimeter_px,
        cm: seg.body_perimeter_px * cmPerPixel,
        kind: 'linear',
      });
    }
    if (seg?.body_pixel_area != null) {
      rows.push({
        name: 'body_area',
        px: seg.body_pixel_area,
        cm: seg.body_pixel_area * (cmPerPixel ** 2),
        kind: 'area',
      });
    }
    if (seg?.torso_pixel_area != null) {
      rows.push({
        name: 'torso_area',
        px: seg.torso_pixel_area,
        cm: seg.torso_pixel_area * (cmPerPixel ** 2),
        kind: 'area',
      });
    }
    return rows;
  }, [cmPerPixel, pxVals, seg]);

  const canConfirm = Boolean(
    imageReady
    && pointA
    && pointB
    && referencePixels
    && referencePixels > MIN_DISTANCE_PX
    && cmValidation.ok
    && cmPerPixel != null
    && !confirmed
    && !busy,
  );

  const instruction = useMemo(() => {
    if (!imageReady) return 'Loading image…';
    if (draggingPoint === 'A') return 'Adjusting Point A…';
    if (draggingPoint === 'B') return 'Adjusting Point B…';
    if (!pointA && !pointB) return 'Click the first endpoint of the known-length segment.';
    if (pointA && !pointB) return 'Point A placed. Click the second endpoint.';
    if (!pointA && pointB) return 'Point B exists. Click to place Point A.';
    if (selectedPoint) {
      return `Point ${selectedPoint} selected — drag to adjust, or click empty area to move it.`;
    }
    return 'Endpoints selected. Drag either point to adjust.';
  }, [imageReady, draggingPoint, pointA, pointB, selectedPoint]);

  const viewW = Math.max(1, naturalSize.width);
  const viewH = Math.max(1, naturalSize.height);
  const markerBase = Math.min(viewW, viewH);
  const baseR = Math.max(6, markerBase * 0.012);
  const hitR = Math.max(18, markerBase * 0.028);

  const updateZoom = useCallback((displayX: number, displayY: number, active: boolean) => {
    setZoomDisplay({ x: displayX, y: displayY });
    setZoomActive(active);
  }, []);

  const placeOrMoveFromClient = useCallback((clientX: number, clientY: number) => {
    if (busy || confirmed || !imageReady) return;
    const img = imgRef.current;
    if (!img) return;
    const mapped = clientToOriginal(clientX, clientY, img);
    if (!mapped) return;
    const { original, displayX, displayY } = mapped;
    updateZoom(displayX, displayY, true);

    const a = pointARef.current;
    const b = pointBRef.current;
    const sel = selectedRef.current;

    if (a && b && sel) {
      pushSnapshot(snapshot());
      if (sel === 'A') setPointA(original);
      else setPointB(original);
      setConfirmed(false);
      return;
    }

    if (!a && !b) {
      pushSnapshot(snapshot());
      setPointA(original);
      setSelectedPoint('A');
      return;
    }
    if (a && !b) {
      if (Math.hypot(original.x - a.x, original.y - a.y) < MIN_DISTANCE_PX) {
        setPointError('The selected endpoints are too close. Move one point farther away.');
        return;
      }
      pushSnapshot(snapshot());
      setPointB(original);
      setSelectedPoint('B');
      return;
    }
    if (!a && b) {
      if (Math.hypot(original.x - b.x, original.y - b.y) < MIN_DISTANCE_PX) {
        setPointError('The selected endpoints are too close. Move one point farther away.');
        return;
      }
      pushSnapshot(snapshot());
      setPointA(original);
      setSelectedPoint('A');
    }

    window.setTimeout(() => {
      if (!draggingRef.current) setZoomActive(false);
    }, 700);
  }, [busy, confirmed, imageReady, pushSnapshot, snapshot, updateZoom]);

  const startDrag = useCallback((id: 'A' | 'B', e: ReactPointerEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (busy || !imageReady) return;

    pushSnapshot(snapshot());
    draggingRef.current = id;
    setDraggingPoint(id);
    setSelectedPoint(id);
    setConfirmed(false);

    const img = imgRef.current;
    if (img) {
      const mapped = clientToOriginal(e.clientX, e.clientY, img);
      if (mapped) updateZoom(mapped.displayX, mapped.displayY, true);
    }

    const move = (ev: PointerEvent) => {
      if (draggingRef.current !== id) return;
      const el = imgRef.current;
      if (!el) return;
      const mapped = clientToOriginal(ev.clientX, ev.clientY, el);
      if (!mapped) return;
      if (id === 'A') setPointA(mapped.original);
      else setPointB(mapped.original);
      setConfirmed(false);
      updateZoom(mapped.displayX, mapped.displayY, true);
    };

    const up = () => {
      draggingRef.current = null;
      setDraggingPoint(null);
      setZoomActive(false);
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
    };

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);

    try {
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  }, [busy, imageReady, pushSnapshot, snapshot, updateZoom]);

  const onBgHitPointerDown = (e: ReactPointerEvent<SVGRectElement>) => {
    e.stopPropagation();
    if (busy || confirmed || !imageReady) return;
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    placeOrMoveFromClient(e.clientX, e.clientY);
  };

  const doUndo = () => {
    const prev = undo(snapshot());
    if (prev) applySnapshot(prev);
  };

  const doRedo = () => {
    const next = redo(snapshot());
    if (next) applySnapshot(next);
  };

  const removeSelected = () => {
    if (!selectedPoint) return;
    pushSnapshot(snapshot());
    if (selectedPoint === 'A') setPointA(null);
    else setPointB(null);
    setSelectedPoint(null);
    setConfirmed(false);
  };

  const resetPoints = () => {
    if (!pointA && !pointB) return;
    pushSnapshot(snapshot());
    setPointA(null);
    setPointB(null);
    setSelectedPoint(null);
    setConfirmed(false);
    setPointError('');
    setZoomActive(false);
    setSuggestNote('');
    setActiveSuggestId(null);
  };

  const restartSelection = () => {
    if (confirmed) {
      const ok = window.confirm(
        'Confirmed scale data will be cleared. Restart endpoint selection?',
      );
      if (!ok) return;
    }
    pushSnapshot(snapshot());
    setPointA(null);
    setPointB(null);
    setSelectedPoint(null);
    setCmRaw('');
    setPointError('');
    setConfirmed(false);
    setZoomActive(false);
    setSuggestNote('');
    setActiveSuggestId(null);
  };

  useEffect(() => {
    if (mode !== 'WITH_REFERENCE') return undefined;

    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if ((e.target as HTMLElement)?.isContentEditable) return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) doRedo();
        else doUndo();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        doRedo();
        return;
      }
      if (e.key === 'Escape') {
        setSelectedPoint(null);
        return;
      }
      if (!selectedRef.current) return;

      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        removeSelected();
        return;
      }

      const step = e.shiftKey ? 10 : 1;
      let dx = 0;
      let dy = 0;
      if (e.key === 'ArrowLeft') dx = -step;
      else if (e.key === 'ArrowRight') dx = step;
      else if (e.key === 'ArrowUp') dy = -step;
      else if (e.key === 'ArrowDown') dy = step;
      else return;

      e.preventDefault();
      const sel = selectedRef.current;
      const cur = sel === 'A' ? pointARef.current : pointBRef.current;
      if (!cur || !sel) return;
      pushSnapshot(snapshot());
      const next = clampPoint(
        { x: cur.x + dx, y: cur.y + dy },
        naturalSize.width || 1,
        naturalSize.height || 1,
      );
      if (sel === 'A') setPointA(next);
      else setPointB(next);
      setConfirmed(false);
    };

    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [mode, naturalSize.width, naturalSize.height]);

  const renderEndpoint = (id: 'A' | 'B', p: ImagePoint) => {
    const isSel = selectedPoint === id;
    const isDrag = draggingPoint === id;
    const isHover = hoveredPoint === id;
    const r = baseR * (isDrag ? 1.35 : isSel ? 1.2 : isHover ? 1.1 : 1);

    return (
      <g key={id}>
        <circle
          className="ref-hit"
          cx={p.x}
          cy={p.y}
          r={hitR}
          fill="transparent"
          style={{ cursor: isDrag ? 'grabbing' : 'grab', touchAction: 'none' }}
          onPointerDown={(e) => startDrag(id, e)}
          onPointerEnter={() => setHoveredPoint(id)}
          onPointerLeave={() => setHoveredPoint((h) => (h === id ? null : h))}
          role="button"
          tabIndex={0}
          aria-label={`Reference endpoint ${id}`}
          aria-pressed={isSel}
          onKeyDown={(e: ReactKeyboardEvent) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              setSelectedPoint(id);
            }
          }}
        />
        {(isSel || isDrag) ? (
          <circle
            cx={p.x}
            cy={p.y}
            r={r * 1.7}
            fill="none"
            stroke="#1b5e20"
            strokeWidth={Math.max(2, markerBase * 0.003)}
            strokeDasharray={isSel && !isDrag ? '4 3' : undefined}
            pointerEvents="none"
          />
        ) : null}
        <circle
          cx={p.x}
          cy={p.y}
          r={r}
          fill={isDrag ? '#c8e6c9' : '#fff'}
          stroke="#1b5e20"
          strokeWidth={Math.max(2, markerBase * 0.0025)}
          pointerEvents="none"
        />
        <text
          x={clamp(p.x + r * 1.6, 0, viewW - 4)}
          y={clamp(p.y - r * 1.2, 12, viewH)}
          fill="#1b5e20"
          fontSize={Math.max(12, markerBase * 0.028)}
          fontWeight={700}
          pointerEvents="none"
        >
          {id}
        </text>
      </g>
    );
  };

  if (mode === 'UNSELECTED') {
    return (
      <LiveProcessingLayout title="Reference Scale" subtitle="Optional" status="Choose how to continue">
        <p className="live-sub">
          If an object of known size is visible in the image, select its two endpoints.
          The application will calculate its pixel length automatically.
        </p>
        <div className="scale-choice-grid">
          <button type="button" className="scale-choice-card" disabled={busy} onClick={() => setMode('WITHOUT_SCALE')}>
            <strong>1. Continue without reference scale</strong>
            <span>Keep measurements in pixels and use normalised features.</span>
          </button>
          <button
            type="button"
            className="scale-choice-card"
            disabled={busy}
            onClick={() => {
              clearHistory();
              setMode('WITH_REFERENCE');
            }}
          >
            <strong>2. Use a known-size reference object</strong>
            <span>Click two endpoints, or auto-suggest from pose (tail→shoulder / chest→withers), then drag to fine-tune.</span>
          </button>
        </div>
      </LiveProcessingLayout>
    );
  }

  if (mode === 'WITHOUT_SCALE') {
    return (
      <LiveProcessingLayout
        title="Reference Scale"
        subtitle="Optional"
        status="Continue without scale"
        footer={(
          <div className="btn-row">
            <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => setMode('UNSELECTED')}>Back</button>
            <button type="button" className="btn btn-primary" disabled={busy} onClick={onSkip}>Continue Without Scale</button>
          </div>
        )}
      >
        <div className="measure-formula" role="status">
          No reference scale was provided. Pixel measurements and normalised features will be used.
        </div>
      </LiveProcessingLayout>
    );
  }

  return (
    <LiveProcessingLayout
      title="Reference Scale"
      subtitle="Place endpoints, then drag to adjust. Enter only the real length in centimetres."
      status={instruction}
      footer={(
        <div className="btn-row">
          <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => { setMode('UNSELECTED'); setCmRaw(''); }}>
            Back
          </button>
          <button type="button" className="btn btn-ghost" disabled={busy} onClick={onSkip}>
            Continue Without Scale
          </button>
          {confirmed ? (
            <>
              <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => setConfirmed(false)}>
                Edit Points
              </button>
              <button
                type="button"
                className="btn btn-primary"
                disabled={busy || !pointA || !pointB || !cmValidation.value || !referencePixels}
                onClick={() => {
                  if (!pointA || !pointB || !referencePixels || cmValidation.value == null) return;
                  onApply({
                    reference_px: referencePixels,
                    reference_cm: cmValidation.value,
                    point_a: pointA,
                    point_b: pointB,
                  });
                }}
              >
                Continue to Features
              </button>
            </>
          ) : (
            <button
              type="button"
              className="btn btn-primary"
              disabled={!canConfirm}
              onClick={() => { if (canConfirm) setConfirmed(true); }}
            >
              Confirm Scale
            </button>
          )}
        </div>
      )}
    >
      <div className="ref-toolbar" role="toolbar" aria-label="Reference point tools">
        <button type="button" className="btn btn-ghost" disabled={!canUndo || busy} onClick={doUndo} title="Undo">Undo</button>
        <button type="button" className="btn btn-ghost" disabled={!canRedo || busy} onClick={doRedo} title="Redo">Redo</button>
        <button type="button" className="btn btn-ghost" disabled={!selectedPoint || busy || confirmed} onClick={removeSelected} title="Remove selected">Remove Selected</button>
        <button type="button" className="btn btn-ghost" disabled={busy || (!pointA && !pointB)} onClick={resetPoints} title="Clear both">Reset Points</button>
        <button type="button" className="btn btn-ghost" disabled={busy} onClick={restartSelection} title="Restart">Restart Selection</button>
      </div>

      {anatomicalSuggestions.length > 0 ? (
        <div className="ref-suggest-panel">
          <div className="ref-suggest-panel__title">Auto-suggest from pose (AP-10K)</div>
          <p className="ref-suggest-panel__hint">
            No extra dataset needed. Places endpoints from detected keypoints; drag to correct.
            True brisket/withers are not in this model.
          </p>
          <div className="ref-suggest-actions">
            {anatomicalSuggestions.map((s) => (
              <button
                key={s.id}
                type="button"
                className={`btn ${activeSuggestId === s.id ? 'btn-primary' : 'btn-ghost'}`}
                disabled={busy || confirmed || !imageReady}
                onClick={() => applySuggestion(s.id)}
                title={s.description}
              >
                {s.label}
              </button>
            ))}
          </div>
          <ul className="ref-suggest-availability">
            {anatomicalAvailability.map((row) => (
              <li key={row.role}>
                <strong>{row.role}:</strong>
                {' '}
                {row.available ? row.source : 'missing'}
              </li>
            ))}
          </ul>
          {suggestNote ? (
            <div className="measure-formula" role="status">{suggestNote}</div>
          ) : null}
        </div>
      ) : (
        <div className="ref-suggest-panel">
          <div className="ref-suggest-panel__title">Auto-suggest unavailable</div>
          <p className="ref-suggest-panel__hint">
            Pose keypoints were not found for this run. Place A/B manually on the image.
          </p>
        </div>
      )}

      <div className="ref-work-area">
        <div className="calib-image-wrap reference-image-wrapper">
          {imageSrc ? (
            <img
              ref={imgRef}
              src={imageSrc}
              alt="Uploaded cow image for reference calibration"
              onLoad={syncDisplaySize}
              draggable={false}
            />
          ) : (
            <div className="error-box">Original image is not available.</div>
          )}
          {imageReady && displaySize.width > 0 ? (
            <svg
              className="calib-overlay ref-overlay"
              width={displaySize.width}
              height={displaySize.height}
              viewBox={`0 0 ${viewW} ${viewH}`}
              preserveAspectRatio="none"
              style={{ touchAction: 'none', pointerEvents: 'auto' }}
            >
              <rect
                className="ref-bg-hit"
                x={0}
                y={0}
                width={viewW}
                height={viewH}
                fill="transparent"
                onPointerDown={onBgHitPointerDown}
              />
              {pointA && pointB ? (
                <line
                  x1={pointA.x}
                  y1={pointA.y}
                  x2={pointB.x}
                  y2={pointB.y}
                  stroke="#2f7d32"
                  strokeWidth={Math.max(2, markerBase * 0.004)}
                  pointerEvents="none"
                />
              ) : null}
              {pointA && pointB && referencePixels != null ? (
                <text
                  x={(pointA.x + pointB.x) / 2}
                  y={(pointA.y + pointB.y) / 2 - markerBase * 0.02}
                  fill="#1b5e20"
                  stroke="#e8f5e9"
                  strokeWidth={3}
                  paintOrder="stroke"
                  fontSize={Math.max(12, markerBase * 0.025)}
                  textAnchor="middle"
                  fontWeight={700}
                  pointerEvents="none"
                >
                  {referencePixels.toFixed(2)}
                  {' '}
                  px
                </text>
              ) : null}
              {pointA ? renderEndpoint('A', pointA) : null}
              {pointB ? renderEndpoint('B', pointB) : null}
            </svg>
          ) : null}
        </div>

        <ReferencePointMagnifier
          active={zoomActive && imageReady}
          image={zoomImage}
          displayX={zoomDisplay.x}
          displayY={zoomDisplay.y}
          imageDisplayWidth={displaySize.width}
          imageDisplayHeight={displaySize.height}
          label={draggingPoint ? `Zoom — Point ${draggingPoint}` : 'Zoom preview'}
        />
      </div>

      {pointError ? <div className="error-box" role="alert">{pointError}</div> : null}

      <div className="ref-live-panel">
        <div><strong>Selected:</strong> {selectedPoint ? `Point ${selectedPoint}` : 'None'}</div>
        <div><strong>Point A:</strong> {pointA ? `x ${pointA.x.toFixed(1)}, y ${pointA.y.toFixed(1)}` : '—'}</div>
        <div><strong>Point B:</strong> {pointB ? `x ${pointB.x.toFixed(1)}, y ${pointB.y.toFixed(1)}` : '—'}</div>
        <div><strong>Distance:</strong> {referencePixels != null ? `${referencePixels.toFixed(2)} px` : '—'}</div>
        <div><strong>Real length:</strong> {cmValidation.ok && cmValidation.value != null ? `${cmValidation.value.toFixed(2)} cm` : '—'}</div>
        <div><strong>Scale:</strong> {cmPerPixel != null ? `${cmPerPixel.toFixed(6)} cm/px` : '—'}</div>
      </div>

      {pointA && pointB ? (
        <div className="scale-form" style={{ marginTop: '1rem' }}>
          <label htmlFor="real-object-length">
            Real object length (cm)
            <input
              id="real-object-length"
              type="number"
              min="0"
              step="any"
              inputMode="decimal"
              value={cmRaw}
              disabled={confirmed || busy}
              onChange={(e) => { setCmRaw(e.target.value); setConfirmed(false); }}
              placeholder="e.g. 20"
              aria-invalid={cmRaw !== '' && !cmValidation.ok}
            />
          </label>
        </div>
      ) : null}

      {cmRaw !== '' && !cmValidation.ok ? (
        <div className="error-box" role="alert">{cmValidation.error}</div>
      ) : null}

      {cmPerPixel != null && referencePixels != null && cmValidation.value != null ? (
        <div className="measure-formula" style={{ marginTop: '0.85rem' }}>
          <div>
            cm/px =
            {' '}
            {cmValidation.value.toFixed(2)}
            {' ÷ '}
            {referencePixels.toFixed(2)}
            {' = '}
            <strong>{cmPerPixel.toFixed(6)}</strong>
          </div>
          {previewConversions.slice(0, 3).map((row) => (
            <div key={row.name} style={{ marginTop: 6 }}>
              {row.name}
              :
              {' '}
              {row.px.toFixed(2)}
              {row.kind === 'area' ? ' px²' : ' px'}
              {' → '}
              <strong>
                {row.cm.toFixed(2)}
                {row.kind === 'area' ? ' cm²' : ' cm'}
              </strong>
            </div>
          ))}
        </div>
      ) : null}

      {confirmed ? (
        <div className="success-box" role="status" style={{ marginTop: '1rem' }}>
          <strong>Reference scale applied successfully.</strong>
          {' '}
          Use Edit Points to adjust — moving points requires Confirm again.
        </div>
      ) : null}

      <div className="btn-row">
        <button type="button" className="btn btn-ghost" onClick={() => setShowTech((v) => !v)}>
          {showTech ? 'Hide' : 'Show'}
          {' '}
          technical details
        </button>
      </div>
      {showTech ? (
        <div className="measure-formula">
          <div>
            Natural:
            {naturalSize.width}
            ×
            {naturalSize.height}
          </div>
          <div>
            Displayed:
            {displaySize.width.toFixed(1)}
            ×
            {displaySize.height.toFixed(1)}
          </div>
          <div>SVG preserveAspectRatio=none; box sized to rendered image</div>
          <div>Drag uses window pointer listeners + refs (not async state)</div>
        </div>
      ) : null}
    </LiveProcessingLayout>
  );
}
