import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { fileUrl, stageFourPointDebug, stageFourPointSuggest } from '../../api';
import type { HeadAnchor, ImagePoint, MeasureStageResponse, PoseStageResponse } from '../../types';
import { formatHeadFacing, forwardShoulderRefLabel, pointDisplayLabel } from '../../utils/formatPartName';
import { aEndFromKeypoints, aEndFromLineDict } from '../../utils/aEndLine';
import {
  BODY_OUTLINE_STROKE,
  contourPathFromBodyContour,
  outlineImageFromFiles,
  outlineStrokeWidth,
} from '../../utils/bodyOutline';

const KEYS = [
  'A_start_lower_chest',
  'A_end_withers',
  'B_start_tail_head',
  'B_end_shoulder_region',
] as const;

type KeyName = (typeof KEYS)[number];

type PointMeta = ImagePoint & {
  name?: string;
  status?: 'auto_suggested' | 'manual_corrected';
  method?: string;
  confidence?: number;
  anatomy_label?: string;
  source_keypoint?: string;
};

const DESCRIPTIONS: Record<KeyName, string> = {
  A_start_lower_chest: 'On Lower chest midpoint (shoulder center ↔ hip center)',
  A_end_withers: 'Upper tip of Body height vertical (extended Back top)',
  B_start_tail_head: 'Tail head (body join, not tip)',
  B_end_shoulder_region: 'Forward shoulder region (formula point — drag to adjust; separate from shoulder marker)',
};

const CM_PER_INCH = 2.54;
const LB_PER_KG = 2.20462;

interface Props {
  measure: MeasureStageResponse | null;
  pose?: PoseStageResponse | null;
  runId: string | null;
  busy: boolean;
  onApply: (payload: {
    reference_px: number;
    reference_cm: number;
    point_a: ImagePoint;
    point_b: ImagePoint;
    four_points: Record<string, PointMeta>;
  }) => void;
}

function clamp(n: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, n));
}

function dist(a: ImagePoint, b: ImagePoint) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function clientToImg(
  clientX: number,
  clientY: number,
  img: HTMLImageElement,
): ImagePoint | null {
  const rect = img.getBoundingClientRect();
  if (rect.width <= 0 || img.naturalWidth <= 0) return null;
  const dx = clamp(clientX - rect.left, 0, rect.width);
  const dy = clamp(clientY - rect.top, 0, rect.height);
  return {
    x: Math.round(((dx / rect.width) * img.naturalWidth) * 100) / 100,
    y: Math.round(((dy / rect.height) * img.naturalHeight) * 100) / 100,
  };
}

function computeLive(
  points: Partial<Record<KeyName, PointMeta>>,
  cmPerPx: number | null,
) {
  const a1 = points.A_start_lower_chest;
  const a2 = points.A_end_withers;
  const b1 = points.B_start_tail_head;
  const b2 = points.B_end_shoulder_region;
  if (!a1 || !a2 || !b1 || !b2) {
    return null;
  }
  const A_px = dist(a1, a2);
  const B_px = dist(b1, b2);
  if (A_px < 1 || B_px < 1) return null;

  const base = { A_px, B_px, scaled: false as const };
  if (cmPerPx == null || cmPerPx <= 0) {
    return base;
  }
  const A_cm = A_px * cmPerPx;
  const B_cm = B_px * cmPerPx;
  const A_in = A_cm / CM_PER_INCH;
  const B_in = B_cm / CM_PER_INCH;
  const C_in = 2 * A_in;
  const C_cm = C_in * CM_PER_INCH;
  const weight_lb = (C_in * C_in * B_in) / 300;
  if (!Number.isFinite(weight_lb) || weight_lb <= 0) {
    return base;
  }
  return {
    A_px,
    B_px,
    A_cm,
    B_cm,
    C_cm,
    C_in,
    weight_lb,
    weight_kg: weight_lb / LB_PER_KG,
    scaled: true as const,
  };
}

export default function SmartphoneDiagonalStep({ measure, pose, runId, busy, onApply }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [display, setDisplay] = useState({ w: 0, h: 0 });
  const [ready, setReady] = useState(false);

  const [refA, setRefA] = useState<ImagePoint | null>(null);
  const [refB, setRefB] = useState<ImagePoint | null>(null);
  const [refCm, setRefCm] = useState('');
  const [refPhase, setRefPhase] = useState<'A' | 'B' | 'done'>('A');

  const [points, setPoints] = useState<Partial<Record<KeyName, PointMeta>>>({});
  const [pointStatuses, setPointStatuses] = useState<Partial<Record<KeyName, string>>>({});
  const [selected, setSelected] = useState<KeyName | 'refA' | 'refB' | null>(null);
  const [placeNext, setPlaceNext] = useState<KeyName | null>('A_start_lower_chest');
  const [mode, setMode] = useState<'reference' | 'anatomy'>('reference');
  const [headDirection, setHeadDirection] = useState<'left' | 'right' | ''>('');
  const [headRequired, setHeadRequired] = useState(false);
  const [headDetected, setHeadDetected] = useState<boolean | null>(null);
  const [headAnchor, setHeadAnchor] = useState<HeadAnchor | null>(null);
  const [pointDetector, setPointDetector] = useState<string | null>(null);
  const [suggestMsg, setSuggestMsg] = useState('');
  const [error, setError] = useState('');
  const [lowerChestGuide, setLowerChestGuide] = useState<{
    detected: boolean;
    p1?: [number, number] | null;
    p2?: [number, number] | null;
    mid?: [number, number] | null;
    label?: string;
  } | null>(null);
  const [aEndLine, setAEndLine] = useState<{
    detected: boolean;
    a_end?: [number, number] | null;
    ground?: [number, number] | null;
    p1?: [number, number] | null;
    p2?: [number, number] | null;
    label?: string;
    line_label?: string;
  } | null>(null);
  const dragging = useRef<KeyName | 'refA' | 'refB' | null>(null);
  const pointsRef = useRef<Partial<Record<KeyName, PointMeta>>>({});
  const debugTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const debugInFlight = useRef(false);
  const debugPending = useRef(false);

  const imageSrc = outlineImageFromFiles(measure?.files)
    || outlineImageFromFiles(pose?.files)
    || fileUrl(measure?.files?.['original_image.jpg']);
  const usingOutlineBase = Boolean(
    measure?.files?.['body_outline.jpg']
    || measure?.files?.['measure_outline.jpg']
    || pose?.files?.['body_outline.jpg']
    || pose?.files?.['measure_outline.jpg'],
  );
  const editContourPath = useMemo(
    () => (usingOutlineBase
      ? null
      : contourPathFromBodyContour(measure?.body_contour || pose?.body_contour)),
    [usingOutlineBase, measure?.body_contour, pose?.body_contour],
  );

  const syncSize = useCallback(() => {
    const img = imgRef.current;
    if (!img?.complete || img.naturalWidth <= 0) {
      setReady(false);
      return;
    }
    const r = img.getBoundingClientRect();
    setNatural({ w: img.naturalWidth, h: img.naturalHeight });
    setDisplay({ w: Math.max(1, r.width), h: Math.max(1, r.height) });
    setReady(true);
  }, []);

  useEffect(() => {
    syncSize();
    const img = imgRef.current;
    if (!img) return undefined;
    const ro = new ResizeObserver(() => syncSize());
    ro.observe(img);
    return () => ro.disconnect();
  }, [syncSize, imageSrc]);

  const refPx = useMemo(() => (refA && refB ? dist(refA, refB) : null), [refA, refB]);
  const cmPerPx = useMemo(() => {
    const cm = Number(refCm);
    if (!refPx || refPx <= 0 || !Number.isFinite(cm) || cm <= 0) return null;
    return cm / refPx;
  }, [refCm, refPx]);

  const live = useMemo(() => computeLive(points, cmPerPx), [points, cmPerPx]);
  const allFour = KEYS.every((k) => points[k]);
  const canContinue = Boolean(
    !busy && ready && refA && refB && refPx && refPx > 2 && cmPerPx && allFour && live?.scaled,
  );

  useEffect(() => {
    pointsRef.current = points;
  }, [points]);

  useEffect(() => () => {
    if (debugTimer.current) clearTimeout(debugTimer.current);
  }, []);

  /** Body height vertical: follow current A End point when placed (manual or suggest). */
  const bodyHeightAxis = useMemo(() => {
    const fromApi = aEndFromLineDict(aEndLine);
    const fromPose = aEndFromKeypoints(
      pose?.selected_detection?.keypoints,
      pose?.selected_detection?.bbox,
    );
    const aEndPt = points.A_end_withers;
    if (aEndPt) {
      const groundY = fromApi?.ground.y
        ?? fromPose?.ground.y
        ?? (aEndPt.y + Math.max(120, (natural.h || 400) * 0.25));
      return {
        aEnd: { x: aEndPt.x, y: aEndPt.y },
        ground: { x: aEndPt.x, y: groundY },
        label: 'A End',
      };
    }
    if (fromApi) return fromApi;
    return fromPose;
  }, [aEndLine, points.A_end_withers, pose?.selected_detection, natural.h]);

  const markManual = (id: KeyName, p: ImagePoint) => {
    const prev = pointsRef.current;
    const next = {
      ...prev,
      [id]: {
        ...p,
        name: id,
        status: 'manual_corrected' as const,
        method: prev[id]?.method || pointDetector || 'manual',
      },
    };
    pointsRef.current = next;
    setPoints(next);
    setPointStatuses((s) => ({ ...s, [id]: 'manual_corrected' }));
  };

  const refreshDebug = useCallback(async (pts?: Partial<Record<KeyName, PointMeta>>) => {
    if (!runId || busy) return;
    const source = pts || pointsRef.current;
    if (!KEYS.every((k) => source[k])) return;
    const keypoints: Record<string, PointMeta> = {};
    for (const k of KEYS) {
      const p = source[k];
      if (!p) return;
      keypoints[k] = p;
    }
    const aEndPt = source.A_end_withers;
    const groundY = aEndLine?.ground?.[1]
      ?? bodyHeightAxis?.ground.y
      ?? (aEndPt ? aEndPt.y + Math.max(120, (natural.h || 400) * 0.25) : null);
    const aePayload = aEndPt && groundY != null
      ? {
        detected: true,
        a_end: [aEndPt.x, aEndPt.y] as [number, number],
        ground: [aEndPt.x, groundY] as [number, number],
        p1: [aEndPt.x, aEndPt.y] as [number, number],
        p2: [aEndPt.x, groundY] as [number, number],
        label: 'A End',
        line_label: 'Body height',
      }
      : (aEndLine || null);

    // If another drag finishes while in-flight, re-run after
    if (debugInFlight.current) {
      debugPending.current = true;
      return;
    }
    debugInFlight.current = true;
    try {
      const res = await stageFourPointDebug(runId, {
        keypoints,
        a_end_line: aePayload,
        lower_chest_guide: lowerChestGuide,
        head_anchor: headAnchor,
        head_direction: headDirection || null,
      });
      if (res.a_end_line?.detected) {
        setAEndLine(res.a_end_line);
      }
      // JPEG saved on server for export; live preview is client SVG (no img URL needed)
    } catch {
      // Optional JPEG export only — live SVG preview does not depend on this
    } finally {
      debugInFlight.current = false;
      if (debugPending.current) {
        debugPending.current = false;
        void refreshDebug(pointsRef.current);
      }
    }
  }, [
    runId, busy, aEndLine, bodyHeightAxis, lowerChestGuide, headAnchor, headDirection, natural.h,
  ]);

  const scheduleDebugRefresh = useCallback((immediate = false) => {
    if (debugTimer.current) clearTimeout(debugTimer.current);
    if (immediate) {
      void refreshDebug(pointsRef.current);
      return;
    }
    debugTimer.current = setTimeout(() => {
      void refreshDebug(pointsRef.current);
    }, 180);
  }, [refreshDebug]);

  const onBgDown = (e: ReactPointerEvent) => {
    if (!ready || busy) return;
    const img = imgRef.current;
    if (!img) return;
    const p = clientToImg(e.clientX, e.clientY, img);
    if (!p) return;

    if (mode === 'reference') {
      if (refPhase === 'A') {
        setRefA(p);
        setRefPhase('B');
        setSelected('refA');
      } else if (refPhase === 'B') {
        setRefB(p);
        setRefPhase('done');
        setSelected('refB');
        setMode('anatomy');
      } else if (selected === 'refA') setRefA(p);
      else if (selected === 'refB') setRefB(p);
      return;
    }

    if (selected && KEYS.includes(selected as KeyName)) {
      markManual(selected as KeyName, p);
      scheduleDebugRefresh(true);
      return;
    }
    if (placeNext) {
      const placed = placeNext;
      markManual(placed, p);
      setSelected(placed);
      const idx = KEYS.indexOf(placed);
      setPlaceNext(idx >= 0 && idx < KEYS.length - 1 ? KEYS[idx + 1] : null);
      scheduleDebugRefresh(true);
    }
  };

  const startDrag = (id: KeyName | 'refA' | 'refB', e: ReactPointerEvent) => {
    e.stopPropagation();
    e.preventDefault();
    dragging.current = id;
    setSelected(id);
    const move = (ev: PointerEvent) => {
      const img = imgRef.current;
      if (!img || !dragging.current) return;
      const p = clientToImg(ev.clientX, ev.clientY, img);
      if (!p) return;
      if (dragging.current === 'refA') setRefA(p);
      else if (dragging.current === 'refB') setRefB(p);
      else markManual(dragging.current as KeyName, p);
      // Live SVG preview follows points state; JPEG file only on release
    };
    const up = () => {
      const dragged = dragging.current;
      dragging.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      if (dragged && KEYS.includes(dragged as KeyName)) {
        // Optional saved JPEG only (non-blocking); UI preview is client-side
        scheduleDebugRefresh(true);
      }
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const suggest = useCallback(async () => {
    if (!runId || busy) return;
    setError('');
    setSuggestMsg('Running Cow Morpho Heuristic…');
    try {
      const payload = headDirection ? { head_direction: headDirection } : {};
      const res = await stageFourPointSuggest(runId, payload);
      if (res.head_direction_required) {
        setHeadRequired(true);
        setSuggestMsg(res.reason || 'Select head direction (Left / Right), then retry Auto-suggest.');
        return;
      }
      if (!res.available || !res.keypoints) {
        setHeadRequired(Boolean(res.head_direction_required));
        setSuggestMsg(res.reason || 'Auto-suggest unavailable — place points manually.');
        return;
      }
      if (res.inferred_head_direction && !headDirection) {
        setHeadDirection(res.inferred_head_direction as 'left' | 'right');
      }
      setHeadDetected(res.head_detected ?? null);
      setHeadAnchor(res.head_anchor?.detected ? res.head_anchor : null);
      setLowerChestGuide(res.lower_chest_guide_line?.detected ? res.lower_chest_guide_line : null);
      setAEndLine(res.a_end_line || null);
      const next: Partial<Record<KeyName, PointMeta>> = {};
      const statuses: Partial<Record<KeyName, string>> = {};
      for (const k of KEYS) {
        const pt = res.keypoints[k];
        if (pt) {
          next[k] = {
            x: pt.x,
            y: pt.y,
            name: k,
            status: (pt.status as PointMeta['status']) || 'auto_suggested',
            method: pt.method || res.point_detector || 'CowMorphoHeuristic',
            confidence: pt.confidence,
            anatomy_label: pt.anatomy_label,
            source_keypoint: pt.source_keypoint,
          };
          statuses[k] = next[k]!.status || 'auto_suggested';
        }
      }
      setPoints(next);
      pointsRef.current = next;
      setPointStatuses(statuses);
      setPointDetector(res.point_detector || res.method || 'CowMorphoHeuristic');
      setMode('anatomy');
      setPlaceNext(null);
      setHeadRequired(false);
      setSuggestMsg('Suggested by Cow Morpho Heuristic — drag to correct if needed.');
      // Server still writes four_point_morpho_debug.jpg; live preview uses SVG below
    } catch (err) {
      setSuggestMsg(err instanceof Error ? err.message : 'Suggest failed');
    }
  }, [runId, busy, headDirection]);

  const autoSuggestRan = useRef(false);
  useEffect(() => {
    if (!runId || busy || autoSuggestRan.current) return;
    autoSuggestRan.current = true;
    void suggest();
  }, [runId, busy, suggest]);

  const submit = () => {
    if (!canContinue || !refA || !refB || !refPx || !live?.scaled) return;
    const cm = Number(refCm);
    const four: Record<string, PointMeta> = {};
    for (const k of KEYS) {
      const p = points[k];
      if (!p) return;
      four[k] = {
        x: p.x,
        y: p.y,
        name: k,
        status: (pointStatuses[k] as PointMeta['status']) || p.status || 'manual_corrected',
        method: p.method || pointDetector || 'manual',
        anatomy_label: p.anatomy_label,
        source_keypoint: p.source_keypoint,
      };
    }
    onApply({
      reference_px: refPx,
      reference_cm: cm,
      point_a: refA,
      point_b: refB,
      four_points: four,
    });
  };

  const vw = Math.max(1, natural.w);
  const vh = Math.max(1, natural.h);
  const r = Math.max(6, Math.min(vw, vh) * 0.012);
  const facingDisplay = formatHeadFacing(headDirection || undefined);
  const headArrowDx = headDirection === 'right' ? r * 3 : headDirection === 'left' ? -r * 3 : 0;

  const shoulderRef = useMemo(() => {
    const markers = measure?.measurements?.segmentation?.shoulder_markers;
    if (!markers?.length) return null;
    const name = headDirection === 'right'
      ? 'right_shoulder'
      : headDirection === 'left'
        ? 'left_shoulder'
        : null;
    if (!name) return markers[0] ?? null;
    return markers.find((m) => m.name === name) ?? null;
  }, [measure, headDirection]);

  const strokeMain = Math.max(2, Math.min(vw, vh) * 0.004);
  const strokeGuide = Math.max(2, Math.min(vw, vh) * 0.003);
  const fontSm = Math.max(9, Math.min(vw, vh) * 0.018);

  /** Shared guides + A/B lines + points (interactive or read-only live preview). */
  const overlayContent = (interactive: boolean) => (
    <>
      {editContourPath ? (
        <path
          d={editContourPath}
          fill="none"
          stroke={BODY_OUTLINE_STROKE}
          strokeWidth={outlineStrokeWidth(vw, vh)}
          strokeLinejoin="round"
          strokeLinecap="round"
          pointerEvents="none"
        />
      ) : null}
      {interactive ? (
        <rect x={0} y={0} width={vw} height={vh} fill="transparent" onPointerDown={onBgDown} />
      ) : null}
      {refA && refB ? (
        <g pointerEvents="none">
          <line x1={refA.x} y1={refA.y} x2={refB.x} y2={refB.y} stroke="#1565c0" strokeWidth={strokeGuide} strokeDasharray="6 4" />
          {refPx != null ? (
            <text
              x={(refA.x + refB.x) / 2}
              y={(refA.y + refB.y) / 2 - r}
              fill="#0d47a1"
              fontSize={Math.max(10, Math.min(vw, vh) * 0.02)}
              fontWeight={700}
              textAnchor="middle"
            >
              {`Reference: ${refPx.toFixed(2)} px`}
            </text>
          ) : null}
        </g>
      ) : null}
      {lowerChestGuide?.detected && lowerChestGuide.p1 && lowerChestGuide.p2 ? (
        <g pointerEvents="none">
          <line
            x1={lowerChestGuide.p1[0]}
            y1={lowerChestGuide.p1[1]}
            x2={lowerChestGuide.p2[0]}
            y2={lowerChestGuide.p2[1]}
            stroke="#00bcd4"
            strokeWidth={strokeGuide}
            strokeDasharray="8 5"
          />
          <circle
            cx={lowerChestGuide.mid ? lowerChestGuide.mid[0] : (lowerChestGuide.p1[0] + lowerChestGuide.p2[0]) / 2}
            cy={lowerChestGuide.mid ? lowerChestGuide.mid[1] : (lowerChestGuide.p1[1] + lowerChestGuide.p2[1]) / 2}
            r={r * 0.85}
            fill="#ffeb3b"
            stroke="#000"
            strokeWidth={1.5}
          />
          <text
            x={lowerChestGuide.mid ? lowerChestGuide.mid[0] : (lowerChestGuide.p1[0] + lowerChestGuide.p2[0]) / 2}
            y={(lowerChestGuide.mid ? lowerChestGuide.mid[1] : (lowerChestGuide.p1[1] + lowerChestGuide.p2[1]) / 2) - r}
            fill="#00838f"
            fontSize={fontSm}
            fontWeight={700}
            textAnchor="middle"
          >
            {lowerChestGuide.label || 'Lower chest'}
          </text>
        </g>
      ) : null}
      {bodyHeightAxis ? (
        <g pointerEvents="none">
          <line
            x1={bodyHeightAxis.aEnd.x}
            y1={bodyHeightAxis.aEnd.y}
            x2={bodyHeightAxis.ground.x}
            y2={bodyHeightAxis.ground.y}
            stroke="#43a047"
            strokeWidth={Math.max(2, Math.min(vw, vh) * 0.0035)}
          />
          <circle cx={bodyHeightAxis.ground.x} cy={bodyHeightAxis.ground.y} r={r * 0.75} fill="#43a047" />
          <circle cx={bodyHeightAxis.aEnd.x} cy={bodyHeightAxis.aEnd.y} r={r * 0.95} fill="#e53935" stroke="#b71c1c" strokeWidth={2} />
          <text x={bodyHeightAxis.aEnd.x + r * 1.4} y={bodyHeightAxis.aEnd.y - r * 0.4} fill="#b71c1c" fontSize={fontSm} fontWeight={700}>
            A End
          </text>
          <text
            x={(bodyHeightAxis.aEnd.x + bodyHeightAxis.ground.x) / 2 + r}
            y={(bodyHeightAxis.aEnd.y + bodyHeightAxis.ground.y) / 2}
            fill="#2e7d32"
            fontSize={fontSm}
            fontWeight={700}
          >
            Body height
          </text>
          <text x={bodyHeightAxis.ground.x + r * 1.4} y={bodyHeightAxis.ground.y + r * 0.6} fill="#2e7d32" fontSize={Math.max(8, Math.min(vw, vh) * 0.016)} fontWeight={600}>
            Ground
          </text>
        </g>
      ) : null}
      {points.A_start_lower_chest && points.A_end_withers ? (
        <line
          x1={points.A_start_lower_chest.x}
          y1={points.A_start_lower_chest.y}
          x2={points.A_end_withers.x}
          y2={points.A_end_withers.y}
          stroke="#2f7d32"
          strokeWidth={strokeMain}
          pointerEvents="none"
        />
      ) : null}
      {points.B_start_tail_head && points.B_end_shoulder_region ? (
        <line
          x1={points.B_start_tail_head.x}
          y1={points.B_start_tail_head.y}
          x2={points.B_end_shoulder_region.x}
          y2={points.B_end_shoulder_region.y}
          stroke="#ef6c00"
          strokeWidth={strokeMain}
          pointerEvents="none"
        />
      ) : null}
      {refA ? (
        <circle
          cx={refA.x}
          cy={refA.y}
          r={r}
          fill="#fff"
          stroke="#1565c0"
          strokeWidth={2}
          style={interactive ? { cursor: 'grab' } : undefined}
          onPointerDown={interactive ? (e) => startDrag('refA', e) : undefined}
          pointerEvents={interactive ? 'auto' : 'none'}
        />
      ) : null}
      {refB ? (
        <circle
          cx={refB.x}
          cy={refB.y}
          r={r}
          fill="#fff"
          stroke="#1565c0"
          strokeWidth={2}
          style={interactive ? { cursor: 'grab' } : undefined}
          onPointerDown={interactive ? (e) => startDrag('refB', e) : undefined}
          pointerEvents={interactive ? 'auto' : 'none'}
        />
      ) : null}
      {KEYS.map((k) => {
        const p = points[k];
        if (!p) return null;
        const color = k.startsWith('A') ? '#2f7d32' : '#ef6c00';
        return (
          <g key={`${interactive ? 'edit' : 'preview'}-${k}`} pointerEvents={interactive ? 'auto' : 'none'}>
            {interactive ? (
              <circle cx={p.x} cy={p.y} r={r * 2.2} fill="transparent" style={{ cursor: 'grab' }} onPointerDown={(e) => startDrag(k, e)} />
            ) : null}
            <circle cx={p.x} cy={p.y} r={r} fill="#fff" stroke={color} strokeWidth={2} pointerEvents="none" />
            <text x={p.x + r * 1.4} y={p.y - r} fill={color} fontSize={fontSm} fontWeight={700} pointerEvents="none">
              {pointDisplayLabel(k, p, headDirection)}
            </text>
          </g>
        );
      })}
      {shoulderRef ? (
        <g pointerEvents="none">
          <circle cx={shoulderRef.x} cy={shoulderRef.y} r={r} fill="#ffeb3b" stroke="#000" strokeWidth={2} />
          <text x={shoulderRef.x + r * 1.4} y={shoulderRef.y + r * 1.6} fill="#f9a825" fontSize={fontSm} fontWeight={700}>
            {forwardShoulderRefLabel(headDirection)}
          </text>
        </g>
      ) : null}
      {headAnchor?.detected && headAnchor.x != null && headAnchor.y != null ? (
        <g pointerEvents="none">
          <circle cx={headAnchor.x} cy={headAnchor.y} r={r * 1.2} fill="#e040fb" stroke="#7b1fa2" strokeWidth={2} />
          {headArrowDx !== 0 ? (
            <line
              x1={headAnchor.x}
              y1={headAnchor.y}
              x2={headAnchor.x + headArrowDx}
              y2={headAnchor.y}
              stroke="#e040fb"
              strokeWidth={strokeGuide}
            />
          ) : null}
          <text x={headAnchor.x + r * 1.5} y={headAnchor.y - r * 0.8} fill="#e040fb" fontSize={Math.max(11, Math.min(vw, vh) * 0.022)} fontWeight={700}>
            Head
          </text>
        </g>
      ) : null}
    </>
  );

  return (
    <div className="live-card">
      <h2 className="live-title">Cow Morpho Heuristic</h2>
      <p className="live-sub">
        Place a known-length reference scale, then review or adjust four body points.
        Points are suggested automatically from the cow mask and pose.
      </p>
      <div className="meta-chips" style={{ marginBottom: '0.75rem' }}>
        <span className="chip">Model: Cow Morpho Heuristic</span>
        <span className="chip">Status: Experimental</span>
        <span className="chip">
          Point detector:
          {' '}
          {pointDetector || 'CowMorphoHeuristic'}
        </span>
        {headDetected != null ? (
          <span className="chip">Head detected: {headDetected ? 'Yes' : 'No'}</span>
        ) : null}
        <span className="chip">Head facing: {facingDisplay}</span>
        {imageSrc && (measure?.files?.['body_outline.jpg'] || measure?.files?.['measure_outline.jpg'] || editContourPath) ? (
          <span className="chip">Body outline: shown</span>
        ) : null}
      </div>

      <div className="options-grid" style={{ marginBottom: '0.75rem' }}>
        <div className="option-block">
          <label htmlFor="head-dir">Head direction</label>
          <select
            id="head-dir"
            value={headDirection}
            disabled={busy}
            onChange={(e) => setHeadDirection(e.target.value as 'left' | 'right' | '')}
            style={{ marginTop: '0.4rem', width: '100%', padding: '0.5rem', borderRadius: 8 }}
          >
            <option value="">Auto (from pose)</option>
            <option value="left">Left</option>
            <option value="right">Right</option>
          </select>
          {headRequired ? (
            <p className="live-sub" style={{ marginTop: 6 }}>
              Pose could not infer facing — pick Left or Right, then Auto-suggest again.
            </p>
          ) : null}
        </div>
      </div>

      <div className="btn-row" style={{ marginBottom: '0.75rem' }}>
        <button type="button" className={`btn ${mode === 'reference' ? 'btn-primary' : 'btn-ghost'}`} disabled={busy} onClick={() => setMode('reference')}>
          1. Reference scale
        </button>
        <button type="button" className={`btn ${mode === 'anatomy' ? 'btn-primary' : 'btn-ghost'}`} disabled={busy || !refA || !refB} onClick={() => setMode('anatomy')}>
          2. Four keypoints
        </button>
        <button type="button" className="btn btn-ghost" disabled={busy || !runId} onClick={() => void suggest()}>
          Re-run auto-suggest
        </button>
      </div>
      {suggestMsg ? <div className="measure-formula" role="status">{suggestMsg}</div> : null}

      <div className="morpho-dual-stage">
        <div className="morpho-dual-stage__pane">
          <p className="live-sub" style={{ marginBottom: '0.35rem' }}>
            <strong>Editor</strong>
            {' '}
            — drag points here
          </p>
          <div className="calib-image-wrap reference-image-wrapper" style={{ maxWidth: '100%' }}>
            {imageSrc ? (
              <img ref={imgRef} src={imageSrc} alt="Cow for diagonal formula" onLoad={syncSize} draggable={false} />
            ) : (
              <div className="error-box">Image unavailable</div>
            )}
            {ready ? (
              <svg
                className="calib-overlay ref-overlay"
                width={display.w}
                height={display.h}
                viewBox={`0 0 ${vw} ${vh}`}
                preserveAspectRatio="none"
                style={{ touchAction: 'none', pointerEvents: 'auto' }}
              >
                {overlayContent(true)}
              </svg>
            ) : null}
          </div>
        </div>

        {ready && imageSrc ? (
          <div className="morpho-dual-stage__pane">
            <p className="live-sub" style={{ marginBottom: '0.35rem' }}>
              <strong>Live preview</strong>
              {' '}
              — updates as you drag (same points as editor)
            </p>
            <div className="calib-image-wrap reference-image-wrapper" style={{ maxWidth: '100%', pointerEvents: 'none' }}>
              <img src={imageSrc} alt="Live Morpho preview" draggable={false} />
              <svg
                className="calib-overlay ref-overlay"
                width={display.w}
                height={display.h}
                viewBox={`0 0 ${vw} ${vh}`}
                preserveAspectRatio="none"
                style={{ pointerEvents: 'none' }}
              >
                {overlayContent(false)}
              </svg>
            </div>
          </div>
        ) : null}
      </div>

      <div className="scale-form" style={{ marginTop: '1rem' }}>
        <label htmlFor="diag-ref-cm">
          Known reference length (cm)
          <input
            id="diag-ref-cm"
            type="number"
            min="0"
            step="any"
            value={refCm}
            disabled={busy}
            onChange={(e) => setRefCm(e.target.value)}
            placeholder="e.g. 20"
          />
        </label>
      </div>

      <div className="ref-live-panel">
        <div><strong>Reference px:</strong> {refPx != null ? refPx.toFixed(2) : '—'}</div>
        <div><strong>cm/px:</strong> {cmPerPx != null ? cmPerPx.toFixed(6) : '—'}</div>
        {KEYS.map((k) => (
          <div key={k}>
            <strong>{pointDisplayLabel(k, points[k], headDirection)}:</strong>
            {' '}
            {points[k] ? `x ${points[k]!.x.toFixed(1)}, y ${points[k]!.y.toFixed(1)}` : 'not placed'}
            {' · '}
            <em>{pointStatuses[k] || points[k]?.status || '—'}</em>
            {' · '}
            <button type="button" className="btn btn-ghost" style={{ padding: '0.2rem 0.5rem' }} disabled={busy} onClick={() => { setMode('anatomy'); setSelected(k); setPlaceNext(k); }}>
              {points[k] ? 'Reposition' : 'Place'}
            </button>
            <span style={{ marginLeft: 8, color: 'var(--fb-muted)', fontSize: '0.85rem' }}>{DESCRIPTIONS[k]}</span>
          </div>
        ))}
      </div>

      <div className="measure-formula" style={{ marginTop: '0.85rem' }}>
        <div style={{ fontWeight: 700, marginBottom: '0.35rem' }}>Smartphone Diagonal Formula</div>
        <div><code>weight_lb = (C² × B) / 300</code></div>
        <div><code>C = 2 × A</code></div>
        <div><code>A, B in inches</code> (from px × cm/px ÷ 2.54)</div>
        {live ? (
          <>
            <hr style={{ border: 'none', borderTop: '1px solid var(--fb-border)', margin: '0.55rem 0' }} />
            <div><strong>A:</strong> {live.A_px.toFixed(2)} px{live.scaled ? ` · ${live.A_cm!.toFixed(2)} cm · ${(live.A_cm! / CM_PER_INCH).toFixed(2)} in` : ''}</div>
            <div><strong>B:</strong> {live.B_px.toFixed(2)} px{live.scaled ? ` · ${live.B_cm!.toFixed(2)} cm · ${(live.B_cm! / CM_PER_INCH).toFixed(2)} in` : ''}</div>
            {live.scaled ? (
              <>
                <div><strong>C = 2·A:</strong> {live.C_cm!.toFixed(2)} cm · {live.C_in!.toFixed(2)} in</div>
                <div><strong>Weight:</strong> {live.weight_lb!.toFixed(2)} lb · {live.weight_kg!.toFixed(2)} kg</div>
                <div style={{ marginTop: 4, color: 'var(--fb-muted)', fontSize: '0.9rem' }}>
                  = (C_in² × B_in) / 300
                </div>
              </>
            ) : (
              <div style={{ marginTop: 6 }}>Enter reference A/B + length (cm) to convert inches and weight.</div>
            )}
          </>
        ) : (
          <div style={{ marginTop: 6 }}>Place all 4 body points to see live A/B values. No mock values.</div>
        )}
        <div style={{ marginTop: 6 }}>Method: Smartphone Diagonal Formula · Model: Cow Morpho Heuristic</div>
      </div>

      {error ? <div className="error-box" role="alert">{error}</div> : null}

      <div className="btn-row" style={{ marginTop: '1rem' }}>
        <button type="button" className="btn btn-primary" disabled={!canContinue} onClick={submit}>
          Confirm scale + 4 points
        </button>
      </div>
    </div>
  );
}
