import { useEffect, useMemo, useState } from 'react';
import { fileUrl } from '../../api';
import type { MeasureStageResponse, PoseStageResponse, SegmentStageResponse } from '../../types';
import { OverlayLabel } from '../../utils/overlayLabels';
import {
  lowerChestFromKeypoints,
  lowerChestFromLineDict,
} from '../../utils/lowerChestLine';
import { aEndFromKeypoints, aEndFromLineDict } from '../../utils/aEndLine';
import {
  contourPathFromBodyContour,
  outlineImageFromFiles,
} from '../../utils/bodyOutline';
import LiveProcessingLayout from './LiveProcessingLayout';
import BodyOutlineSvg from './BodyOutlineSvg';

interface Props {
  segment: SegmentStageResponse | null;
  pose?: PoseStageResponse | null;
  measure?: MeasureStageResponse | null;
  onDone?: () => void;
  onRetry?: () => void;
}

export default function SegmentationStep({ segment, pose, measure, onDone, onRetry }: Props) {
  const [view, setView] = useState<'original' | 'overlay' | 'mask' | 'regions'>('overlay');
  const [area, setArea] = useState(0);
  const [peri, setPeri] = useState(0);
  const [torso, setTorso] = useState(0);
  const [upperChest, setUpperChest] = useState(0);
  const [lowerChestArea, setLowerChestArea] = useState(0);

  const skipped = segment?.status === 'skipped';
  const failed = segment?.status === 'failed';
  const seg = segment?.segmentation;
  const targetArea = seg?.body_pixel_area || 0;
  const targetPeri = seg?.body_perimeter_px || 0;
  const targetTorso = seg?.torso_pixel_area || 0;
  const targetUpper = seg?.upper_chest_pixel_area || 0;
  const targetLower = seg?.lower_chest_pixel_area || 0;
  const tailDetected = Boolean(seg?.tail_anchor?.detected);
  const shoulderCount = seg?.shoulder_markers?.length ?? 0;

  const imageSize = segment?.image_size || pose?.image_size || { width: 1, height: 1 };
  const w = imageSize.width || 1;
  const h = imageSize.height || 1;
  const fs = Math.max(w, h) * 0.022;
  const strokeW = Math.max(w, h) * 0.0035;

  const lowerChest = useMemo(() => {
    const fromSeg = lowerChestFromLineDict(seg?.lower_chest_line);
    if (fromSeg) return fromSeg;
    return lowerChestFromKeypoints(pose?.selected_detection?.keypoints);
  }, [seg?.lower_chest_line, pose?.selected_detection?.keypoints]);

  const aEndAxis = useMemo(() => {
    const fromSeg = aEndFromLineDict(seg?.a_end_line);
    if (fromSeg) return fromSeg;
    return aEndFromKeypoints(
      pose?.selected_detection?.keypoints,
      pose?.selected_detection?.bbox,
    );
  }, [seg?.a_end_line, pose?.selected_detection?.keypoints, pose?.selected_detection?.bbox]);

  const contourPath = useMemo(
    () => contourPathFromBodyContour(
      segment?.body_contour || measure?.body_contour || pose?.body_contour,
    ),
    [segment?.body_contour, measure?.body_contour, pose?.body_contour],
  );
  const outlineImg = outlineImageFromFiles(segment?.files)
    || outlineImageFromFiles(measure?.files)
    || outlineImageFromFiles(pose?.files);

  const showGuide = (view === 'overlay' || view === 'regions') && Boolean(lowerChest || aEndAxis);

  useEffect(() => {
    if (!segment) return undefined;
    if (failed) return undefined;
    if (skipped) {
      const t = window.setTimeout(() => onDone?.(), 900);
      return () => window.clearTimeout(t);
    }
    let frame = 0;
    const frames = 24;
    const id = window.setInterval(() => {
      frame += 1;
      const t = frame / frames;
      setArea(Math.round(targetArea * t));
      setPeri(Math.round(targetPeri * t));
      setTorso(Math.round(targetTorso * t));
      setUpperChest(Math.round(targetUpper * t));
      setLowerChestArea(Math.round(targetLower * t));
      if (frame >= frames) {
        window.clearInterval(id);
      }
    }, 40);
    return () => window.clearInterval(id);
  }, [segment, skipped, failed, targetArea, targetPeri, targetTorso, targetUpper, targetLower, onDone]);

  if (!segment) {
    return (
      <LiveProcessingLayout
        title="Segmenting body"
        status="Analysing cow body shape..."
        scanning
      >
        <div className="live-image-stage" />
      </LiveProcessingLayout>
    );
  }

  if (skipped) {
    return (
      <LiveProcessingLayout
        title="Segmentation"
        status="Skipped — continuing with pose measurements"
        footer={(
          <div className="btn-row">
            <button type="button" className="btn btn-primary" onClick={onDone}>Continue</button>
          </div>
        )}
      >
        <p className="live-sub">{segment.message}</p>
      </LiveProcessingLayout>
    );
  }

  if (failed) {
    return (
      <LiveProcessingLayout
        title="Segmentation"
        status="Segmentation failed"
        footer={(
          <div className="btn-row">
            {onRetry ? (
              <button type="button" className="btn btn-ghost" onClick={onRetry}>Retry</button>
            ) : null}
            <button type="button" className="btn btn-primary" onClick={onDone}>
              Continue Without Segmentation
            </button>
          </div>
        )}
      >
        <div className="error-box">{segment.error || 'No cow mask found'}</div>
      </LiveProcessingLayout>
    );
  }

  const src = fileUrl(
    view === 'mask'
      ? segment.files?.['segmentation_mask.png']
      : view === 'original'
        ? (segment.files?.['body_outline.jpg']
          || segment.files?.['measure_outline.jpg']
          || segment.files?.['original_image.jpg'])
        : segment.files?.['segmentation_overlay.png'],
  );
  const originalWithOutline = view === 'original' && !src && outlineImg ? outlineImg : null;
  const usingBakedOutlineBase = view === 'original' && Boolean(
    segment.files?.['body_outline.jpg'] || segment.files?.['measure_outline.jpg'] || originalWithOutline,
  );
  // Overlay/regions PNGs already have OpenCV red contours; baked original has outline JPEG.
  const showOutlineSvg = view !== 'mask'
    && view !== 'overlay'
    && view !== 'regions'
    && !usingBakedOutlineBase;

  return (
    <LiveProcessingLayout
      title="Body segmentation"
      subtitle="Green = upper chest band, orange = lower chest band. Cyan = Lower chest line (shoulder→hip center)."
      status="Analysing cow body shape..."
      footer={(
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={onDone}>
            Continue
          </button>
        </div>
      )}
    >
      <div className="seg-tabs">
        {(['original', 'overlay', 'regions', 'mask'] as const).map((v) => (
          <button
            key={v}
            type="button"
            className={view === v ? 'active' : ''}
            onClick={() => setView(v)}
          >
            {v === 'regions' ? 'Regions' : v[0].toUpperCase() + v.slice(1)}
          </button>
        ))}
      </div>
      <div className="live-image-stage" style={{ position: 'relative' }}>
        {(src || originalWithOutline) ? (
          <img src={src || originalWithOutline || undefined} alt={`Segmentation ${view}`} />
        ) : null}
        {showOutlineSvg ? (
          <BodyOutlineSvg path={contourPath} width={w} height={h} zIndex={2} />
        ) : null}
        {showGuide && lowerChest ? (
          <svg
            className="overlay-svg"
            viewBox={`0 0 ${w} ${h}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ pointerEvents: 'none' }}
          >
            <line
              x1={lowerChest.p1.x}
              y1={lowerChest.p1.y}
              x2={lowerChest.p2.x}
              y2={lowerChest.p2.y}
              stroke="#00bcd4"
              strokeWidth={strokeW}
              strokeDasharray={`${strokeW * 3} ${strokeW * 2}`}
            />
            <circle cx={lowerChest.p1.x} cy={lowerChest.p1.y} r={fs * 0.28} fill="#00bcd4" />
            <circle cx={lowerChest.p2.x} cy={lowerChest.p2.y} r={fs * 0.28} fill="#00bcd4" />
            <circle
              cx={lowerChest.mid.x}
              cy={lowerChest.mid.y}
              r={fs * 0.38}
              fill="#ffeb3b"
              stroke="#000"
              strokeWidth={1.5}
            />
            <OverlayLabel
              x={lowerChest.mid.x + fs * 0.4}
              y={lowerChest.mid.y - fs * 0.5}
              text={lowerChest.label}
              fontSize={fs * 0.8}
              fill="#ffffff"
              stroke="#00838f"
            />
          </svg>
        ) : null}
        {showGuide && aEndAxis ? (
          <svg
            className="overlay-svg"
            viewBox={`0 0 ${w} ${h}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ pointerEvents: 'none' }}
          >
            <line
              x1={aEndAxis.aEnd.x}
              y1={aEndAxis.aEnd.y}
              x2={aEndAxis.ground.x}
              y2={aEndAxis.ground.y}
              stroke="#43a047"
              strokeWidth={strokeW}
            />
            <circle cx={aEndAxis.ground.x} cy={aEndAxis.ground.y} r={fs * 0.28} fill="#43a047" />
            <circle
              cx={aEndAxis.aEnd.x}
              cy={aEndAxis.aEnd.y}
              r={fs * 0.42}
              fill="#e53935"
              stroke="#b71c1c"
              strokeWidth={2}
            />
            <OverlayLabel
              x={aEndAxis.aEnd.x + fs * 0.45}
              y={aEndAxis.aEnd.y - fs * 0.35}
              text="A End"
              fontSize={fs * 0.8}
              fill="#ffffff"
              stroke="#b71c1c"
            />
            <OverlayLabel
              x={(aEndAxis.aEnd.x + aEndAxis.ground.x) / 2 + fs * 0.35}
              y={(aEndAxis.aEnd.y + aEndAxis.ground.y) / 2}
              text="Body height"
              fontSize={fs * 0.75}
              fill="#ffffff"
              stroke="#2e7d32"
            />
            <OverlayLabel
              x={aEndAxis.ground.x + fs * 0.45}
              y={aEndAxis.ground.y + fs * 0.45}
              text="Ground"
              fontSize={fs * 0.7}
              fill="#ffffff"
              stroke="#2e7d32"
            />
          </svg>
        ) : null}
      </div>
      <div className="meta-chips" style={{ marginTop: '0.75rem' }}>
        <span className="chip">Body area: {area.toLocaleString()} px²</span>
        <span className="chip">Torso area: {torso.toLocaleString()} px²</span>
        <span className="chip">Upper chest: {upperChest.toLocaleString()} px²</span>
        <span className="chip">Lower chest band: {lowerChestArea.toLocaleString()} px²</span>
        <span className="chip">Perimeter: {peri.toLocaleString()} px</span>
        <span className="chip">Tail: {tailDetected ? 'detected' : 'not detected'}</span>
        <span className="chip">Shoulders: {shoulderCount} marked</span>
        {lowerChest ? <span className="chip">Lower chest line: shown</span> : null}
        {aEndAxis ? <span className="chip">Body height: shown</span> : null}
        {contourPath || outlineImg ? <span className="chip">Body outline: shown</span> : null}
      </div>
      <p className="live-sub">
        Upper and lower chest bands split the mask using pose keypoints (shoulders, neck, elbows).
        The Lower chest line is shoulder center → hip center; yellow mid point is A Start.
        Green vertical is Body height; red-bordered upper tip is A End (extended Back top).
        Tail marker is the tail-head join on the body mask (not the tail tip); forward shoulder only.
      </p>
    </LiveProcessingLayout>
  );
}
