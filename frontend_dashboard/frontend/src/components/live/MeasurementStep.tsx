import { useEffect, useMemo, useState } from 'react';
import { fileUrl } from '../../api';
import type { KeypointPoint, MeasureStageResponse, PoseStageResponse } from '../../types';
import { formatHeadFacing, formatMeasurementName } from '../../utils/formatPartName';
import { OverlayLabel } from '../../utils/overlayLabels';
import {
  contourPathFromBodyContour,
  outlineImageFromFiles,
} from '../../utils/bodyOutline';
import LiveProcessingLayout from './LiveProcessingLayout';
import AnimatedMeasurementLine from './AnimatedMeasurementLine';
import BodyOutlineSvg from './BodyOutlineSvg';

interface Props {
  measure: MeasureStageResponse | null;
  pose?: PoseStageResponse | null;
  onContinue?: () => void;
}

const HEAD_KPT_NAMES = ['nose', 'left_eye', 'right_eye'] as const;

function computeHeadMarker(keypoints: Record<string, KeypointPoint> | undefined) {
  if (!keypoints) return null;
  const pts = HEAD_KPT_NAMES
    .map((n) => keypoints[n])
    .filter((p) => p && p.confidence > 0);
  if (!pts.length) return null;
  return {
    x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
    y: pts.reduce((s, p) => s + p.y, 0) / pts.length,
  };
}

export default function MeasurementStep({ measure, pose, onContinue }: Props) {
  const seq = measure?.measurement_sequence || [];
  const [index, setIndex] = useState(0);
  const [auto, setAuto] = useState(true);
  const outlineImg = outlineImageFromFiles(measure?.files) || outlineImageFromFiles(pose?.files);
  const original = fileUrl(measure?.files?.['original_image.jpg']);
  const baseImg = outlineImg || original;
  const imageSize = measure?.image_size || { width: 1, height: 1 };
  const current = seq[index];
  const w = imageSize.width || 1;
  const h = imageSize.height || 1;
  const fs = Math.max(w, h) * 0.022;

  const keypoints = pose?.selected_detection?.keypoints;
  const headMarker = useMemo(() => computeHeadMarker(keypoints), [keypoints]);
  const headFacing = pose?.head_direction ?? null;
  const arrowDx = headFacing === 'right' ? fs * 2.5 : headFacing === 'left' ? -fs * 2.5 : 0;

  const contourPath = useMemo(
    () => contourPathFromBodyContour(measure?.body_contour || pose?.body_contour),
    [measure?.body_contour, pose?.body_contour],
  );

  const hasOutline = Boolean(outlineImg || contourPath);

  useEffect(() => {
    if (!measure || !auto || !seq.length) return undefined;
    if (index >= seq.length - 1) {
      setAuto(false);
      return undefined;
    }
    const t = window.setTimeout(() => setIndex((i) => Math.min(i + 1, seq.length - 1)), 900);
    return () => window.clearTimeout(t);
  }, [measure, auto, index, seq.length]);

  if (!measure) {
    return (
      <LiveProcessingLayout
        title="Measuring body"
        status="Computing distances and cow outline…"
        scanning
      >
        <div className="live-image-stage" />
      </LiveProcessingLayout>
    );
  }

  return (
    <LiveProcessingLayout
      title="Body measurements"
      subtitle="Each distance uses real keypoint coordinates — no invented points."
      status={`${index + 1} / ${seq.length || 1}: ${formatMeasurementName(current?.name)}`}
      footer={(
        <div className="btn-row">
          <button
            type="button"
            className="btn btn-ghost"
            disabled={index <= 0}
            onClick={() => { setAuto(false); setIndex((i) => i - 1); }}
          >
            Previous
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={index >= seq.length - 1}
            onClick={() => { setAuto(false); setIndex((i) => i + 1); }}
          >
            Next
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => { setIndex(0); setAuto(true); }}
          >
            Auto Play
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => setAuto(false)}
          >
            Pause
          </button>
          {onContinue ? (
            <button type="button" className="btn btn-primary" onClick={onContinue}>
              Continue to Body Analysis
            </button>
          ) : null}
        </div>
      )}
    >
      <div className="live-image-stage" style={{ position: 'relative' }}>
        {baseImg ? <img src={baseImg} alt="Measurement base" /> : null}
        <BodyOutlineSvg path={outlineImg ? null : contourPath} width={w} height={h} zIndex={2} />
        <AnimatedMeasurementLine
          calc={current}
          imageSize={imageSize}
          history={seq.slice(0, index)}
        />
        {headMarker ? (
          <svg
            className="overlay-svg"
            viewBox={`0 0 ${w} ${h}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ pointerEvents: 'none', zIndex: 4 }}
          >
            <circle
              cx={headMarker.x}
              cy={headMarker.y}
              r={fs * 0.5}
              fill="#e040fb"
              stroke="#7b1fa2"
              strokeWidth={2}
            />
            {arrowDx !== 0 ? (
              <line
                x1={headMarker.x}
                y1={headMarker.y}
                x2={headMarker.x + arrowDx}
                y2={headMarker.y}
                stroke="#e040fb"
                strokeWidth={Math.max(w, h) * 0.003}
              />
            ) : null}
            <OverlayLabel
              x={headMarker.x + fs * 0.5}
              y={headMarker.y - fs * 0.6}
              text="HEAD"
              fontSize={fs * 0.85}
              fill="#ffffff"
              stroke="#7b1fa2"
            />
          </svg>
        ) : null}
      </div>
      <div className="chip-row" style={{ marginTop: 8 }}>
        {hasOutline ? <span className="chip">Body outline: shown</span> : null}
        {measure.body_contour_error ? (
          <span className="chip">Outline: {measure.body_contour_error}</span>
        ) : null}
        {headMarker ? (
          <>
            <span className="chip">HEAD: shown</span>
            <span className="chip">Head facing: {formatHeadFacing(headFacing)}</span>
          </>
        ) : null}
      </div>
      {current ? (
        <div className="measure-formula">
          <div><strong>{formatMeasurementName(current.name)}</strong></div>
          <div>
            Start point = ({current.x1 ?? '—'}, {current.y1 ?? '—'})
          </div>
          <div>
            End point = ({current.x2 ?? '—'}, {current.y2 ?? '—'})
          </div>
          <div style={{ marginTop: 6 }}>{current.formula}</div>
          <div style={{ marginTop: 6 }}>{current.substituted}</div>
          <div style={{ marginTop: 6 }}>
            Result:&nbsp;
            {current.result_px != null ? `${current.result_px} px` : 'unavailable'}
          </div>
        </div>
      ) : null}
    </LiveProcessingLayout>
  );
}
