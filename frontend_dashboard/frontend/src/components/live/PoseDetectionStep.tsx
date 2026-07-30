import { useEffect, useMemo, useRef, useState } from 'react';
import { fileUrl } from '../../api';
import type { KeypointPoint, PoseStageResponse } from '../../types';
import { formatHeadFacing } from '../../utils/formatPartName';
import { OverlayLabel } from '../../utils/overlayLabels';
import { lowerChestFromKeypoints } from '../../utils/lowerChestLine';
import { aEndFromKeypoints, aEndFromLineDict } from '../../utils/aEndLine';
import {
  contourPathFromBodyContour,
  outlineImageFromFiles,
} from '../../utils/bodyOutline';
import LiveProcessingLayout from './LiveProcessingLayout';
import KeypointRevealAnimation from './KeypointRevealAnimation';
import BodyOutlineSvg from './BodyOutlineSvg';

interface Props {
  pose: PoseStageResponse | null;
  onRevealDone?: () => void;
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

export default function PoseDetectionStep({ pose, onRevealDone }: Props) {
  const [visibleCount, setVisibleCount] = useState(0);
  const [done, setDone] = useState(false);
  const doneRef = useRef(false);
  const original = fileUrl(pose?.files?.['original_image.jpg']);
  const poseImg = fileUrl(pose?.files?.['pose_image.jpg']);
  const outlineImg = outlineImageFromFiles(pose?.files);
  const baseImg = outlineImg || original;
  const contourPath = useMemo(
    () => contourPathFromBodyContour(pose?.body_contour),
    [pose?.body_contour],
  );
  const detection = pose?.selected_detection;
  const imageSize = pose?.image_size || { width: 1, height: 1 };
  const w = imageSize.width || 1;
  const h = imageSize.height || 1;
  const fs = Math.max(w, h) * 0.022;

  const headMarker = useMemo(
    () => computeHeadMarker(detection?.keypoints),
    [detection?.keypoints],
  );

  const lowerChest = useMemo(
    () => lowerChestFromKeypoints(detection?.keypoints),
    [detection?.keypoints],
  );

  const aEndAxis = useMemo(() => {
    const fromApi = aEndFromLineDict(pose?.a_end_line);
    if (fromApi) return fromApi;
    return aEndFromKeypoints(detection?.keypoints, detection?.bbox);
  }, [pose?.a_end_line, detection?.keypoints, detection?.bbox]);

  const headDetected = pose?.head_detected ?? Boolean(headMarker);
  const headFacing = pose?.head_direction ?? null;

  const orderedPoints = useMemo(() => {
    if (!detection?.keypoints || !pose?.keypoint_groups) return [];
    const groups = pose.keypoint_groups;
    const order = [
      ...(groups.head || []),
      ...(groups.upper_body || []),
      ...(groups.front_legs || []),
      ...(groups.rear_body || []),
    ];
    return order
      .map((name) => {
        const p = detection.keypoints[name];
        if (!p || p.confidence <= 0) return null;
        return { name, ...p };
      })
      .filter((p): p is NonNullable<typeof p> => p != null);
  }, [detection, pose]);

  // Show Lower chest once shoulders + hips have been revealed
  const showLowerChest = useMemo(() => {
    if (!lowerChest || !orderedPoints.length) return false;
    if (done) return true;
    const needed = new Set(['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']);
    const shown = orderedPoints.slice(0, visibleCount).map((p) => p.name);
    return [...needed].filter((n) => shown.includes(n)).length >= 2
      && shown.some((n) => n.includes('shoulder'))
      && shown.some((n) => n.includes('hip'));
  }, [lowerChest, orderedPoints, visibleCount, done]);

  const showAEnd = useMemo(() => {
    if (!aEndAxis || !orderedPoints.length) return false;
    if (done) return true;
    const shown = orderedPoints.slice(0, visibleCount).map((p) => p.name);
    return shown.some((n) => n.includes('shoulder') || n === 'neck')
      && shown.some((n) => n.includes('hoof'));
  }, [aEndAxis, orderedPoints, visibleCount, done]);

  useEffect(() => {
    doneRef.current = false;
    if (!pose) return undefined;
    if (!orderedPoints.length) {
      setDone(true);
      if (!doneRef.current) {
        doneRef.current = true;
        onRevealDone?.();
      }
      return undefined;
    }
    setVisibleCount(0);
    setDone(false);
    let i = 0;
    const id = window.setInterval(() => {
      i += 1;
      setVisibleCount(i);
      if (i >= orderedPoints.length) {
        window.clearInterval(id);
        setDone(true);
        if (!doneRef.current) {
          doneRef.current = true;
          onRevealDone?.();
        }
      }
    }, 120);
    return () => window.clearInterval(id);
  }, [pose, orderedPoints, onRevealDone]);

  if (!pose) {
    return (
      <LiveProcessingLayout
        title="Detecting pose"
        subtitle="Estimating 17 anatomical keypoints on the selected cow."
        status="Detecting cow body points..."
        scanning
      >
        <div className="live-image-stage" />
      </LiveProcessingLayout>
    );
  }

  const arrowDx = headFacing === 'right' ? fs * 2.5 : headFacing === 'left' ? -fs * 2.5 : 0;
  const strokeW = Math.max(w, h) * 0.0035;

  return (
    <LiveProcessingLayout
      title="Pose keypoints"
      subtitle="Keypoints appear by group: head → upper body → front legs → rear."
      status={(
        <span>
          Detected body points:&nbsp;
          <span className="kpt-counter">
            {visibleCount}/{orderedPoints.length || pose.total_points || 17}
          </span>
        </span>
      )}
    >
      <div className="live-image-stage" style={{ position: 'relative' }}>
        <img src={baseImg || poseImg || undefined} alt="Pose base" style={{ opacity: done ? 0.35 : 1 }} />
        <BodyOutlineSvg path={outlineImg ? null : contourPath} width={w} height={h} zIndex={3} />
        <KeypointRevealAnimation
          points={orderedPoints.slice(0, visibleCount)}
          imageSize={imageSize}
        />
        {headMarker && !done ? (
          <svg
            className="overlay-svg"
            viewBox={`0 0 ${w} ${h}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ pointerEvents: 'none' }}
          >
            <circle cx={headMarker.x} cy={headMarker.y} r={fs * 0.5} fill="#e040fb" stroke="#7b1fa2" strokeWidth={2} />
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
        {done && poseImg ? (
          <img
            src={poseImg}
            alt="Full pose"
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              opacity: 0.95,
              zIndex: 2,
            }}
          />
        ) : null}
        {showLowerChest && lowerChest ? (
          <svg
            className="overlay-svg"
            viewBox={`0 0 ${w} ${h}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ pointerEvents: 'none', zIndex: 5 }}
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
        {showAEnd && aEndAxis ? (
          <svg
            className="overlay-svg"
            viewBox={`0 0 ${w} ${h}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ pointerEvents: 'none', zIndex: 6 }}
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
        <span className="chip">Head detected: {headDetected ? 'Yes' : 'No'}</span>
        <span className="chip">Head facing: {formatHeadFacing(headFacing)}</span>
        {lowerChest ? <span className="chip">Lower chest: shown</span> : null}
        {aEndAxis ? <span className="chip">Body height: shown</span> : null}
        {outlineImg || contourPath ? <span className="chip">Body outline: shown</span> : null}
      </div>
      {pose.low_confidence_keypoints?.length ? (
        <div className="meta-chips" style={{ marginTop: '0.5rem' }}>
          <span className="chip">
            Low confidence: {pose.low_confidence_keypoints.join(', ')}
          </span>
        </div>
      ) : null}
    </LiveProcessingLayout>
  );
}
