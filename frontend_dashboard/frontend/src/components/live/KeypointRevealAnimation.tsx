import type { ImageSize, KeypointPoint } from '../../types';
import { formatPartName } from '../../utils/formatPartName';
import { OverlayLabel } from '../../utils/overlayLabels';

interface Point extends KeypointPoint {
  name: string;
}

interface Props {
  points?: Point[];
  imageSize?: ImageSize;
}

const HEAD_NAMES = new Set(['nose', 'left_eye', 'right_eye']);

export default function KeypointRevealAnimation({
  points = [],
  imageSize = { width: 1, height: 1 },
}: Props) {
  const w = imageSize.width || 1;
  const h = imageSize.height || 1;
  const fs = Math.max(w, h) * 0.022;
  const byName = Object.fromEntries(points.map((p) => [p.name, p]));
  const pairs: Array<[string, string]> = [
    ['nose', 'left_eye'], ['nose', 'right_eye'], ['left_eye', 'right_eye'],
    ['nose', 'neck'], ['neck', 'left_shoulder'], ['neck', 'right_shoulder'],
    ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_front_hoof'],
    ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_front_hoof'],
    ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
    ['left_hip', 'left_knee'], ['left_knee', 'left_back_hoof'],
    ['right_hip', 'right_knee'], ['right_knee', 'right_back_hoof'],
    ['left_hip', 'tail_root'], ['right_hip', 'tail_root'],
  ];

  const headPoints = points.filter((p) => HEAD_NAMES.has(p.name));
  const headBadge = headPoints.length > 0
    ? {
        x: headPoints.reduce((s, p) => s + p.x, 0) / headPoints.length,
        y: headPoints.reduce((s, p) => s + p.y, 0) / headPoints.length,
      }
    : null;

  return (
    <svg
      className="overlay-svg"
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {pairs
        .filter(([a, b]) => byName[a] && byName[b])
        .map(([a, b]) => (
          <line
            key={`${a}-${b}`}
            x1={byName[a].x}
            y1={byName[a].y}
            x2={byName[b].x}
            y2={byName[b].y}
            stroke="#81c784"
            strokeWidth={Math.max(w, h) * 0.003}
            opacity={0.85}
          />
        ))}
      {points.map((p) => {
        const low = (p.confidence ?? 0) < 0.3;
        const r = Math.max(w, h) * 0.008;
        return (
          <g key={p.name}>
            <circle
              cx={p.x}
              cy={p.y}
              r={r}
              fill={low ? '#f4c430' : '#7CFF8A'}
              stroke={low ? '#8a6d00' : '#1b5e20'}
              strokeWidth={Math.max(w, h) * 0.002}
              opacity={low ? 0.75 : 1}
            />
            <OverlayLabel
              x={p.x + r + fs * 0.2}
              y={p.y + fs * 0.15}
              text={formatPartName(p.name)}
              fontSize={fs * 0.75}
              stroke={low ? '#5a4800' : '#1b5e20'}
            />
          </g>
        );
      })}
      {headBadge ? (
        <g>
          <rect
            x={headBadge.x - fs * 1.1}
            y={headBadge.y - fs * 2.4}
            width={fs * 2.2}
            height={fs * 1.1}
            rx={fs * 0.2}
            fill="#7b1fa2"
            opacity={0.9}
          />
          <OverlayLabel
            x={headBadge.x}
            y={headBadge.y - fs * 1.55}
            text="Head"
            fontSize={fs * 0.8}
            anchor="middle"
            fill="#ffffff"
            stroke="#4a148c"
          />
        </g>
      ) : null}
    </svg>
  );
}
