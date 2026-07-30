import type { ImageSize, PixelCalculation, PixelCalcSegment } from '../../types';
import { formatMeasurementName, formatPartName } from '../../utils/formatPartName';
import { OverlayLabel } from '../../utils/overlayLabels';

interface Props {
  calc?: PixelCalculation;
  imageSize?: ImageSize;
  history?: PixelCalculation[];
}

function fontSize(w: number, h: number) {
  return Math.max(w, h) * 0.022;
}

function endpointLabel(
  x: number,
  y: number,
  partName: string | null | undefined,
  fs: number,
  offsetY: number,
  key?: string,
) {
  if (!partName) return null;
  return (
    <OverlayLabel
      key={key || `lbl-${partName}-${x}-${y}`}
      x={x + fs * 0.4}
      y={y + offsetY}
      text={formatPartName(partName)}
      fontSize={fs * 0.85}
    />
  );
}

function drawSegment(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  w: number,
  h: number,
  strong: boolean,
  key: string,
) {
  const sw = Math.max(w, h) * (strong ? 0.004 : 0.0025);
  return (
    <line
      key={key}
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={strong ? '#2f7d32' : '#81c784'}
      strokeWidth={sw}
    />
  );
}

function drawCalc(c: PixelCalculation, w: number, h: number, strong: boolean) {
  if (!c.available) return null;
  const fs = fontSize(w, h);
  const r = Math.max(w, h) * 0.006;
  const segments = (c.segments || []) as PixelCalcSegment[];

  if (segments.length > 0) {
    const joints: Array<{ x: number; y: number; name: string }> = [];
    const first = segments[0];
    if (first) joints.push({ x: first.x1, y: first.y1, name: first.from });
    for (const seg of segments) {
      joints.push({ x: seg.x2, y: seg.y2, name: seg.to });
    }
    const mx = joints.reduce((s, j) => s + j.x, 0) / joints.length;
    const my = joints.reduce((s, j) => s + j.y, 0) / joints.length;

    return (
      <g key={`${c.name}-${strong ? 'a' : 'h'}`} opacity={strong ? 1 : 0.35}>
        {segments.map((seg, i) => drawSegment(seg.x1, seg.y1, seg.x2, seg.y2, w, h, strong, `${c.name}-seg-${i}`))}
        {joints.map((j) => (
          <circle key={`${c.name}-${j.name}`} cx={j.x} cy={j.y} r={r} fill="#fff" stroke="#1b5e20" strokeWidth={2} />
        ))}
        {strong
          ? joints.map((j, i) => endpointLabel(j.x, j.y, j.name, fs, -fs * 0.3, `${c.name}-lbl-${j.name}-${i}`))
          : null}
        {strong && c.result_px != null ? (
          <>
            <OverlayLabel
              x={mx}
              y={my - fs * 1.2}
              text={formatMeasurementName(c.name)}
              fontSize={fs}
              anchor="middle"
            />
            <OverlayLabel
              x={mx}
              y={my + fs * 0.2}
              text={`${c.result_px.toFixed(1)} px`}
              fontSize={fs * 0.9}
              anchor="middle"
            />
          </>
        ) : null}
      </g>
    );
  }

  if (c.x1 == null || c.y1 == null || c.x2 == null || c.y2 == null) return null;
  const mx = (c.x1 + c.x2) / 2;
  const my = (c.y1 + c.y2) / 2;

  return (
    <g key={`${c.name}-${strong ? 'a' : 'h'}`} opacity={strong ? 1 : 0.35}>
      {drawSegment(c.x1, c.y1, c.x2, c.y2, w, h, strong, `${c.name}-line`)}
      <circle cx={c.x1} cy={c.y1} r={r} fill="#fff" stroke="#1b5e20" strokeWidth={2} />
      <circle cx={c.x2} cy={c.y2} r={r} fill="#fff" stroke="#1b5e20" strokeWidth={2} />
      {strong ? (
        <>
          {endpointLabel(c.x1, c.y1, c.start_point, fs, -fs * 0.3)}
          {endpointLabel(c.x2, c.y2, c.end_point, fs, fs * 1.1)}
          {c.result_px != null ? (
            <>
              <OverlayLabel
                x={mx}
                y={my - fs * 1.2}
                text={formatMeasurementName(c.name)}
                fontSize={fs}
                anchor="middle"
              />
              <OverlayLabel
                x={mx}
                y={my + fs * 0.2}
                text={`${c.result_px.toFixed(1)} px`}
                fontSize={fs * 0.9}
                anchor="middle"
              />
            </>
          ) : null}
        </>
      ) : null}
    </g>
  );
}

export default function AnimatedMeasurementLine({
  calc,
  imageSize,
  history = [],
}: Props) {
  const w = imageSize?.width || 1;
  const h = imageSize?.height || 1;

  return (
    <svg className="overlay-svg" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet">
      {history.map((c) => drawCalc(c, w, h, false))}
      {calc ? drawCalc(calc, w, h, true) : null}
    </svg>
  );
}
