import { fileUrl } from '../../api';
import type { DetectStageResponse } from '../../types';
import LiveProcessingLayout from './LiveProcessingLayout';

interface Props {
  detect: DetectStageResponse | null;
  onSelect: (cowId: number) => void;
  busy: boolean;
}

export default function CowSelectionStep({ detect, onSelect, busy }: Props) {
  const detection = fileUrl(detect?.files?.['detection_image.jpg']);
  const original = fileUrl(detect?.files?.['original_image.jpg']);
  const cows = detect?.detections || [];
  const size = detect?.image_size || { width: 1, height: 1 };

  return (
    <LiveProcessingLayout
      title="Select a cow"
      subtitle="Multiple animals were found. Tap a box or card to continue."
      status="Selection required to continue"
    >
      <div className="live-image-stage" style={{ position: 'relative' }}>
        {(detection || original) ? (
          <img src={detection || original || undefined} alt="Multiple cows" />
        ) : null}
        <svg
          className="overlay-svg"
          viewBox={`0 0 ${size.width} ${size.height}`}
          preserveAspectRatio="xMidYMid meet"
          style={{ pointerEvents: 'auto' }}
        >
          {cows.map((c) => {
            const [x1, y1, x2, y2] = c.bbox;
            return (
              <g key={c.cow_id} style={{ cursor: busy ? 'wait' : 'pointer' }}>
                <rect
                  x={x1}
                  y={y1}
                  width={Math.max(1, x2 - x1)}
                  height={Math.max(1, y2 - y1)}
                  fill="rgba(47,125,50,0.15)"
                  stroke="#2f7d32"
                  strokeWidth={Math.max(size.width, size.height) * 0.004}
                  onClick={() => { if (!busy) onSelect(c.cow_id); }}
                />
                <text
                  x={x1 + 8}
                  y={Math.max(24, y1 - 8)}
                  fill="#1b5e20"
                  fontSize={Math.max(size.width, size.height) * 0.028}
                  fontWeight={700}
                  onClick={() => { if (!busy) onSelect(c.cow_id); }}
                >
                  {`Cow ${c.cow_id + 1}`}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="cow-grid">
        {cows.map((c) => (
          <button
            key={c.cow_id}
            type="button"
            className={`cow-card ${c.cow_id === detect?.selected_cow_id ? 'selected' : ''}`}
            disabled={busy}
            onClick={() => onSelect(c.cow_id)}
          >
            <strong>Cow {c.cow_id + 1}</strong>
            <div style={{ color: 'var(--fb-muted)', fontSize: '0.85rem', marginTop: 4 }}>
              Confidence {(c.bbox_confidence * 100).toFixed(1)}%
            </div>
            <div style={{ color: 'var(--fb-muted)', fontSize: '0.85rem' }}>
              Area {Math.round(c.bbox_area).toLocaleString()} px²
            </div>
          </button>
        ))}
      </div>
    </LiveProcessingLayout>
  );
}
