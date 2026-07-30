import { fileUrl } from '../../api';
import type { DetectStageResponse } from '../../types';
import LiveProcessingLayout from './LiveProcessingLayout';

interface Props {
  detect: DetectStageResponse | null;
  localPreview: string | null;
  busy: boolean;
}

export default function CowDetectionStep({ detect, localPreview, busy }: Props) {
  const original = fileUrl(detect?.files?.['original_image.jpg']);
  const detection = fileUrl(detect?.files?.['detection_image.jpg']);
  const done = !!detect;
  const primary = detect?.detections?.find((d) => d.cow_id === detect.selected_cow_id);

  return (
    <LiveProcessingLayout
      title="Detecting cow"
      subtitle="Local YOLO finds each cow before pose estimation begins."
      status={done
        ? `Detection complete — ${detect.num_cows_detected} cow(s) found`
        : 'Detecting cow...'}
      scanning={!done && busy}
    >
      <div className="live-image-stage">
        {!done ? (
          <>
            {localPreview || original ? (
              <img src={localPreview || original || undefined} alt="Scanning" />
            ) : null}
            <div className="scan-line" />
          </>
        ) : (
          <img className="bbox-reveal" src={detection || undefined} alt="Cow detections" />
        )}
      </div>
      {done ? (
        <div className="meta-chips" style={{ marginTop: '0.85rem' }}>
          <span className="chip">Cows: {detect.num_cows_detected}</span>
          <span className="chip">
            Primary: #{detect.selected_cow_id}
            {' '}
            ({(((primary?.bbox_confidence) || 0) * 100).toFixed(0)}% conf)
          </span>
        </div>
      ) : null}
    </LiveProcessingLayout>
  );
}
