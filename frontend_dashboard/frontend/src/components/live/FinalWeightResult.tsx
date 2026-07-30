import { downloadUrl, fileUrl } from '../../api';
import type { CompleteReport } from '../../types';
import LiveProcessingLayout from './LiveProcessingLayout';

interface Props {
  report: CompleteReport | null;
  onAgain: () => void;
  onOpenDetails: () => void;
}

function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(Number(v))) return '—';
  return Number(v).toFixed(digits);
}

export default function FinalWeightResult({ report, onAgain, onOpenDetails }: Props) {
  const final = report?.final || {};
  const weight = final.weight_kg;
  const mode = final.selected_model || 'Cow Morpho Heuristic';
  const pointDetector = final.point_detector || 'CowMorphoHeuristic';
  const scaleUsed = Boolean(final.reference_scale_used || report?.scale?.reference_scale_used || report?.scale?.provided);
  const annotated = fileUrl(
    report?.files?.['body_outline.jpg']
      || report?.files?.['measure_outline.jpg']
      || report?.files?.['measurements_image.jpg']
      || report?.files?.['pose_image.jpg']
      || report?.files?.['detection_image.jpg'],
  );
  const original = fileUrl(report?.files?.['original_image.jpg']);
  const cm = report?.measurements?.measurements_cm;
  const lengthCm = final.body_length_cm ?? cm?.body_length ?? null;
  const heightCm = final.body_height_cm ?? cm?.body_height ?? null;
  const chestCm = final.chest_depth_proxy_cm ?? cm?.chest_depth_proxy ?? null;

  return (
    <LiveProcessingLayout
      title="Estimated Cow Weight"
      subtitle="Guided detection complete."
    >
      <div className="final-weight">
        <div className="kg">
          {weight != null ? Number(weight).toFixed(2) : '—'}
          <span className="unit"> kg</span>
        </div>
      </div>

      <div className="live-image-stage" style={{ marginTop: '1rem' }}>
        {(annotated || original) ? (
          <img src={annotated || original || undefined} alt="Annotated cow" />
        ) : null}
      </div>

      <div className="meta-chips" style={{ justifyContent: 'center', marginTop: '1rem' }}>
        <span className="chip">Cow #{final.selected_cow_id ?? '—'}</span>
        <span className="chip">Detected cows: {final.num_cows ?? '—'}</span>
        <span className="chip">Keypoints: {final.detected_points ?? '—'}/17</span>
        <span className="chip">
          Reference scale:
          {' '}
          {scaleUsed ? 'Applied' : 'Not provided'}
        </span>
        <span className="chip">Segmentation: {final.segmentation_status || '—'}</span>
        <span className="chip">Model: {mode}</span>
        {pointDetector ? (
          <span className="chip">Point detector: {pointDetector}</span>
        ) : null}
      </div>

      <div className="measure-formula" style={{ marginTop: '1rem', textAlign: 'left' }}>
        <div>
          <strong>Body length:</strong>
          {' '}
          {fmtNum(final.body_length_px)}
          {' '}
          px
          {scaleUsed ? ` · ${fmtNum(lengthCm)} cm` : ' (measurements remain in pixels)'}
        </div>
        <div>
          <strong>Body height:</strong>
          {' '}
          {fmtNum(final.body_height_px)}
          {' '}
          px
          {scaleUsed ? ` · ${fmtNum(heightCm)} cm` : ''}
        </div>
        <div>
          <strong>Chest-depth proxy:</strong>
          {' '}
          {fmtNum(final.chest_depth_proxy_px)}
          {' '}
          px
          {scaleUsed ? ` · ${fmtNum(chestCm)} cm` : ''}
        </div>
        {scaleUsed && final.cm_per_px != null ? (
          <div style={{ marginTop: 6 }}>
            Scale:
            {' '}
            {fmtNum(final.cm_per_px, 6)}
            {' '}
            cm/px
          </div>
        ) : null}
      </div>

      {(final.diagonal_method || final.A_px != null || report?.weight?.smartphone_diagonal) ? (
        <div className="measure-formula" style={{ marginTop: '1rem', textAlign: 'left' }}>
          <div>
            <strong>{final.diagonal_method || 'Smartphone Diagonal Formula'}</strong>
            {' · '}
            {final.diagonal_status || 'Experimental'}
          </div>
          {final.point_detector ? (
            <div>
              Point detector:
              {' '}
              {final.point_detector}
            </div>
          ) : null}
          <div>
            A:
            {' '}
            {fmtNum(final.A_px)}
            {' '}
            px ·
            {' '}
            {fmtNum(final.A_cm)}
            {' '}
            cm
          </div>
          <div>
            B:
            {' '}
            {fmtNum(final.B_px)}
            {' '}
            px ·
            {' '}
            {fmtNum(final.B_cm)}
            {' '}
            cm
          </div>
          <div>
            Estimated heart girth C:
            {' '}
            {fmtNum(final.estimated_heart_girth_C_cm)}
            {' '}
            cm ·
            {' '}
            {fmtNum(final.estimated_heart_girth_C_in)}
            {' '}
            in
          </div>
          <div>
            Weight:
            {' '}
            {fmtNum(final.weight_lb)}
            {' '}
            lb ·
            {' '}
            {fmtNum(final.weight_kg)}
            {' '}
            kg
          </div>
        </div>
      ) : null}

      <p className="disclaimer">
        Estimated result only — not a replacement for a calibrated livestock weighing scale.
      </p>

      {final.warnings?.length ? (
        <div
          className="error-box"
          style={{ background: '#fff8e6', color: '#6b4f00', borderColor: '#f0d78c' }}
        >
          {final.warnings.join(' · ')}
        </div>
      ) : null}

      <div className="btn-row" style={{ justifyContent: 'center' }}>
        <button type="button" className="btn btn-primary" onClick={onAgain}>
          Analyse Another Cow
        </button>
        <button type="button" className="btn btn-ghost" onClick={onOpenDetails}>
          View Step-by-Step Details
        </button>
        {report?.run_id ? (
          <>
            <a className="btn btn-ghost" href={downloadUrl(report.run_id, 'image')} download>
              Download Final Image
            </a>
            <a className="btn btn-ghost" href={downloadUrl(report.run_id, 'json')} download>
              Download JSON Report
            </a>
            <a className="btn btn-ghost" href={downloadUrl(report.run_id, 'csv')} download>
              Download CSV Report
            </a>
          </>
        ) : null}
      </div>
    </LiveProcessingLayout>
  );
}
