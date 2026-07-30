import { downloadUrl, fileUrl } from '../../api';
import type { CompleteReport } from '../../types';

interface Props {
  open: boolean;
  onClose: () => void;
  report: CompleteReport | null;
}

export default function TechnicalDetailsDrawer({ open, onClose, report }: Props) {
  if (!open || !report) return null;

  const det = report.selected_detection;
  const kpts = det?.keypoints || {};
  const calcs = report.measurements?.pixel_calculations || [];
  const features = report.normalized_features || [];

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} role="presentation" />
      <aside className="drawer" role="dialog" aria-label="Technical details">
        <div className="drawer-header">
          <strong>Step-by-step details</strong>
          <button type="button" className="btn btn-ghost" onClick={onClose}>Close</button>
        </div>
        <div className="drawer-body">
          <div className="drawer-section">
            <h3>Downloads</h3>
            <div className="btn-row">
              <a className="btn btn-ghost" href={downloadUrl(report.run_id, 'json')} download>JSON</a>
              <a className="btn btn-ghost" href={downloadUrl(report.run_id, 'csv')} download>CSV</a>
              <a className="btn btn-ghost" href={downloadUrl(report.run_id, 'image')} download>Image</a>
            </div>
          </div>

          <div className="drawer-section">
            <h3>Result images</h3>
            <div className="meta-chips">
              {Object.entries(report.files || {})
                .filter(([k]) => /\.(jpg|png)$/i.test(k))
                .map(([name, path]) => (
                  <a key={name} className="chip" href={fileUrl(path) || '#'} target="_blank" rel="noreferrer">
                    {name}
                  </a>
                ))}
            </div>
          </div>

          <div className="drawer-section">
            <h3>Detections</h3>
            <pre>{JSON.stringify(report.detections, null, 2)}</pre>
          </div>

          <div className="drawer-section">
            <h3>Keypoints (cow #{report.selected_cow_id})</h3>
            <table className="tech-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>X</th>
                  <th>Y</th>
                  <th>Conf</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(kpts).map(([name, p]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{p.x}</td>
                    <td>{p.y}</td>
                    <td>{p.confidence}</td>
                    <td>{p.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="drawer-section">
            <h3>Pixel formulas</h3>
            {calcs.map((c) => (
              <div key={c.name} className="measure-formula" style={{ marginBottom: 8 }}>
                <strong>{c.name}</strong>
                <div>{c.formula}</div>
                <div>{c.substituted}</div>
                <div>→ {c.result_px != null ? `${c.result_px} px` : 'n/a'}</div>
              </div>
            ))}
          </div>

          <div className="drawer-section">
            <h3>Segmentation</h3>
            <pre>{JSON.stringify(report.measurements?.segmentation ?? report.steps?.segmentation, null, 2)}</pre>
          </div>

          <div className="drawer-section">
            <h3>Scale / calibration</h3>
            <pre>{JSON.stringify(report.scale, null, 2)}</pre>
          </div>

          <div className="drawer-section">
            <h3>Normalised features</h3>
            <table className="tech-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {features.map((f) => (
                  <tr key={f.name}>
                    <td>{f.name}</td>
                    <td>{f.value == null ? 'missing' : Number(f.value).toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="drawer-section">
            <h3>Prediction</h3>
            <pre>{JSON.stringify(report.weight, null, 2)}</pre>
            {report?.weight?.heuristic && typeof report.weight.heuristic === 'object' ? (
              <div className="measure-formula" style={{ marginTop: 8 }}>
                <strong>Heuristic baseline (h5model) — technical compare only</strong>
                <div>
                  Weight:
                  {' '}
                  {(report.weight.heuristic as { weight_kg?: number }).weight_kg != null
                    ? Number((report.weight.heuristic as { weight_kg?: number }).weight_kg).toFixed(2)
                    : '—'}
                  {' '}
                  kg — not used for the main Cow Morpho Heuristic result.
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </aside>
    </>
  );
}
