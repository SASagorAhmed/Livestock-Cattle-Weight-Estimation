import { useEffect, useMemo, useState } from 'react';
import type { NormalizedFeature, ScaleStageResponse } from '../../types';
import LiveProcessingLayout from './LiveProcessingLayout';

interface Props {
  features?: NormalizedFeature[];
  measurePx?: Record<string, number | null | undefined>;
  scale: ScaleStageResponse | null;
  onDone?: () => void;
}

const DISPLAY_ORDER = [
  'body_length',
  'body_height',
  'body_length / body_height',
  'chest_depth_proxy',
  'body_area',
  'torso_area',
  'body_perimeter',
  'front_leg',
  'back_leg',
];

export default function FeaturePreparationStep({
  features = [],
  measurePx = {},
  scale,
  onDone,
}: Props) {
  const [count, setCount] = useState(0);
  const [openTech, setOpenTech] = useState(false);

  const displayList = useMemo(() => {
    const fromNorm = features.map((f) => ({
      key: f.name,
      label: f.name,
      value: f.value,
      missing: f.value == null,
      detail: f,
    }));

    const extras = [
      { key: 'body_length', label: 'body length', value: measurePx.body_length ?? null },
      { key: 'body_height', label: 'body height', value: measurePx.body_height ?? null },
      { key: 'chest_depth_proxy', label: 'chest-depth proxy', value: measurePx.chest_depth_proxy ?? null },
      {
        key: 'front_leg',
        label: 'useful front-leg measurement',
        value: measurePx.left_front_leg_length ?? measurePx.right_front_leg_length ?? null,
      },
      {
        key: 'back_leg',
        label: 'useful back-leg measurement',
        value: measurePx.left_back_leg_length ?? measurePx.right_back_leg_length ?? null,
      },
    ].map((e) => ({
      ...e,
      missing: e.value == null,
      detail: null as NormalizedFeature | null,
    }));

    const merged = [...extras];
    for (const f of fromNorm) {
      if (!merged.some((m) => m.key === f.key || m.label === f.label)) {
        merged.push({
          key: f.key,
          label: f.label,
          value: f.value,
          missing: f.missing,
          detail: f.detail,
        });
      }
    }

    merged.sort((a, b) => {
      const ai = DISPLAY_ORDER.findIndex((n) => a.key.includes(n) || a.label.includes(n));
      const bi = DISPLAY_ORDER.findIndex((n) => b.key.includes(n) || b.label.includes(n));
      return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
    });
    return merged;
  }, [features, measurePx]);

  useEffect(() => {
    setCount(0);
    if (!displayList.length) {
      const t = window.setTimeout(() => onDone?.(), 500);
      return () => window.clearTimeout(t);
    }
    let i = 0;
    const id = window.setInterval(() => {
      i += 1;
      setCount(i);
      if (i >= displayList.length) {
        window.clearInterval(id);
        window.setTimeout(() => onDone?.(), 450);
      }
    }, 180);
    return () => window.clearInterval(id);
  }, [displayList, onDone]);

  return (
    <LiveProcessingLayout
      title="Preparing features"
      subtitle="Preparing body features for weight estimation..."
      status={`Features ready: ${Math.min(count, displayList.length)} / ${displayList.length || 0}`}
    >
      {scale?.scale ? (
        <div className="meta-chips" style={{ marginBottom: '0.75rem' }}>
          <span className="chip">
            {scale.scale.provided
              ? `Scale ${scale.scale.cm_per_px} cm/px`
              : 'No reference scale was provided. Pixel measurements and normalised features will be used.'}
          </span>
        </div>
      ) : null}

      <ul className="feature-list">
        {displayList.slice(0, count).map((f) => (
          <li key={f.key}>
            <span>{f.label}</span>
            <strong style={{ color: f.missing ? 'var(--fb-danger)' : undefined }}>
              {f.missing ? 'missing' : Number(f.value).toFixed(4)}
            </strong>
          </li>
        ))}
      </ul>

      <div className="btn-row">
        <button type="button" className="btn btn-ghost" onClick={() => setOpenTech((v) => !v)}>
          {openTech ? 'Hide' : 'View'} technical details
        </button>
      </div>
      {openTech ? (
        <div className="measure-formula">
          {features.map((f) => (
            <div key={`t-${f.name}`} style={{ marginBottom: 8 }}>
              <strong>{f.name}</strong>
              <div>{f.formula}</div>
              <div>
                {f.numerator} / {f.denominator}
                {' '}
                =
                {' '}
                {f.value == null ? 'missing' : f.value}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </LiveProcessingLayout>
  );
}
