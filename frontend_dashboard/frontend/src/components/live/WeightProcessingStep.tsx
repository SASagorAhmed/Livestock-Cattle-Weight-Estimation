import { useEffect, useState } from 'react';
import type { PredictStageResponse } from '../../types';
import LiveProcessingLayout from './LiveProcessingLayout';

const DEFAULT_STAGES = [
  'Preparing model input',
  'Running local weight model',
  'Calculating final estimate',
];

interface Props {
  predict: PredictStageResponse | null;
  busy: boolean;
}

export default function WeightProcessingStep({ predict, busy }: Props) {
  const stages = predict?.progress_stages?.length
    ? predict.progress_stages
    : DEFAULT_STAGES;
  const [idx, setIdx] = useState(0);
  const done = !!predict;

  useEffect(() => {
    if (done) {
      setIdx(stages.length);
      return undefined;
    }
    setIdx(0);
    let i = 0;
    const id = window.setInterval(() => {
      i = Math.min(i + 1, stages.length - 1);
      setIdx(i);
    }, 700);
    return () => window.clearInterval(id);
  }, [done, stages.length]);

  return (
    <LiveProcessingLayout
      title="Estimating cow weight"
      subtitle="All computation stays on this machine — no cloud upload."
      status={done ? 'Estimate ready' : 'Estimating cow weight...'}
      scanning={busy && !done}
    >
      <ul className="weight-stages">
        {stages.map((label, i) => {
          let cls = '';
          if (done || i < idx) cls = 'done';
          else if (i === idx) cls = 'active';
          return (
            <li key={label} className={cls}>{label}</li>
          );
        })}
      </ul>
    </LiveProcessingLayout>
  );
}
