import type { ReactNode } from 'react';
import { FLOW_STEPS } from '../../hooks/useDetectionFlow';
import type { ProcessingStage } from '../../types';

const ORDER = FLOW_STEPS.map((s) => s.id);

function rank(stage: ProcessingStage | null | undefined): number {
  if (!stage || stage === 'FAILED') return -1;
  const i = ORDER.indexOf(stage);
  return i < 0 ? 0 : i;
}

interface Props {
  stage: ProcessingStage | null | undefined;
}

export default function StepProgressIndicator({ stage }: Props) {
  const current = rank(stage);
  return (
    <div className="step-progress" aria-label="Processing progress">
      {FLOW_STEPS.map((step, idx) => {
        let cls = 'step-dot';
        if (stage === 'COMPLETED') cls += ' done';
        else if (idx < current) cls += ' done';
        else if (idx === current) cls += ' active';
        const label: ReactNode =
          idx < current || stage === 'COMPLETED' ? '✓' : String(idx + 1);
        return (
          <div key={step.id} className={cls} title={step.label} aria-label={step.label}>
            {label}
          </div>
        );
      })}
    </div>
  );
}
