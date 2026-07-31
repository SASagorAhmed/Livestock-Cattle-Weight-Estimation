import { useCallback, useState, type ReactNode } from 'react';

import { STAGES, useDetectionFlow } from './hooks/useDetectionFlow';
import StepProgressIndicator from './components/live/StepProgressIndicator';
import UploadStep from './components/live/UploadStep';
import CowDetectionStep from './components/live/CowDetectionStep';
import CowSelectionStep from './components/live/CowSelectionStep';
import PoseDetectionStep from './components/live/PoseDetectionStep';
import MeasurementStep from './components/live/MeasurementStep';
import SegmentationStep from './components/live/SegmentationStep';
import SmartphoneDiagonalStep from './components/live/SmartphoneDiagonalStep';
import FeaturePreparationStep from './components/live/FeaturePreparationStep';
import WeightProcessingStep from './components/live/WeightProcessingStep';
import FinalWeightResult from './components/live/FinalWeightResult';
import TechnicalDetailsDrawer from './components/live/TechnicalDetailsDrawer';

export default function App() {
  const flow = useDetectionFlow();
  const { state } = flow;
  const [detailsOpen, setDetailsOpen] = useState(false);

  const onPoseRevealDone = useCallback(() => {
    if (state.stage === STAGES.DETECTING_POSE && !state.measure) {
      void flow.advanceFromPose();
    }
  }, [flow, state.measure, state.stage]);

  const onMeasureContinue = useCallback(() => {
    if (state.stage === STAGES.MEASURING_BODY) {
      void flow.advanceFromMeasure();
    }
  }, [flow, state.stage]);

  const onSegmentDone = useCallback(() => {
    if (state.stage === STAGES.SEGMENTING_BODY && state.segment) {
      flow.advanceFromSegment();
    }
  }, [flow, state.segment, state.stage]);

  const onSegmentRetry = useCallback(() => {
    void flow.retrySegment();
  }, [flow]);

  const onFeaturesDone = useCallback(() => {
    if (state.stage === STAGES.PREPARING_FEATURES && state.scale) {
      void flow.advanceFromFeatures();
    }
  }, [flow, state.scale, state.stage]);

  let body: ReactNode = null;
  switch (state.stage) {
    case STAGES.UPLOAD:
      body = (
        <UploadStep
          busy={state.busy}
          onStart={flow.startDetection}
        />
      );
      break;
    case STAGES.DETECTING_COW:
      body = (
        <CowDetectionStep
          detect={state.detect}
          localPreview={state.localPreview}
          busy={state.busy}
        />
      );
      break;
    case STAGES.SELECTING_COW:
      body = (
        <CowSelectionStep
          detect={state.detect}
          onSelect={(id) => void flow.selectCow(id)}
          busy={state.busy}
        />
      );
      break;
    case STAGES.DETECTING_POSE:
      body = (
        <PoseDetectionStep
          pose={state.pose}
          onRevealDone={onPoseRevealDone}
        />
      );
      break;
    case STAGES.MEASURING_BODY:
      body = (
        <MeasurementStep
          measure={state.measure}
          pose={state.pose}
          onContinue={onMeasureContinue}
        />
      );
      break;
    case STAGES.SEGMENTING_BODY:
      body = (
        <SegmentationStep
          segment={state.segment}
          pose={state.pose}
          measure={state.measure}
          onDone={onSegmentDone}
          onRetry={onSegmentRetry}
        />
      );
      break;
    case STAGES.SCALE_INPUT:
      body = (
        <SmartphoneDiagonalStep
          measure={state.measure}
          pose={state.pose}
          segment={state.segment}
          runId={state.runId}
          busy={state.busy}
          onApply={(payload) => void flow.applyScale(payload)}
        />
      );
      break;
    case STAGES.PREPARING_FEATURES:
      body = (
        <FeaturePreparationStep
          features={state.features}
          measurePx={state.measure?.measurements?.measurements_px || {}}
          scale={state.scale}
          onDone={onFeaturesDone}
        />
      );
      break;
    case STAGES.PREDICTING_WEIGHT:
      body = (
        <WeightProcessingStep
          predict={state.predict}
          busy={state.busy}
        />
      );
      break;
    case STAGES.COMPLETED:
      body = (
        <FinalWeightResult
          report={state.report}
          onAgain={flow.reset}
          onOpenDetails={() => setDetailsOpen(true)}
        />
      );
      break;
    case STAGES.FAILED:
      body = (
        <div className="live-card">
          <h1 className="live-title">Something went wrong</h1>
          <div className="error-box">{state.error}</div>
          <p className="live-sub">
            Failed during: {state.failedStage || 'unknown stage'}
          </p>
          <div className="btn-row">
            <button type="button" className="btn btn-primary" onClick={flow.retry}>
              Retry
            </button>
            <button type="button" className="btn btn-ghost" onClick={flow.reset}>
              Start Over
            </button>
          </div>
        </div>
      );
      break;
    default:
      body = null;
  }

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <div className="brand">
          <div className="brand-mark">CW</div>
          <div className="brand-text">
            <strong>Cow Weight Detection</strong>
            <span>Live cattle weight estimation</span>
          </div>
        </div>
        {state.stage !== STAGES.UPLOAD ? (
          <button type="button" className="btn btn-ghost" onClick={flow.reset}>
            Start over
          </button>
        ) : null}
      </header>

      <main className="app-main">
        <StepProgressIndicator
          stage={state.stage === STAGES.FAILED ? state.failedStage : state.stage}
        />
        {state.error && state.stage !== STAGES.FAILED ? (
          <div className="error-box">{state.error}</div>
        ) : null}
        {body}
      </main>

      <TechnicalDetailsDrawer
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        report={state.report}
      />
    </div>
  );
}
