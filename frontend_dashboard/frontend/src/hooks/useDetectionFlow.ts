import { useCallback, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import {
  createRun,
  getRun,
  stageDetect,
  stageMeasure,
  stagePose,
  stagePredict,
  stageScale,
  stageSegment,
  stageSelectCow,
} from '../api';
import type {
  FlowState,
  PredictionMode,
  ProcessingStage,
  StartDetectionInput,
} from '../types';

export const STAGES: Record<ProcessingStage, ProcessingStage> = {
  UPLOAD: 'UPLOAD',
  DETECTING_COW: 'DETECTING_COW',
  SELECTING_COW: 'SELECTING_COW',
  DETECTING_POSE: 'DETECTING_POSE',
  MEASURING_BODY: 'MEASURING_BODY',
  SEGMENTING_BODY: 'SEGMENTING_BODY',
  SCALE_INPUT: 'SCALE_INPUT',
  PREPARING_FEATURES: 'PREPARING_FEATURES',
  PREDICTING_WEIGHT: 'PREDICTING_WEIGHT',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
};

export const FLOW_STEPS: Array<{ id: ProcessingStage; label: string }> = [
  { id: 'UPLOAD', label: 'Upload' },
  { id: 'DETECTING_COW', label: 'Detect' },
  { id: 'SELECTING_COW', label: 'Select' },
  { id: 'DETECTING_POSE', label: 'Pose' },
  { id: 'MEASURING_BODY', label: 'Measure' },
  { id: 'SEGMENTING_BODY', label: 'Segment' },
  { id: 'SCALE_INPUT', label: 'Scale' },
  { id: 'PREPARING_FEATURES', label: 'Features' },
  { id: 'PREDICTING_WEIGHT', label: 'Predict' },
  { id: 'COMPLETED', label: 'Result' },
];

function initialState(): FlowState {
  return {
    stage: 'UPLOAD',
    runId: null,
    error: '',
    busy: false,
    enableSeg: true,
    predictionMode: 'smartphone_diagonal',
    localPreview: null,
    detect: null,
    pose: null,
    measure: null,
    segment: null,
    scale: null,
    features: [],
    predict: null,
    report: null,
    failedStage: null,
    poseRevealDone: false,
  };
}

function errMsg(err: unknown): string {
  if (isAxiosError(err)) {
    const data = err.response?.data as { error?: string } | undefined;
    return data?.error || err.message;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}

export function useDetectionFlow() {
  const [state, setState] = useState<FlowState>(initialState);
  const abortRef = useRef(0);
  const stateRef = useRef(state);
  const advancingRef = useRef<string | null>(null);
  stateRef.current = state;

  const patch = useCallback((partial: Partial<FlowState>) => {
    setState((s) => ({ ...s, ...partial }));
  }, []);

  const reset = useCallback(() => {
    abortRef.current += 1;
    advancingRef.current = null;
    setState(initialState());
  }, []);

  const fail = useCallback((stage: ProcessingStage, err: unknown) => {
    advancingRef.current = null;
    setState((s) => ({
      ...s,
      busy: false,
      error: errMsg(err),
      failedStage: stage,
      stage: 'FAILED',
    }));
  }, []);

  const runPose = useCallback(async (runId: string, gen: number) => {
    patch({ stage: 'DETECTING_POSE', busy: true, poseRevealDone: false, error: '' });
    const pose = await stagePose(runId);
    if (gen !== abortRef.current) return;
    patch({ pose, busy: false });
  }, [patch]);

  const advanceFromPose = useCallback(async () => {
    const s = stateRef.current;
    const gen = abortRef.current;
    const runId = s.runId;
    if (!runId || advancingRef.current === 'measure') return;
    // Re-use only if Measure already has the red outline bake
    const hasOutline = Boolean(
      s.measure?.files?.['body_outline.jpg'] || s.measure?.files?.['measure_outline.jpg'],
    );
    if (s.measure && hasOutline) {
      if (s.stage === 'DETECTING_POSE') {
        patch({ stage: 'MEASURING_BODY', poseRevealDone: true });
      }
      return;
    }
    advancingRef.current = 'measure';
    patch({ stage: 'MEASURING_BODY', busy: true, poseRevealDone: true });
    try {
      const measure = await stageMeasure(runId);
      if (gen !== abortRef.current) return;
      patch({
        measure,
        features: measure.measurements?.normalized_features || [],
        busy: false,
      });
    } catch (err) {
      if (gen === abortRef.current) fail('MEASURING_BODY', err);
    } finally {
      if (advancingRef.current === 'measure') advancingRef.current = null;
    }
  }, [fail, patch]);

  const runSegment = useCallback(async () => {
    const s = stateRef.current;
    const gen = abortRef.current;
    const runId = s.runId;
    if (!runId || advancingRef.current === 'segment') return;
    advancingRef.current = 'segment';
    patch({ stage: 'SEGMENTING_BODY', segment: null, busy: true, error: '' });
    try {
      const segment = await stageSegment(runId);
      if (gen !== abortRef.current) return;
      let features = s.features;
      if (segment.status === 'completed') {
        const report = await getRun(runId);
        features = report.normalized_features || features;
      }
      patch({ segment, features, busy: false });
    } catch (err) {
      if (gen === abortRef.current) {
        patch({
          busy: false,
          segment: {
            run_id: runId,
            stage: 'segment',
            status: 'failed',
            error: errMsg(err),
            segmentation: null,
          },
        });
      }
    } finally {
      if (advancingRef.current === 'segment') advancingRef.current = null;
    }
  }, [patch]);

  const advanceFromMeasure = useCallback(async () => {
    const s = stateRef.current;
    if (!s.runId) return;
    if (s.segment && s.stage === 'MEASURING_BODY') {
      patch({ stage: 'SEGMENTING_BODY' });
      return;
    }
    await runSegment();
  }, [patch, runSegment]);

  const retrySegment = useCallback(async () => {
    await runSegment();
  }, [runSegment]);

  const advanceFromSegment = useCallback(() => {
    if (stateRef.current.stage !== 'SEGMENTING_BODY') return;
    patch({ stage: 'SCALE_INPUT', busy: false });
  }, [patch]);

  const startDetection = useCallback(async (input: StartDetectionInput) => {
    const gen = ++abortRef.current;
    const mode: PredictionMode = input.predictionMode || 'smartphone_diagonal';
    patch({
      ...initialState(),
      enableSeg: input.enableSeg,
      predictionMode: mode,
      localPreview: input.localPreview,
      busy: true,
      stage: 'DETECTING_COW',
      error: '',
    });
    try {
      const fd = new FormData();
      fd.append('file', input.file);
      // Guided flow always enables segmentation attempt; backend may still skip/fail gracefully
      fd.append('enable_segmentation', 'true');
      fd.append('prediction_mode', mode);

      const created = await createRun(fd);
      if (gen !== abortRef.current) return;
      const runId = created.run_id;
      patch({ runId });

      const detect = await stageDetect(runId);
      if (gen !== abortRef.current) return;
      patch({ detect, busy: false });

      if (detect.needs_cow_selection) {
        patch({ stage: 'SELECTING_COW' });
        return;
      }

      // Brief success beat before pose
      await new Promise((r) => setTimeout(r, 900));
      if (gen !== abortRef.current) return;
      await runPose(runId, gen);
    } catch (err) {
      if (gen === abortRef.current) fail('DETECTING_COW', err);
    }
  }, [fail, patch, runPose]);

  const selectCow = useCallback(async (cowId: number) => {
    const gen = ++abortRef.current;
    const s = stateRef.current;
    const runId = s.runId;
    if (!runId) return;
    patch({ busy: true, error: '' });
    try {
      const select = await stageSelectCow(runId, cowId);
      if (gen !== abortRef.current) return;
      patch({
        detect: s.detect
          ? {
              ...s.detect,
              selected_cow_id: select.selected_cow_id,
              files: { ...s.detect.files, ...select.files },
            }
          : s.detect,
      });
      await runPose(runId, gen);
    } catch (err) {
      if (gen === abortRef.current) fail('SELECTING_COW', err);
    }
  }, [fail, patch, runPose]);

  const applyScale = useCallback(async (opts: {
    skip?: boolean;
    reference_px?: number;
    reference_cm?: number;
    point_a?: { x: number; y: number } | null;
    point_b?: { x: number; y: number } | null;
    four_points?: Record<string, { x: number; y: number }> | null;
  } = {}) => {
    const gen = abortRef.current;
    const runId = stateRef.current.runId;
    if (!runId || advancingRef.current === 'scale') return;
    advancingRef.current = 'scale';
    patch({ busy: true, error: '', stage: 'PREPARING_FEATURES' });
    try {
      const scale = await stageScale(runId, opts);
      if (gen !== abortRef.current) return;
      patch({
        scale,
        features: scale.normalized_features || [],
        busy: false,
      });
    } catch (err) {
      if (gen === abortRef.current) fail('SCALE_INPUT', err);
    } finally {
      if (advancingRef.current === 'scale') advancingRef.current = null;
    }
  }, [fail, patch]);

  const advanceFromFeatures = useCallback(async () => {
    const s = stateRef.current;
    const gen = abortRef.current;
    const runId = s.runId;
    if (!runId || advancingRef.current === 'predict' || s.stage === 'PREDICTING_WEIGHT') return;
    advancingRef.current = 'predict';
    patch({ stage: 'PREDICTING_WEIGHT', busy: true });
    try {
      const predict = await stagePredict(runId);
      if (gen !== abortRef.current) return;
      const report = predict.report || (await getRun(runId));
      patch({
        predict,
        report,
        features: report.normalized_features || s.features,
        busy: false,
        stage: 'COMPLETED',
      });
    } catch (err) {
      if (gen === abortRef.current) fail('PREDICTING_WEIGHT', err);
    } finally {
      if (advancingRef.current === 'predict') advancingRef.current = null;
    }
  }, [fail, patch]);

  const retry = useCallback(() => {
    setState((s) => ({
      ...initialState(),
      enableSeg: s.enableSeg,
      predictionMode: s.predictionMode,
      localPreview: s.localPreview,
      stage: 'UPLOAD',
    }));
  }, []);

  return {
    state,
    patch,
    reset,
    startDetection,
    selectCow,
    applyScale,
    advanceFromPose,
    advanceFromMeasure,
    advanceFromSegment,
    advanceFromFeatures,
    retrySegment,
    retry,
  };
}
