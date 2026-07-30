import axios from 'axios';
import type {
  CapabilitiesResponse,
  CompleteReport,
  CreateRunResponse,
  DetectStageResponse,
  HeadAnchor,
  MeasureStageResponse,
  PoseStageResponse,
  PredictStageResponse,
  ScaleStageResponse,
  SegmentStageResponse,
  SelectCowResponse,
} from './types';

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '')
  || 'http://127.0.0.1:5001';

const API = axios.create({
  baseURL: API_BASE,
  timeout: 600000,
});

export async function getCapabilities(): Promise<CapabilitiesResponse> {
  const { data } = await API.get<CapabilitiesResponse>('/api/capabilities');
  return data;
}

export async function createRun(formData: FormData): Promise<CreateRunResponse> {
  const { data } = await API.post<CreateRunResponse>('/api/runs', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function stageDetect(runId: string): Promise<DetectStageResponse> {
  const { data } = await API.post<DetectStageResponse>(`/api/runs/${runId}/detect`);
  return data;
}

export async function stageSelectCow(runId: string, cowId: number): Promise<SelectCowResponse> {
  const { data } = await API.post<SelectCowResponse>(`/api/runs/${runId}/select-cow`, {
    cow_id: cowId,
  });
  return data;
}

export async function stagePose(runId: string): Promise<PoseStageResponse> {
  const { data } = await API.post<PoseStageResponse>(`/api/runs/${runId}/pose`);
  return data;
}

export async function stageMeasure(runId: string): Promise<MeasureStageResponse> {
  const { data } = await API.post<MeasureStageResponse>(`/api/runs/${runId}/measure`);
  return data;
}

export async function stageSegment(runId: string): Promise<SegmentStageResponse> {
  const { data } = await API.post<SegmentStageResponse>(`/api/runs/${runId}/segment`);
  return data;
}

export async function stageScale(
  runId: string,
  payload: {
    skip?: boolean;
    reference_px?: number;
    reference_cm?: number;
    point_a?: { x: number; y: number } | null;
    point_b?: { x: number; y: number } | null;
    four_points?: Record<string, { x: number; y: number }> | null;
  },
): Promise<ScaleStageResponse> {
  const { data } = await API.post<ScaleStageResponse>(`/api/runs/${runId}/scale`, payload);
  return data;
}

export async function stageFourPointSuggest(
  runId: string,
  payload: { head_direction?: 'left' | 'right' | string } = {},
): Promise<{
  available: boolean;
  reason?: string | null;
  keypoints?: Record<string, {
    x: number;
    y: number;
    confidence?: number;
    status?: string;
    method?: string;
    name?: string;
    anatomy_label?: string;
    source_keypoint?: string;
  }> | null;
  model_path?: string | null;
  model_available?: boolean;
  head_direction_required?: boolean;
  inferred_head_direction?: string | null;
  head_direction_used?: string | null;
  head_detected?: boolean;
  head_anchor?: HeadAnchor | null;
  point_detector?: string | null;
  method?: string | null;
  lower_chest_guide_line?: {
    detected: boolean;
    p1?: [number, number] | null;
    p2?: [number, number] | null;
    mid?: [number, number] | null;
    label?: string;
  } | null;
  a_end_line?: {
    detected: boolean;
    a_end?: [number, number] | null;
    ground?: [number, number] | null;
    p1?: [number, number] | null;
    p2?: [number, number] | null;
    label?: string;
    line_label?: string;
  } | null;
  files?: Record<string, string>;
}> {
  const { data } = await API.post(`/api/runs/${runId}/four-point-suggest`, payload);
  return data;
}

export async function stageFourPointDebug(
  runId: string,
  payload: {
    keypoints: Record<string, {
      x: number;
      y: number;
      status?: string;
      method?: string;
      confidence?: number;
      anatomy_label?: string;
    }>;
    a_end_line?: {
      detected?: boolean;
      a_end?: [number, number] | null;
      ground?: [number, number] | null;
      p1?: [number, number] | null;
      p2?: [number, number] | null;
      label?: string;
      line_label?: string;
    } | null;
    lower_chest_guide?: {
      detected?: boolean;
      p1?: [number, number] | null;
      p2?: [number, number] | null;
      mid?: [number, number] | null;
      label?: string;
    } | null;
    head_anchor?: HeadAnchor | null;
    head_direction?: string | null;
  },
): Promise<{
  status: string;
  a_end_line?: {
    detected?: boolean;
    a_end?: [number, number] | null;
    ground?: [number, number] | null;
    p1?: [number, number] | null;
    p2?: [number, number] | null;
    label?: string;
    line_label?: string;
  } | null;
  files?: Record<string, string>;
}> {
  const { data } = await API.post(`/api/runs/${runId}/four-point-debug`, payload);
  return data;
}

export async function stagePredict(runId: string): Promise<PredictStageResponse> {
  const { data } = await API.post<PredictStageResponse>(`/api/runs/${runId}/predict`);
  return data;
}

export async function getRun(runId: string): Promise<CompleteReport> {
  const { data } = await API.get<CompleteReport>(`/api/run/${runId}`);
  return data;
}

export function fileUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (path.startsWith('http')) return path;
  return `${API_BASE}${path}`;
}

export function downloadUrl(runId: string, kind: 'json' | 'csv' | 'image'): string {
  return `${API_BASE}/api/run/${runId}/download/${kind}`;
}

export default API;
