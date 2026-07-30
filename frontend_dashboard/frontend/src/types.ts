export type ProcessingStage =
  | 'UPLOAD'
  | 'DETECTING_COW'
  | 'SELECTING_COW'
  | 'DETECTING_POSE'
  | 'MEASURING_BODY'
  | 'SEGMENTING_BODY'
  | 'SCALE_INPUT'
  | 'PREPARING_FEATURES'
  | 'PREDICTING_WEIGHT'
  | 'COMPLETED'
  | 'FAILED';

export type PredictionMode = 'heuristic' | 'measurement' | 'smartphone_diagonal';

export interface ImageSize {
  width: number;
  height: number;
}

export interface KeypointPoint {
  x: number;
  y: number;
  confidence: number;
  status?: string;
}

export interface CowDetection {
  cow_id: number;
  bbox: [number, number, number, number] | number[];
  bbox_confidence: number;
  bbox_area: number;
  keypoints: Record<string, KeypointPoint>;
}

export interface PixelCalcSegment {
  from: string;
  to: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  distance_px: number;
}

export interface PixelCalculation {
  name: string;
  start_point?: string | null;
  end_point?: string | null;
  x1?: number | null;
  y1?: number | null;
  x2?: number | null;
  y2?: number | null;
  formula: string;
  substituted: string;
  result_px: number | null;
  available: boolean;
  segments?: PixelCalcSegment[];
}

export interface HeadAnchor {
  x: number;
  y: number;
  detected: boolean;
}

export interface NormalizedFeature {
  name: string;
  formula?: string;
  value: number | null;
  numerator?: number | null;
  denominator?: number | null;
}

export type ScaleMode =
  | 'UNSELECTED'
  | 'WITHOUT_SCALE'
  | 'WITH_REFERENCE';

export interface ImagePoint {
  x: number;
  y: number;
}

export interface ReferenceScale {
  pointA: ImagePoint;
  pointB: ImagePoint;
  referencePixels: number;
  referenceCentimetres: number;
  centimetresPerPixel: number;
}

export interface ScaleInfo {
  provided: boolean;
  reference_scale_used?: boolean;
  cm_per_px?: number | null;
  reference_px?: number | null;
  reference_cm?: number | null;
  point_a?: ImagePoint | null;
  point_b?: ImagePoint | null;
  reference_pixels?: number | null;
  reference_length_cm?: number | null;
  converted_measurements?: Record<string, number | null> | null;
  message?: string;
}

export interface SegmentationInfo {
  body_pixel_area?: number;
  torso_pixel_area?: number;
  body_perimeter_px?: number;
  upper_chest_band?: [number, number];
  lower_chest_band?: [number, number];
  upper_chest_pixel_area?: number;
  lower_chest_pixel_area?: number;
  lower_chest_line?: {
    detected?: boolean;
    p1?: number[] | null;
    p2?: number[] | null;
    mid?: number[] | null;
    label?: string;
  };
  a_end_line?: {
    detected?: boolean;
    a_end?: number[] | null;
    ground?: number[] | null;
    p1?: number[] | null;
    p2?: number[] | null;
    label?: string;
  };
  tail_anchor?: {
    x: number | null;
    y: number | null;
    detected: boolean;
    source?: string;
    confidence?: number;
  };
  shoulder_markers?: Array<{
    name: string;
    x: number;
    y: number;
    confidence?: number;
  }>;
  belly_boundary_points?: number[][];
  [key: string]: unknown;
}

export interface MeasurementsDict {
  measurements_px?: Record<string, number | null>;
  measurements_cm?: Record<string, number | null> | null;
  pixel_calculations?: PixelCalculation[];
  normalized_features?: NormalizedFeature[];
  segmentation?: SegmentationInfo | null;
  measurement_lines?: Array<{ p1: number[]; p2: number[]; name: string }>;
  [key: string]: unknown;
}

export interface CreateRunResponse {
  run_id: string;
  status: string;
  enable_segmentation: boolean;
  prediction_mode: PredictionMode;
  image_size?: ImageSize;
  files: Record<string, string>;
}

export interface DetectStageResponse {
  run_id: string;
  stage: string;
  status: string;
  num_cows_detected: number;
  needs_cow_selection: boolean;
  selected_cow_id: number;
  detections: CowDetection[];
  files: Record<string, string>;
  image_size?: ImageSize;
}

export interface SelectCowResponse {
  run_id: string;
  stage: string;
  status: string;
  selected_cow_id: number;
  selected_detection: CowDetection;
  files: Record<string, string>;
}

export interface BodyContour {
  points: number[][];
  closed?: boolean;
  line_label?: string;
}

export interface PoseStageResponse {
  run_id: string;
  stage: string;
  status: string;
  selected_cow_id: number;
  selected_detection: CowDetection;
  keypoint_groups: Record<string, string[]>;
  skeleton: number[][];
  detected_points: number;
  total_points: number;
  low_confidence_keypoints: string[];
  head_direction?: 'left' | 'right' | null;
  head_detected?: boolean;
  /** Display-only Morpho-style silhouette; not a model feature. */
  body_contour?: BodyContour | null;
  body_contour_error?: string | null;
  /** A End / Body height snapped to red silhouette top when mask present. */
  a_end_line?: {
    detected?: boolean;
    a_end?: number[] | null;
    ground?: number[] | null;
    p1?: number[] | null;
    p2?: number[] | null;
    label?: string;
    line_label?: string;
  } | null;
  files: Record<string, string>;
  image_size?: ImageSize;
}

export interface MeasureStageResponse {
  run_id: string;
  stage: string;
  status: string;
  measurements: MeasurementsDict;
  measurement_sequence: PixelCalculation[];
  /** Display-only cow silhouette (Morpho-style red outline); not a model feature. */
  body_contour?: BodyContour | null;
  /** Present when outline bake/segment failed during measure (measurements still ok). */
  body_contour_error?: string | null;
  files: Record<string, string>;
  image_size?: ImageSize;
}

export interface SegmentStageResponse {
  run_id: string;
  stage: string;
  status: 'completed' | 'skipped' | 'failed' | string;
  message?: string;
  error?: string;
  segmentation: SegmentationInfo | null;
  body_contour?: BodyContour | null;
  body_contour_error?: string | null;
  image_size?: { width: number; height: number };
  files?: Record<string, string>;
}

export interface ScaleStageResponse {
  run_id: string;
  stage: string;
  status: string;
  scale: ScaleInfo;
  measurements?: MeasurementsDict;
  normalized_features: NormalizedFeature[];
}

export interface WeightPayload {
  selected_mode: string;
  heuristic?: Record<string, unknown>;
  measurement_model?: Record<string, unknown>;
  smartphone_diagonal?: Record<string, unknown>;
  selected?: { weight_kg?: number; weight_lb?: number; [key: string]: unknown };
  progress_stages?: string[];
  [key: string]: unknown;
}

export interface FinalSummary {
  weight_kg?: number | null;
  weight_lb?: number | null;
  selected_cow_id?: number | null;
  selected_model?: string;
  num_cows?: number;
  pose_status?: string;
  segmentation_status?: string;
  scale_status?: string;
  reference_scale_used?: boolean;
  detected_points?: number;
  body_length_px?: number | null;
  body_height_px?: number | null;
  chest_depth_proxy_px?: number | null;
  body_length_cm?: number | null;
  body_height_cm?: number | null;
  chest_depth_proxy_cm?: number | null;
  cm_per_px?: number | null;
  low_confidence_keypoints?: string[];
  warnings?: string[];
  processing_time_sec?: number;
  A_px?: number | null;
  B_px?: number | null;
  A_cm?: number | null;
  B_cm?: number | null;
  estimated_heart_girth_C_cm?: number | null;
  estimated_heart_girth_C_in?: number | null;
  diagonal_method?: string | null;
  diagonal_status?: string | null;
  point_detector?: string | null;
  smartphone_diagonal?: Record<string, unknown> | null;
}

export interface CompleteReport {
  run_id: string;
  created_at?: string;
  disclaimer?: string;
  steps?: Record<string, { status: string; [key: string]: unknown }>;
  warnings?: string[];
  num_cows_detected?: number;
  selected_cow_id?: number | null;
  detections?: CowDetection[];
  selected_detection?: CowDetection | null;
  measurements?: MeasurementsDict | null;
  scale?: ScaleInfo;
  normalized_features?: NormalizedFeature[];
  weight?: WeightPayload | null;
  files?: Record<string, string>;
  final?: FinalSummary | null;
  image_size?: ImageSize;
  [key: string]: unknown;
}

export interface PredictStageResponse {
  run_id: string;
  stage: string;
  status: string;
  progress_stages?: string[];
  weight: WeightPayload;
  final: FinalSummary;
  report: CompleteReport;
  files: Record<string, string>;
}

export interface CapabilitiesResponse {
  primary_model: {
    id: string;
    label: string;
    status: string;
  };
  segmentation: boolean;
  webcam: boolean;
}

export interface StartDetectionInput {
  file: File;
  enableSeg: boolean;
  predictionMode: PredictionMode;
  localPreview: string | null;
}

export interface FlowState {
  stage: ProcessingStage;
  runId: string | null;
  error: string;
  busy: boolean;
  enableSeg: boolean;
  predictionMode: PredictionMode;
  localPreview: string | null;
  detect: DetectStageResponse | null;
  pose: PoseStageResponse | null;
  measure: MeasureStageResponse | null;
  segment: SegmentStageResponse | null;
  scale: ScaleStageResponse | null;
  features: NormalizedFeature[];
  predict: PredictStageResponse | null;
  report: CompleteReport | null;
  failedStage: ProcessingStage | null;
  poseRevealDone: boolean;
}
