import type { ImagePoint, KeypointPoint } from '../../types';

const MIN_CONF = 0.2;

export type AnatomicalLineId = 'body_length' | 'height_proxy';

export interface SuggestedEndpoint {
  point: ImagePoint;
  sourceKeypoint: string;
  confidence: number;
  note: string;
}

export interface AnatomicalLineSuggestion {
  id: AnatomicalLineId;
  label: string;
  description: string;
  /** Maps to Reference Scale Point A */
  pointA: SuggestedEndpoint;
  /** Maps to Reference Scale Point B */
  pointB: SuggestedEndpoint;
  warnings: string[];
}

function pick(
  keypoints: Record<string, KeypointPoint> | undefined,
  name: string,
): KeypointPoint | null {
  const p = keypoints?.[name];
  if (!p) return null;
  if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return null;
  if ((p.confidence ?? 0) < MIN_CONF) return null;
  return p;
}

function asEndpoint(
  p: KeypointPoint,
  name: string,
  note: string,
): SuggestedEndpoint {
  return {
    point: { x: Math.round(p.x * 100) / 100, y: Math.round(p.y * 100) / 100 },
    sourceKeypoint: name,
    confidence: p.confidence,
    note,
  };
}

/** Prefer the higher-confidence shoulder (side-view cows usually show one clearly). */
function bestShoulder(
  keypoints: Record<string, KeypointPoint> | undefined,
): { name: string; point: KeypointPoint } | null {
  const left = pick(keypoints, 'left_shoulder');
  const right = pick(keypoints, 'right_shoulder');
  if (left && right) {
    return left.confidence >= right.confidence
      ? { name: 'left_shoulder', point: left }
      : { name: 'right_shoulder', point: right };
  }
  if (left) return { name: 'left_shoulder', point: left };
  if (right) return { name: 'right_shoulder', point: right };
  return null;
}

/**
 * Approximate lower chest / brisket region from elbows (AP-10K has no brisket/belly).
 * Uses the midpoint of available elbows; falls back to a single elbow.
 */
function lowerChestProxy(
  keypoints: Record<string, KeypointPoint> | undefined,
): { name: string; point: KeypointPoint } | null {
  const le = pick(keypoints, 'left_elbow');
  const re = pick(keypoints, 'right_elbow');
  if (le && re) {
    return {
      name: 'elbow_center',
      point: {
        x: (le.x + re.x) / 2,
        y: (le.y + re.y) / 2,
        confidence: Math.min(le.confidence, re.confidence),
      },
    };
  }
  if (le) return { name: 'left_elbow', point: le };
  if (re) return { name: 'right_elbow', point: re };
  return null;
}

/**
 * Build auto-suggestions for the four anatomical intents using AP-10K only:
 * P1 lower belly/brisket → elbow proxy (approx)
 * P2 withers → neck (approx)
 * P3 tail head → tail_root
 * P4 forward shoulder → best shoulder
 */
export function suggestAnatomicalLines(
  keypoints: Record<string, KeypointPoint> | undefined | null,
): AnatomicalLineSuggestion[] {
  if (!keypoints) return [];

  const out: AnatomicalLineSuggestion[] = [];

  const tail = pick(keypoints, 'tail_root');
  const shoulder = bestShoulder(keypoints);
  if (tail && shoulder) {
    out.push({
      id: 'body_length',
      label: 'Body length (tail → shoulder)',
      description:
        'P3 tail head → P4 forward shoulder. Best AP-10K match for a body-length reference.',
      pointA: asEndpoint(tail, 'tail_root', 'P3 ≈ tail_root'),
      pointB: asEndpoint(
        shoulder.point,
        shoulder.name,
        'P4 ≈ shoulder (side-view forward shoulder proxy)',
      ),
      warnings: [],
    });
  }

  const chest = lowerChestProxy(keypoints);
  const neck = pick(keypoints, 'neck');
  if (chest && neck) {
    out.push({
      id: 'height_proxy',
      label: 'Height proxy (chest → withers)',
      description:
        'P1 lower chest → P2 withers. Brisket/belly is NOT in AP-10K; elbows approximate lower chest; neck approximates withers.',
      pointA: asEndpoint(
        chest.point,
        chest.name,
        'P1 ≈ elbow/chest proxy — not true brisket/belly',
      ),
      pointB: asEndpoint(neck, 'neck', 'P2 ≈ neck — not true withers'),
      warnings: [
        'True brisket and withers are not detected by the current model. Drag to correct after suggesting.',
      ],
    });
  }

  return out;
}

export function summarizeAvailablePoints(
  keypoints: Record<string, KeypointPoint> | undefined | null,
): Array<{ role: string; available: boolean; source: string }> {
  const shoulder = bestShoulder(keypoints || undefined);
  const chest = lowerChestProxy(keypoints || undefined);
  return [
    {
      role: 'P1 lower belly / brisket',
      available: Boolean(chest),
      source: chest ? `${chest.name} (approx)` : 'not available',
    },
    {
      role: 'P2 withers',
      available: Boolean(pick(keypoints || undefined, 'neck')),
      source: pick(keypoints || undefined, 'neck') ? 'neck (approx)' : 'not available',
    },
    {
      role: 'P3 tail head',
      available: Boolean(pick(keypoints || undefined, 'tail_root')),
      source: pick(keypoints || undefined, 'tail_root') ? 'tail_root' : 'not available',
    },
    {
      role: 'P4 forward shoulder',
      available: Boolean(shoulder),
      source: shoulder ? shoulder.name : 'not available',
    },
  ];
}
