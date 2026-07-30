/** Human-readable labels for anatomical keypoints and measurement names. */

const MEASUREMENT_LABELS: Record<string, string> = {
  body_length: 'Body length',
  lower_chest: 'Lower chest',
  body_height: 'Body height',
  chest_depth_proxy: 'Chest depth',
  shoulder_width: 'Shoulder width',
  hip_width: 'Hip width',
  left_front_leg_length: 'Left front leg',
  right_front_leg_length: 'Right front leg',
  left_back_leg_length: 'Left back leg',
  right_back_leg_length: 'Right back leg',
  torso_diagonal: 'Torso diagonal',
  shoulder_center: 'Shoulder center',
  hip_center: 'Hip center',
  back_top: 'Back top',
  ground: 'Ground',
  elbow_center: 'Elbow center',
};

export function formatPartName(name: string | null | undefined): string {
  if (!name) return '—';
  if (MEASUREMENT_LABELS[name]) return MEASUREMENT_LABELS[name];
  return name
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export function formatMeasurementName(name: string | null | undefined): string {
  if (!name) return '—';
  return MEASUREMENT_LABELS[name] || formatPartName(name);
}

export function formatHeadFacing(dir: string | null | undefined): string {
  if (!dir) return 'Unknown';
  const d = dir.toLowerCase();
  if (d === 'left') return 'Left';
  if (d === 'right') return 'Right';
  return dir;
}

const MORPHO_AXIS_PREFIX: Record<string, string> = {
  A_start_lower_chest: 'A Start',
  A_end_withers: 'A End',
  B_start_tail_head: 'B Start',
  B_end_shoulder_region: 'B End',
};

const MORPHO_DEFAULT_ANATOMY: Record<string, string> = {
  A_start_lower_chest: 'Lower chest',
  A_end_withers: 'Back top',
  B_start_tail_head: 'Tail head',
  B_end_shoulder_region: 'Forward shoulder region',
};

function morphoAnatomyLabel(
  key: string,
  point?: { anatomy_label?: string; source_keypoint?: string } | null,
): string {
  if (key === 'B_start_tail_head') {
    return point?.anatomy_label || MORPHO_DEFAULT_ANATOMY[key];
  }
  if (key === 'B_end_shoulder_region') {
    return point?.anatomy_label || MORPHO_DEFAULT_ANATOMY[key];
  }
  return point?.anatomy_label
    || (point?.source_keypoint ? formatPartName(point.source_keypoint) : null)
    || MORPHO_DEFAULT_ANATOMY[key]
    || '';
}

export function pointDisplayLabel(
  key: string,
  point?: { anatomy_label?: string; source_keypoint?: string } | null,
  _headDirection?: string | null,
): string {
  const prefix = MORPHO_AXIS_PREFIX[key] || key;
  const anatomy = morphoAnatomyLabel(key, point);
  return anatomy ? `${prefix} · ${anatomy}` : prefix;
}

export function forwardShoulderRefLabel(headDirection?: string | null): string {
  const d = headDirection?.toLowerCase();
  if (d === 'right') return 'Right shoulder';
  if (d === 'left') return 'Left shoulder';
  return 'Shoulder';
}
