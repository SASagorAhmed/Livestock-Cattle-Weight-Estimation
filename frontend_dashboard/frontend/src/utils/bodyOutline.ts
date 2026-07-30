/** Shared Morpho-style red body outline across Pose / Measure / Segment / Morpho. */

import { fileUrl } from '../api';
import type { BodyContour } from '../types';

export const BODY_OUTLINE_STROKE = '#e53935';

export function outlineImageFromFiles(
  files?: Record<string, string> | null,
): string | null {
  if (!files) return null;
  return fileUrl(files['body_outline.jpg'] || files['measure_outline.jpg']);
}

export function contourPathFromBodyContour(
  contour?: BodyContour | null,
): string | null {
  const pts = contour?.points;
  if (!pts || pts.length < 3) return null;
  const parts = pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x} ${y}`);
  if (contour?.closed !== false) parts.push('Z');
  return parts.join(' ');
}

export function outlineStrokeWidth(imageW: number, imageH: number): number {
  return Math.max(2, Math.max(imageW, imageH) * 0.0028);
}
