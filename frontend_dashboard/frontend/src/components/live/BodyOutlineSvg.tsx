/** SVG red cow silhouette (Morpho-style) for overlay on live steps. */

import { BODY_OUTLINE_STROKE, outlineStrokeWidth } from '../../utils/bodyOutline';

interface Props {
  path: string | null | undefined;
  width: number;
  height: number;
  zIndex?: number;
}

export default function BodyOutlineSvg({ path, width, height, zIndex = 2 }: Props) {
  if (!path) return null;
  const w = width || 1;
  const h = height || 1;
  return (
    <svg
      className="overlay-svg"
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ pointerEvents: 'none', zIndex }}
    >
      <path
        d={path}
        fill="none"
        stroke={BODY_OUTLINE_STROKE}
        strokeWidth={outlineStrokeWidth(w, h)}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
