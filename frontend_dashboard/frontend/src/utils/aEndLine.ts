/** A End = Back top on cow upper border (never detection bbox top). */

export type Pt = { x: number; y: number };

export type AEndAxis = {
  aEnd: Pt;
  ground: Pt;
  label: string;
};

type KptLike = { x: number; y: number; confidence?: number; status?: string } | null | undefined;

function ok(p: KptLike, minConf = 0.15): Pt | null {
  if (!p || p.status === 'missing') return null;
  const x = Number(p.x);
  const y = Number(p.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  const conf = Number(p.confidence ?? 0);
  if (conf < minConf && p.status !== 'ok') return null;
  return { x, y };
}

function shoulderCenterX(
  keypoints: Record<string, KptLike> | null | undefined,
  bbox?: number[] | null,
): number | null {
  let hx: number | null = null;
  if (keypoints) {
    const ls = ok(keypoints.left_shoulder);
    const rs = ok(keypoints.right_shoulder);
    if (ls && rs) hx = 0.5 * (ls.x + rs.x);
    else if (ls) hx = ls.x;
    else if (rs) hx = rs.x;
  }
  if (hx == null && bbox && bbox.length >= 4) {
    hx = 0.5 * (bbox[0] + bbox[2]);
  }
  if (hx == null) return null;
  if (bbox && bbox.length >= 4) {
    hx = Math.max(bbox[0], Math.min(bbox[2], hx));
  }
  return hx;
}

function keypointTopY(keypoints: Record<string, KptLike> | null | undefined): number | null {
  if (!keypoints) return null;
  const ys: number[] = [];
  for (const n of ['neck', 'tail_root', 'left_shoulder', 'right_shoulder'] as const) {
    const p = ok(keypoints[n]);
    if (p) ys.push(p.y);
  }
  return ys.length ? Math.min(...ys) : null;
}

function groundY(keypoints: Record<string, KptLike> | null | undefined): number | null {
  if (!keypoints) return null;
  const ys: number[] = [];
  for (const n of [
    'left_front_hoof', 'right_front_hoof', 'left_back_hoof', 'right_back_hoof',
  ] as const) {
    const p = ok(keypoints[n]);
    if (p) ys.push(p.y);
  }
  return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : null;
}

/** Keypoint Back top only — never bbox top (that floats outside the cow). */
export function aEndFromKeypoints(
  keypoints: Record<string, KptLike> | null | undefined,
  bbox?: number[] | null,
): AEndAxis | null {
  const hx = shoulderCenterX(keypoints, bbox);
  const gy = groundY(keypoints);
  const kptTop = keypointTopY(keypoints);
  if (hx == null || gy == null || kptTop == null) return null;
  let upper = kptTop;
  let ground = gy;
  if (bbox && bbox.length >= 4) {
    const by1 = Number(bbox[1]);
    const by2 = Number(bbox[3]);
    upper = Math.max(by1, Math.min(by2, upper));
    ground = Math.max(by1, Math.min(by2, ground));
  }
  return {
    aEnd: { x: hx, y: upper },
    ground: { x: hx, y: ground },
    label: 'A End',
  };
}

/** From backend a_end_line JSON (already clamped server-side when mask present). */
export function aEndFromLineDict(
  line: {
    detected?: boolean;
    a_end?: number[] | null;
    ground?: number[] | null;
    p1?: number[] | null;
    p2?: number[] | null;
    label?: string;
  } | null | undefined,
): AEndAxis | null {
  const top = line?.a_end || line?.p1;
  const bot = line?.ground || line?.p2;
  if (!line?.detected || !top || !bot || top.length < 2 || bot.length < 2) return null;
  const aEnd = { x: Number(top[0]), y: Number(top[1]) };
  const ground = { x: Number(bot[0]), y: Number(bot[1]) };
  if (![aEnd.x, aEnd.y, ground.x, ground.y].every(Number.isFinite)) return null;
  return { aEnd, ground, label: line.label || 'A End' };
}
