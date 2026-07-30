/** Lower chest guide: shoulder center → hip center (display only). */

export type Pt = { x: number; y: number };

export type LowerChestLine = {
  p1: Pt;
  p2: Pt;
  mid: Pt;
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

function mid2(a: Pt, b: Pt): Pt {
  return { x: 0.5 * (a.x + b.x), y: 0.5 * (a.y + b.y) };
}

function center(
  a: KptLike,
  b: KptLike,
): Pt | null {
  const pa = ok(a);
  const pb = ok(b);
  if (pa && pb) return mid2(pa, pb);
  return pa || pb;
}

/** From AP-10K keypoints dict or named map. */
export function lowerChestFromKeypoints(
  keypoints: Record<string, KptLike> | null | undefined,
): LowerChestLine | null {
  if (!keypoints) return null;
  const sc = center(keypoints.left_shoulder, keypoints.right_shoulder);
  const hc = center(keypoints.left_hip, keypoints.right_hip);
  if (!sc || !hc) return null;
  return {
    p1: sc,
    p2: hc,
    mid: mid2(sc, hc),
    label: 'Lower chest',
  };
}

/** From backend lower_chest_line JSON. */
export function lowerChestFromLineDict(
  line: {
    detected?: boolean;
    p1?: number[] | null;
    p2?: number[] | null;
    mid?: number[] | null;
    label?: string;
  } | null | undefined,
): LowerChestLine | null {
  if (!line?.detected || !line.p1 || !line.p2 || line.p1.length < 2 || line.p2.length < 2) {
    return null;
  }
  const p1 = { x: Number(line.p1[0]), y: Number(line.p1[1]) };
  const p2 = { x: Number(line.p2[0]), y: Number(line.p2[1]) };
  if (![p1.x, p1.y, p2.x, p2.y].every(Number.isFinite)) return null;
  let mid = mid2(p1, p2);
  if (line.mid && line.mid.length >= 2) {
    const mx = Number(line.mid[0]);
    const my = Number(line.mid[1]);
    if (Number.isFinite(mx) && Number.isFinite(my)) mid = { x: mx, y: my };
  }
  return {
    p1,
    p2,
    mid,
    label: line.label || 'Lower chest',
  };
}
