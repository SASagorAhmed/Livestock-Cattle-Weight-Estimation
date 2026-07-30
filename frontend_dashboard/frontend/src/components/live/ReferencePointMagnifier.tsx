import { useEffect, useRef } from 'react';

interface Props {
  /** When false, show idle placeholder */
  active: boolean;
  image: HTMLImageElement | null;
  /** Pointer/focus in display CSS pixels relative to image top-left */
  displayX: number;
  displayY: number;
  imageDisplayWidth: number;
  imageDisplayHeight: number;
  label?: string;
}

const SIZE = 140;
const ZOOM = 2.5;

/**
 * Fixed (in-flow) zoom preview — not absolutely positioned over the image.
 */
export default function ReferencePointMagnifier({
  active,
  image,
  displayX,
  displayY,
  imageDisplayWidth,
  imageDisplayHeight,
  label = 'Zoom preview',
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = '#0f1a11';
    ctx.fillRect(0, 0, SIZE, SIZE);

    if (!active || !image || image.naturalWidth <= 0 || imageDisplayWidth <= 0) {
      ctx.fillStyle = '#81c784';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Place or drag a point', SIZE / 2, SIZE / 2);
      return;
    }

    const nw = image.naturalWidth;
    const nh = image.naturalHeight;
    const scaleX = nw / Math.max(1, imageDisplayWidth);
    const scaleY = nh / Math.max(1, imageDisplayHeight);
    const ox = displayX * scaleX;
    const oy = displayY * scaleY;
    const srcHalf = ((SIZE / 2) / ZOOM) * Math.max(scaleX, scaleY);

    const sx = Math.max(0, Math.min(nw - 1, ox - srcHalf));
    const sy = Math.max(0, Math.min(nh - 1, oy - srcHalf));
    const sw = Math.min(srcHalf * 2, nw - sx);
    const sh = Math.min(srcHalf * 2, nh - sy);

    try {
      ctx.drawImage(image, sx, sy, sw, sh, 0, 0, SIZE, SIZE);
    } catch {
      ctx.fillStyle = '#81c784';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Preview unavailable', SIZE / 2, SIZE / 2);
      return;
    }

    ctx.strokeStyle = '#7CFF8A';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(SIZE / 2, 0);
    ctx.lineTo(SIZE / 2, SIZE);
    ctx.moveTo(0, SIZE / 2);
    ctx.lineTo(SIZE, SIZE / 2);
    ctx.stroke();

    ctx.strokeStyle = '#fff';
    ctx.beginPath();
    ctx.arc(SIZE / 2, SIZE / 2, 7, 0, Math.PI * 2);
    ctx.stroke();
  }, [active, image, displayX, displayY, imageDisplayWidth, imageDisplayHeight]);

  return (
    <div className="ref-zoom-panel">
      <div className="ref-zoom-panel__label">{label}</div>
      <canvas
        ref={canvasRef}
        className="ref-zoom-panel__canvas"
        width={SIZE}
        height={SIZE}
        aria-hidden="true"
      />
    </div>
  );
}
