/** SVG text labels readable on photo backgrounds. */

interface OverlayLabelProps {
  x: number;
  y: number;
  text: string;
  fontSize: number;
  anchor?: 'start' | 'middle' | 'end';
  fill?: string;
  stroke?: string;
}

export function OverlayLabel({
  x,
  y,
  text,
  fontSize,
  anchor = 'start',
  fill = '#ffffff',
  stroke = '#1b5e20',
}: OverlayLabelProps) {
  return (
    <text
      x={x}
      y={y}
      fontSize={fontSize}
      textAnchor={anchor}
      fill={fill}
      stroke={stroke}
      strokeWidth={Math.max(1, fontSize * 0.08)}
      paintOrder="stroke"
      fontWeight={600}
    >
      {text}
    </text>
  );
}
