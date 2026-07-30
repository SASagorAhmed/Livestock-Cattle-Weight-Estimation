import type { ReactNode } from 'react';

interface Props {
  title: string;
  subtitle?: string;
  status?: ReactNode;
  scanning?: boolean;
  children?: ReactNode;
  footer?: ReactNode;
}

export default function LiveProcessingLayout({
  title,
  subtitle,
  status,
  scanning = false,
  children,
  footer,
}: Props) {
  return (
    <div className="live-card live-layout">
      <div>
        <h1 className="live-title">{title}</h1>
        {subtitle ? <p className="live-sub">{subtitle}</p> : null}
      </div>
      {status ? (
        <div className={`status-banner ${scanning ? 'scanning' : ''}`}>{status}</div>
      ) : null}
      {children}
      {footer}
    </div>
  );
}
