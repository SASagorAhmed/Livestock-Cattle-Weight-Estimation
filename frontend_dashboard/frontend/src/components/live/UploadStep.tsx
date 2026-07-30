import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react';
import type { StartDetectionInput } from '../../types';

interface Props {
  busy: boolean;
  onStart: (input: StartDetectionInput) => void;
}

export default function UploadStep({ onStart, busy }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [camError, setCamError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [camOn, setCamOn] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  const onDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }, []);

  const clearImage = () => {
    setFile(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const startCam = async () => {
    setCamError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });
      streamRef.current = stream;
      setCamOn(true);
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          void videoRef.current.play();
        }
      });
    } catch (err) {
      setCamError(err instanceof Error ? err.message : 'Camera unavailable');
    }
  };

  const captureCam = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      setFile(new File([blob], `webcam_${Date.now()}.jpg`, { type: 'image/jpeg' }));
      streamRef.current?.getTracks().forEach((t) => t.stop());
      setCamOn(false);
    }, 'image/jpeg', 0.92);
  };

  return (
    <div className="live-card">
      <h1 className="live-title">Cow Weight Detection</h1>
      <p className="live-sub">
        Upload a clear side-view photo of a cow. Detection runs offline on this machine,
        one guided step at a time.
      </p>

      <div
        className={`dropzone ${drag ? 'active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <h3>Drop cattle image here</h3>
        <p>JPG or PNG · max 15 MB</p>
        <div className="btn-row" style={{ justifyContent: 'center' }}>
          <button type="button" className="btn btn-primary" onClick={() => inputRef.current?.click()}>
            Browse Image
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => void startCam()}>
            Use webcam
          </button>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,.jpg,.jpeg,.png"
          hidden
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
      </div>

      {camError ? <div className="error-box" style={{ marginTop: '1rem' }}>{camError}</div> : null}

      {camOn ? (
        <div className="preview-wrap" style={{ marginTop: '1rem' }}>
          <video ref={videoRef} playsInline muted style={{ width: '100%', maxHeight: 360 }} />
          <div className="btn-row" style={{ padding: '0.75rem', background: 'white' }}>
            <button type="button" className="btn btn-primary" onClick={captureCam}>Capture</button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => {
                streamRef.current?.getTracks().forEach((t) => t.stop());
                setCamOn(false);
              }}
            >
              Cancel camera
            </button>
          </div>
        </div>
      ) : null}

      {preview ? (
        <div className="preview-wrap">
          <img src={preview} alt="Upload preview" />
        </div>
      ) : null}

      {file ? (
        <div className="btn-row">
          <button type="button" className="btn btn-ghost" onClick={clearImage}>
            Replace Image
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => inputRef.current?.click()}>
            Browse Image
          </button>
        </div>
      ) : null}

      <div className="options-grid">
        <div className="option-block" style={{ border: '1px solid var(--fb-border)', borderRadius: 12, padding: '1rem' }}>
          <strong>Model: Cow Morpho Heuristic</strong>
          <span className="chip" style={{ marginLeft: 8 }}>Experimental</span>
          <p className="live-sub" style={{ marginTop: '0.5rem', marginBottom: 0 }}>
            Uses cow detection, body mask, and pose internally to suggest four body points,
            then estimates weight from a reference scale.
          </p>
        </div>
      </div>

      <div className="tips">
        Image guidance
        <ul>
          <li>Full cow should be visible</li>
          <li>Side-view image is preferred</li>
          <li>Cow should be standing</li>
          <li>Avoid blurry images</li>
          <li>Avoid heavy obstruction</li>
        </ul>
      </div>

      <div className="btn-row">
        <button
          type="button"
          className="btn btn-primary"
          disabled={!file || busy}
          onClick={() => {
            if (!file) return;
            onStart({
              file,
              enableSeg: true,
              predictionMode: 'smartphone_diagonal',
              localPreview: preview,
            });
          }}
        >
          Start Weight Detection
        </button>
      </div>
    </div>
  );
}
