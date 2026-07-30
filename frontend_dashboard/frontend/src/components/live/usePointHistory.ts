import { useCallback, useState } from 'react';
import type { ImagePoint } from '../../types';

export type SelectedPointId = 'A' | 'B' | null;

export type PointSnapshot = {
  pointA: ImagePoint | null;
  pointB: ImagePoint | null;
  selectedPoint: SelectedPointId;
};

export function usePointHistory(initial: PointSnapshot = {
  pointA: null,
  pointB: null,
  selectedPoint: null,
}) {
  const [history, setHistory] = useState<PointSnapshot[]>([]);
  const [future, setFuture] = useState<PointSnapshot[]>([]);

  const pushSnapshot = useCallback((snapshot: PointSnapshot) => {
    setHistory((h) => [...h, {
      pointA: snapshot.pointA ? { ...snapshot.pointA } : null,
      pointB: snapshot.pointB ? { ...snapshot.pointB } : null,
      selectedPoint: snapshot.selectedPoint,
    }]);
    setFuture([]);
  }, []);

  const undo = useCallback((current: PointSnapshot): PointSnapshot | null => {
    if (history.length === 0) return null;
    const prev = history[history.length - 1];
    setHistory((h) => h.slice(0, -1));
    setFuture((f) => [
      {
        pointA: current.pointA ? { ...current.pointA } : null,
        pointB: current.pointB ? { ...current.pointB } : null,
        selectedPoint: current.selectedPoint,
      },
      ...f,
    ]);
    return {
      pointA: prev.pointA ? { ...prev.pointA } : null,
      pointB: prev.pointB ? { ...prev.pointB } : null,
      selectedPoint: prev.selectedPoint,
    };
  }, [history]);

  const redo = useCallback((current: PointSnapshot): PointSnapshot | null => {
    if (future.length === 0) return null;
    const next = future[0];
    setFuture((f) => f.slice(1));
    setHistory((h) => [
      ...h,
      {
        pointA: current.pointA ? { ...current.pointA } : null,
        pointB: current.pointB ? { ...current.pointB } : null,
        selectedPoint: current.selectedPoint,
      },
    ]);
    return {
      pointA: next.pointA ? { ...next.pointA } : null,
      pointB: next.pointB ? { ...next.pointB } : null,
      selectedPoint: next.selectedPoint,
    };
  }, [future]);

  const clearHistory = useCallback(() => {
    setHistory([]);
    setFuture([]);
  }, []);

  return {
    history,
    future,
    canUndo: history.length > 0,
    canRedo: future.length > 0,
    pushSnapshot,
    undo,
    redo,
    clearHistory,
    initial,
  };
}
