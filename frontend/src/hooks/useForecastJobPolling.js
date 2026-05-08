import { useCallback, useMemo, useState } from "react";
import { getForecastJobStatus, startForecastJob } from "../services/api";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function useForecastJobPolling() {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [isPolling, setIsPolling] = useState(false);

  const runForecastAsync = useCallback(async ({
    sessionId,
    horizonHours = 24,
    trainSize = 80,
    skipFuture = false,
    pollIntervalMs = 2000,
    timeoutMs = 15 * 60 * 1000,
  }) => {
    if (!sessionId) {
      throw new Error("sessionId is required");
    }

    setStatus("starting");
    setError(null);
    setResult(null);

    await startForecastJob(sessionId, horizonHours, trainSize, skipFuture);

    setStatus("processing");
    setIsPolling(true);

    const startedAt = Date.now();

    while (Date.now() - startedAt < timeoutMs) {
      const job = await getForecastJobStatus(sessionId);
      const nextStatus = job?.status || "processing";
      setStatus(nextStatus);

      if (nextStatus === "completed") {
        setIsPolling(false);
        setResult(job.result || null);
        return job.result;
      }

      if (nextStatus === "failed") {
        const nextError = job?.error || "Forecast job failed";
        setIsPolling(false);
        setError(nextError);
        throw new Error(nextError);
      }

      await sleep(pollIntervalMs);
    }

    setIsPolling(false);
    setStatus("failed");
    setError("Forecast job timed out while polling status endpoint");
    throw new Error("Forecast job timed out");
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setError(null);
    setResult(null);
    setIsPolling(false);
  }, []);

  const isBusy = useMemo(() => {
    return status === "starting" || status === "processing" || isPolling;
  }, [status, isPolling]);

  return {
    status,
    error,
    result,
    isPolling,
    isBusy,
    runForecastAsync,
    reset,
  };
}
