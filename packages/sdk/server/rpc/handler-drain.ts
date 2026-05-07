/**
 * In-flight handler accounting + bounded drain helper.
 *
 * Used by `__shutdown__` so that the pre-terminate cleanup hook (which
 * calls unloadAllModels and would free native model contexts) cannot run
 * while a request handler is mid-flight on a worker C++ thread (e.g.
 * WhisperModel::process inside whisper_full).
 *
 * Without this, the addon's whisper_context can be freed under a
 * JobRunner thread that is still calling whisper_full(...), producing an
 * iOS Mach exception (type 309) ~hundreds of ms after the JS path went
 * silent.
 *
 * Pure module: no logging, no I/O, no Bare runtime dependencies. Safe to
 * import from unit tests under any JS runtime.
 */

let inFlightHandlers = 0;
let drainResolvers: Array<() => void> = [];

export function trackHandlerStart(): void {
  inFlightHandlers += 1;
}

export function trackHandlerEnd(): void {
  if (inFlightHandlers > 0) {
    inFlightHandlers -= 1;
  }
  if (inFlightHandlers === 0 && drainResolvers.length > 0) {
    const resolvers = drainResolvers;
    drainResolvers = [];
    for (const resolve of resolvers) {
      try {
        resolve();
      } catch {
        // Drain notifications are best-effort; one bad resolver must not
        // prevent the others from firing.
      }
    }
  }
}

/**
 * Resolves true once all currently-tracked handlers have completed, or
 * false if the drain takes longer than `timeoutMs`. Never throws.
 *
 * Callers that need a hard upper bound (e.g. handleShutdown, which has a
 * 10 s ceiling on the client side) should pass a timeout strictly below
 * that ceiling so they retain time to actually run cleanup.
 */
export function awaitHandlerDrain(timeoutMs: number): Promise<boolean> {
  if (inFlightHandlers === 0) {
    return Promise.resolve(true);
  }

  return new Promise<boolean>((resolve) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const onDrain = () => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      resolve(true);
    };

    drainResolvers.push(onDrain);

    timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      const idx = drainResolvers.indexOf(onDrain);
      if (idx >= 0) drainResolvers.splice(idx, 1);
      resolve(false);
    }, timeoutMs);
  });
}

// Test-only: snapshot the current in-flight count. Not part of the
// public surface; used by shutdown-drain.test.ts to assert behaviour.
export function _getInFlightHandlerCount(): number {
  return inFlightHandlers;
}
