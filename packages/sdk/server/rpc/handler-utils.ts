import {
  type QvacConfig,
  type Request,
  type Response,
  type RuntimeContext,
  type ProfilingRequestMeta,
  PROFILING_KEY,
} from "@/schemas";
import type RPC from "bare-rpc";
import {
  sendErrorResponse,
  sendStreamErrorResponse,
} from "@/server/error-handlers";
import { PluginHandlerTypeMismatchError } from "@/utils/errors-server";
import { setSDKConfig } from "@/server/bare/registry/config-registry";
import { setRuntimeContext } from "@/server/bare/registry/runtime-context-registry";
import { getServerLogger } from "@/logging";
import { type ServerProfiler } from "./profiling";
import { isTerminalChunk } from "./rpc-utils";
import { createProgressThrottle } from "./progress-throttle";
import {
  trackHandlerStart,
  trackHandlerEnd,
  awaitHandlerDrain,
} from "./handler-drain";

// Re-export the drain helpers so existing callers can keep importing from
// handler-utils. The implementations live in handler-drain.ts which is a
// pure module (no logger / no Bare runtime imports), making it safe to
// unit-test under any JS runtime.
export {
  trackHandlerStart,
  trackHandlerEnd,
  awaitHandlerDrain,
} from "./handler-drain";

const logger = getServerLogger();

function getProfilingMetaFromRequest(
  request: Request,
): ProfilingRequestMeta | undefined {
  if (PROFILING_KEY in request) {
    return (request as Record<string, unknown>)[
      PROFILING_KEY
    ] as ProfilingRequestMeta;
  }
  return undefined;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
type ReplyHandler = (
  request: any,
  ...args: any[]
) => Promise<Response> | Response;
type StreamHandler = (request: any, ...args: any[]) => AsyncGenerator<Response>;
type ProgressHandler = (request: any, ...args: any[]) => Promise<Response>;
type DuplexStreamHandler = (request: any, inputStream: any) => AsyncGenerator<Response>;
/* eslint-enable @typescript-eslint/no-explicit-any */

export type HandlerEntry = {
  type: "reply" | "stream" | "duplex";
  handler: ReplyHandler | StreamHandler | ProgressHandler | DuplexStreamHandler;
  delegatedHandler?: ReplyHandler | StreamHandler | ProgressHandler | DuplexStreamHandler;
  isDelegated?: (request: Request) => boolean;
  supportsProgress?: boolean | ((request: Request) => boolean);
};

async function executeReplyHandler(
  req: RPC.IncomingRequest,
  request: Request,
  handler: ReplyHandler,
  profiler: ServerProfiler,
  isDelegated: boolean,
) {
  profiler.startHandler();
  try {
    let response: Response;
    if (isDelegated) {
      const profilingMeta = getProfilingMetaFromRequest(request);
      response = await handler(
        request,
        profilingMeta ? { profilingMeta } : undefined,
      );
    } else {
      response = await handler(request);
    }
    profiler.endHandler();
    req.reply(profiler.serialize(response, true), "utf-8");
  } catch (error) {
    profiler.endHandler();
    sendErrorResponse(req, error, profiler);
  }
}

async function executeStreamHandler(
  req: RPC.IncomingRequest,
  request: Request,
  handler: StreamHandler,
  profiler: ServerProfiler,
  isDelegated: boolean,
) {
  const stream = req.createResponseStream();
  profiler.startHandler();
  let sentFinalChunk = false;

  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let generator: AsyncGenerator<Response, any, any>;
    if (isDelegated) {
      const profilingMeta = getProfilingMetaFromRequest(request);
      generator = handler(
        request,
        profilingMeta ? { profilingMeta } : undefined,
      );
    } else {
      generator = handler(request);
    }

    for await (const response of generator) {
      if (isTerminalChunk(response)) {
        profiler.endHandler();
        stream.write(profiler.serialize(response, true) + "\n", "utf-8");
        sentFinalChunk = true;
      } else {
        stream.write(profiler.serialize(response, false) + "\n", "utf-8");
      }
    }
    
    // Fallback
    if (!sentFinalChunk) {
      profiler.endHandler();
      const trailer = profiler.serialize();
      if (trailer) {
        stream.write(trailer + "\n", "utf-8");
      }
    }

    stream.end();
  } catch (error) {
    profiler.endHandler();
    sendStreamErrorResponse(stream, error, profiler);
  }
}

async function executeProgressHandler(
  req: RPC.IncomingRequest,
  request: Request,
  handler: ProgressHandler,
  profiler: ServerProfiler,
  isDelegated: boolean,
) {
  const stream = req.createResponseStream();
  profiler.startHandler();

  const writeBatch = (updates: Response[]) => {
    const payload = updates
      .map((u) => profiler.serialize(u, false))
      .join("\n");
    stream.write(payload + "\n", "utf-8");
  };

  const throttle = createProgressThrottle<Response>(writeBatch);

  try {
    let response: Response;
    if (isDelegated) {
      const profilingMeta = getProfilingMetaFromRequest(request);
      const options: {
        progressCallback: typeof throttle.push;
        profilingMeta?: ProfilingRequestMeta;
      } = { progressCallback: throttle.push };
      if (profilingMeta) {
        options.profilingMeta = profilingMeta;
      }
      response = await handler(request, options);
    } else {
      response = await handler(request, throttle.push);
    }
    throttle.flush();
    profiler.endHandler();
    stream.write(profiler.serialize(response, true) + "\n", "utf-8");
    stream.end();
  } catch (error) {
    throttle.flush();
    profiler.endHandler();
    sendStreamErrorResponse(stream, error, profiler);
  }
}

export async function executeDuplexHandler(
  _req: RPC.IncomingRequest,
  request: Request,
  entry: HandlerEntry,
  inputStream: ReturnType<RPC.IncomingRequest["createRequestStream"]>,
  outputStream: ReturnType<RPC.IncomingRequest["createResponseStream"]>,
  profiler: ServerProfiler,
) {
  const handler =
    entry.delegatedHandler && entry.isDelegated?.(request)
      ? entry.delegatedHandler
      : entry.handler;

  trackHandlerStart();
  profiler.startHandler();

  try {
    for await (const response of (handler as DuplexStreamHandler)(
      request,
      inputStream,
    )) {
      outputStream.write(
        profiler.serialize(response, false) + "\n",
        "utf-8",
      );
    }
    profiler.endHandler();
    const trailer = profiler.serialize();
    if (trailer) {
      outputStream.write(trailer + "\n", "utf-8");
    }
    outputStream.end();
  } catch (error) {
    profiler.endHandler();
    sendStreamErrorResponse(outputStream, error, profiler);
  } finally {
    trackHandlerEnd();
  }
}

// Unified handler executor with delegation and progress support
export async function executeHandler(
  req: RPC.IncomingRequest,
  request: Request,
  entry: HandlerEntry,
  profiler: ServerProfiler,
) {
  const isDelegated = !!(
    entry.delegatedHandler && entry.isDelegated?.(request)
  );
  const handler = isDelegated ? entry.delegatedHandler! : entry.handler;

  const wantsProgress =
    "withProgress" in request &&
    request.withProgress &&
    (typeof entry.supportsProgress === "function"
      ? entry.supportsProgress(request)
      : entry.supportsProgress);

  if (entry.type === "duplex") {
    throw new PluginHandlerTypeMismatchError(
      request.type,
      "reply or stream",
      "duplex",
    );
  }

  trackHandlerStart();
  try {
    if (entry.type === "stream") {
      await executeStreamHandler(
        req,
        request,
        handler as StreamHandler,
        profiler,
        isDelegated,
      );
    } else if (wantsProgress) {
      await executeProgressHandler(
        req,
        request,
        handler as ProgressHandler,
        profiler,
        isDelegated,
      );
    } else {
      await executeReplyHandler(
        req,
        request,
        handler as ReplyHandler,
        profiler,
        isDelegated,
      );
    }
  } finally {
    trackHandlerEnd();
  }
}

// Internal config initialization (bypasses schema)
type InitConfigMessage = {
  type: "__init_config";
  config: QvacConfig;
  runtimeContext?: RuntimeContext;
};

export function isInitConfigMessage(data: unknown): data is InitConfigMessage {
  return (
    typeof data === "object" &&
    data !== null &&
    "type" in data &&
    data.type === "__init_config"
  );
}

export function handleInitConfig(
  req: RPC.IncomingRequest,
  data: InitConfigMessage,
) {
  try {
    if (data.config) {
      setSDKConfig(data.config);
    }
    if (data.runtimeContext) {
      setRuntimeContext(data.runtimeContext);
    }
    req.reply(JSON.stringify({ success: true }), "utf-8");
  } catch (error) {
    req.reply(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      }),
      "utf-8",
    );
  }
}

// Internal pre-terminate cleanup signal. The SDK client sends this before
// tearing down the bare runtime (e.g. Worklet.terminate() on mobile) so
// addons can release env-bound state while their JS environment is still
// alive. Reply success/failure, never throws to the dispatcher.
type ShutdownMessage = {
  type: "__shutdown__";
};

export function isShutdownMessage(data: unknown): data is ShutdownMessage {
  return (
    typeof data === "object" &&
    data !== null &&
    "type" in data &&
    (data as { type?: unknown }).type === "__shutdown__"
  );
}

// Drain budget for in-flight handlers before pre-terminate cleanup runs.
// Strictly below the 10s SHUTDOWN_RPC_TIMEOUT_MS in expo-rpc-client so the
// client never sees a phantom timeout from this gate; remaining ~2s is
// reserved for the actual cleanup body (unloadAllModels, registry teardown).
const SHUTDOWN_HANDLER_DRAIN_MS = 8000;

export async function handleShutdown(req: RPC.IncomingRequest): Promise<void> {
  try {
    // Wait for any in-flight handler (e.g. a transcribe call mid-whisper_full
    // on a JobRunner C++ thread) to complete before tearing down addons.
    // unloadAllModels invokes WhisperModel::unload -> whisper_free(ctx); if
    // that races with whisper_full(ctx, ...) the addon use-after-frees and
    // the iOS host process is killed by the kernel (Mach exception 309).
    //
    // Best-effort: on timeout, warn and proceed so a wedged handler cannot
    // indefinitely block client teardown.
    const drained = await awaitHandlerDrain(SHUTDOWN_HANDLER_DRAIN_MS);
    if (!drained) {
      logger.warn(
        `Pre-terminate handler drain timed out after ${SHUTDOWN_HANDLER_DRAIN_MS}ms; proceeding with cleanup anyway`,
      );
    }

    // Lazy import to avoid the import cycle:
    //   handler-utils -> worker-core -> create-server -> handle-request
    //   -> handler-utils. By the time this runs, all modules are loaded.
    const { cleanupForTerminate } = await import("@/server/worker-core");
    await cleanupForTerminate();
    req.reply(JSON.stringify({ success: true }), "utf-8");
  } catch (error) {
    req.reply(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      }),
      "utf-8",
    );
  }
}
