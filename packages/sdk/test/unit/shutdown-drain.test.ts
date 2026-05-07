// @ts-ignore brittle has no type declarations
import test from "brittle";
import {
  trackHandlerStart,
  trackHandlerEnd,
  awaitHandlerDrain,
  _getInFlightHandlerCount,
} from "@/server/rpc/handler-drain";

type BrittleT = {
  is: Function;
  ok: Function;
  exception: Function;
  execution: Function;
  not: Function;
  alike: Function;
  teardown: Function;
};

// =============================================================================
// In-flight handler tracking + pre-terminate drain
//
// These helpers gate __shutdown__ behind in-flight handler completion so the
// pre-terminate cleanup hook (which calls unloadAllModels and would free
// native model contexts) cannot run while a request handler is mid-flight on
// a worker C++ thread (e.g. WhisperModel::process inside whisper_full).
//
// Background: iOS Mach exception 309 crash, ~400 ms after FFmpegDecoder
// unloaded, when whisper_free(ctx) ran underneath an active whisper_full.
// =============================================================================

test("trackHandlerStart / trackHandlerEnd: counter increments and decrements", (t: BrittleT) => {
  const before = _getInFlightHandlerCount();
  trackHandlerStart();
  t.is(
    _getInFlightHandlerCount(),
    before + 1,
    "start increments count by 1",
  );
  trackHandlerStart();
  t.is(
    _getInFlightHandlerCount(),
    before + 2,
    "second start increments count again",
  );
  trackHandlerEnd();
  t.is(
    _getInFlightHandlerCount(),
    before + 1,
    "end decrements count by 1",
  );
  trackHandlerEnd();
  t.is(
    _getInFlightHandlerCount(),
    before,
    "end returns count to baseline",
  );
});

test("trackHandlerEnd: extra calls do not underflow the counter", (t: BrittleT) => {
  // Drain to baseline 0 first.
  while (_getInFlightHandlerCount() > 0) trackHandlerEnd();
  t.is(_getInFlightHandlerCount(), 0, "baseline is 0");

  trackHandlerEnd();
  t.is(
    _getInFlightHandlerCount(),
    0,
    "end on 0 does not produce a negative count",
  );
  trackHandlerEnd();
  t.is(_getInFlightHandlerCount(), 0, "end on 0 stays at 0");
});

test("awaitHandlerDrain: resolves immediately with true when no handlers in flight", async (t: BrittleT) => {
  while (_getInFlightHandlerCount() > 0) trackHandlerEnd();
  t.is(_getInFlightHandlerCount(), 0, "no handlers in flight before drain");

  const start = Date.now();
  const drained = await awaitHandlerDrain(1000);
  const elapsed = Date.now() - start;

  t.is(drained, true, "drain resolves true when nothing is in flight");
  t.ok(elapsed < 50, `drain returns immediately (took ${elapsed}ms)`);
});

test("awaitHandlerDrain: resolves with true only after every in-flight handler ends", async (t: BrittleT) => {
  while (_getInFlightHandlerCount() > 0) trackHandlerEnd();

  trackHandlerStart();
  trackHandlerStart();

  let drainSettled = false;
  const drainPromise = awaitHandlerDrain(2000).then((result) => {
    drainSettled = true;
    return result;
  });

  await new Promise((resolve) => setTimeout(resolve, 30));
  t.is(
    drainSettled,
    false,
    "drain has not settled while two handlers are in flight",
  );

  trackHandlerEnd();
  await new Promise((resolve) => setTimeout(resolve, 30));
  t.is(
    drainSettled,
    false,
    "drain still has not settled while one handler remains in flight",
  );

  trackHandlerEnd();
  const drained = await drainPromise;
  t.is(drained, true, "drain resolves true once last handler ends");
  t.is(_getInFlightHandlerCount(), 0, "counter is back at 0");
});

test("awaitHandlerDrain: resolves with false when timeout elapses before drain", async (t: BrittleT) => {
  while (_getInFlightHandlerCount() > 0) trackHandlerEnd();

  trackHandlerStart();

  const start = Date.now();
  const drained = await awaitHandlerDrain(100);
  const elapsed = Date.now() - start;

  t.is(drained, false, "drain resolves false on timeout");
  t.ok(
    elapsed >= 100,
    `drain waited at least the configured timeout (took ${elapsed}ms)`,
  );
  t.ok(
    elapsed < 500,
    `drain did not wait significantly longer than configured (took ${elapsed}ms)`,
  );

  // Cleanup so other tests start from a known baseline.
  trackHandlerEnd();
});

test("awaitHandlerDrain: multiple concurrent waiters all resolve on drain", async (t: BrittleT) => {
  while (_getInFlightHandlerCount() > 0) trackHandlerEnd();

  trackHandlerStart();

  const drainA = awaitHandlerDrain(2000);
  const drainB = awaitHandlerDrain(2000);
  const drainC = awaitHandlerDrain(2000);

  await new Promise((resolve) => setTimeout(resolve, 30));
  trackHandlerEnd();

  const [a, b, c] = await Promise.all([drainA, drainB, drainC]);
  t.is(a, true, "first waiter sees drain");
  t.is(b, true, "second waiter sees drain");
  t.is(c, true, "third waiter sees drain");
});

test("awaitHandlerDrain: a timed-out waiter does not block subsequent drains", async (t: BrittleT) => {
  while (_getInFlightHandlerCount() > 0) trackHandlerEnd();

  trackHandlerStart();
  const fast = await awaitHandlerDrain(50);
  t.is(fast, false, "first call times out while handler is still in flight");

  // Now end the handler. A fresh drain should resolve immediately even
  // though the previous waiter exited via the timeout path.
  trackHandlerEnd();

  const start = Date.now();
  const second = await awaitHandlerDrain(1000);
  const elapsed = Date.now() - start;
  t.is(second, true, "second drain resolves true after handler ends");
  t.ok(
    elapsed < 50,
    `second drain returns immediately (took ${elapsed}ms)`,
  );
});
