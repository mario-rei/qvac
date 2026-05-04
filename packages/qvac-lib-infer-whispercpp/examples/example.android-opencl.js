'use strict'

/**
 * Example: Android OpenCL transcription
 *
 * This example only works on Android devices with an Adreno-class GPU and a
 * working OpenCL ICD. On other platforms `use_gpu: true` will simply pick
 * Metal / Vulkan / CPU instead, and the new top-level options (`openclCacheDir`,
 * `backendsDir`) are no-ops everywhere except Android.
 *
 * Usage:
 *   bare examples/example.android-opencl.js <modelPath> <audioPath> [openclCacheDir]
 *
 * The optional 3rd argument overrides the default OpenCL kernel cache
 * directory; otherwise the example writes to a per-app cache path under
 * /data/data/io.tether.test.qvac/cache/whisper. The directory must be
 * writable by the app's UID.
 *
 * What to look for in the runtime stats:
 *  - `realTimeFactor` should drop noticeably vs. CPU on the second run, once
 *    `GGML_OPENCL_CACHE_DIR` is populated and the JIT compile cost is paid.
 *  - The first run will be slower because OpenCL kernels are JIT-compiled
 *    and cached under `<openclCacheDir>/opencl-cache`.
 */

const fs = require('bare-fs')
const process = require('bare-process')

const TranscriptionWhispercpp = require('../index.js')
const binding = require('../binding.js')

const LOG_PRIORITIES = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
binding.setLogger((priority, message) => {
  const priorityName = LOG_PRIORITIES[priority] || `UNKNOWN(${priority})`
  console.log(`[C++ ${priorityName}] ${message}`)
})

const DEFAULT_OPENCL_CACHE_DIR = '/data/data/io.tether.test.qvac/cache/whisper'

function parseArgs () {
  const args = process.argv.slice(2)
  const modelPath = args[0]
  const audioPath = args[1]
  const openclCacheDir = args[2] || DEFAULT_OPENCL_CACHE_DIR

  if (!modelPath || !audioPath) {
    console.error('Usage: bare examples/example.android-opencl.js <modelPath> <audioPath> [openclCacheDir]')
    process.exit(1)
  }

  return { modelPath, audioPath, openclCacheDir }
}

function assertFileExists (filePath, label) {
  if (!fs.existsSync(filePath)) {
    console.error(`${label} not found at ${filePath}`)
    process.exit(1)
  }
}

async function main () {
  const { modelPath, audioPath, openclCacheDir } = parseArgs()
  assertFileExists(modelPath, 'Model file')
  assertFileExists(audioPath, 'Audio file')

  // Best-effort: make sure the cache directory exists. We don't fail if it
  // can't be created because the C++ side will simply skip setenv when the
  // path is empty, and ggml will fall back to its default cache location.
  try {
    fs.mkdirSync(openclCacheDir, { recursive: true })
  } catch (err) {
    console.warn(`[warn] could not create openclCacheDir at ${openclCacheDir}: ${err.message}`)
  }

  console.log('=== Whisper Android OpenCL Example ===\n')
  console.log(`Model:           ${modelPath}`)
  console.log(`Audio:           ${audioPath}`)
  console.log(`OpenCL cacheDir: ${openclCacheDir}`)

  const constructorArgs = {
    files: { model: modelPath },
    opts: { stats: true }
  }

  const config = {
    whisperConfig: {
      audio_format: 's16le',
      language: 'en',
      // GPU is opt-in across all backends; on Android with an Adreno ICD this
      // will route through the ggml OpenCL backend. On other platforms the
      // same flag selects Metal / Vulkan transparently.
      use_gpu: true
    },
    contextParams: {
      use_gpu: true
    },
    // Top-level addon options consumed at backend-init time. Ignored on
    // non-Android platforms.
    openclCacheDir
  }

  const model = new TranscriptionWhispercpp(constructorArgs, config)
  await model.load()

  // Stream a single chunk from the audio file (assumed to be raw 16 kHz mono
  // s16le, matching `audio_format` above). For end-to-end audio decoding see
  // examples/example.live-transcription.js.
  const audioBuffer = fs.readFileSync(audioPath)

  const response = await model.run([new Uint8Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.byteLength)])

  const transcripts = []
  response.onUpdate(out => {
    const items = Array.isArray(out) ? out : [out]
    for (const item of items) {
      if (item && typeof item.text === 'string') transcripts.push(item)
    }
  })

  const result = await response.await()

  console.log('\n=== Transcript ===')
  for (const t of transcripts) {
    const text = (t.text || '').trim()
    if (text) console.log(`[${t.start?.toFixed?.(2)}s -> ${t.end?.toFixed?.(2)}s] ${text}`)
  }

  // `result.stats` is populated when constructor args set `opts: { stats: true }`.
  // The stats include realTimeFactor / tokensPerSecond which are the easiest
  // way to confirm OpenCL was selected: a real GPU run on Adreno should be
  // noticeably faster than the CPU baseline.
  if (result && result.stats) {
    console.log('\n=== Runtime stats ===')
    console.log(JSON.stringify(result.stats, null, 2))
  }

  await model.destroy()
  binding.releaseLogger()
}

main().catch(err => {
  console.error(err)
  binding.releaseLogger()
  process.exit(1)
})
