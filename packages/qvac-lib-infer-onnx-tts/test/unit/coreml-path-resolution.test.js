'use strict'

const path = require('bare-path')
const fs = require('bare-fs')
const os = require('bare-os')
const test = require('brittle')
const sinon = require('sinon')
const ONNXTTS = require('../../index.js')
const { resolveCoremlModelPath } = require('../../index.js')
const { TTSInterface } = require('../../tts.js')
const MockedBinding = require('../mock/MockedBinding.js')
const process = require('process')

global.process = process

const tmpDir = path.join(os.tmpdir(), 'coreml-path-test-' + Date.now())

test('setup: create temp model fixtures', (t) => {
  fs.mkdirSync(tmpDir, { recursive: true })

  fs.writeFileSync(path.join(tmpDir, 'speech_encoder_q4.onnx'), 'fake')
  fs.writeFileSync(path.join(tmpDir, 'speech_encoder_q4.onnx_data'), 'fake')
  fs.writeFileSync(path.join(tmpDir, 'speech_encoder_q4_coreml.onnx'), 'fake')

  fs.writeFileSync(path.join(tmpDir, 'big_model.onnx'), 'fake')
  fs.writeFileSync(path.join(tmpDir, 'big_model.onnx_data'), 'fake')

  fs.writeFileSync(path.join(tmpDir, 'embed_tokens.onnx'), 'fake')

  fs.writeFileSync(path.join(tmpDir, 'tokenizer.json'), '{}')

  t.pass('fixtures created')
})

// --- Direct unit tests for resolveCoremlModelPath ---

test('resolveCoremlModelPath: returns _coreml variant when .onnx_data and variant both exist', (t) => {
  const input = path.join(tmpDir, 'speech_encoder_q4.onnx')
  const expected = path.join(tmpDir, 'speech_encoder_q4_coreml.onnx')
  t.is(resolveCoremlModelPath(input), expected)
})

test('resolveCoremlModelPath: returns original path when .onnx_data exists but no _coreml variant', (t) => {
  const input = path.join(tmpDir, 'big_model.onnx')
  t.is(resolveCoremlModelPath(input), input)
})

test('resolveCoremlModelPath: returns original path when no .onnx_data exists', (t) => {
  const input = path.join(tmpDir, 'embed_tokens.onnx')
  t.is(resolveCoremlModelPath(input), input)
})

test('resolveCoremlModelPath: returns falsy input as-is', (t) => {
  t.is(resolveCoremlModelPath(''), '')
  t.is(resolveCoremlModelPath(null), null)
  t.is(resolveCoremlModelPath(undefined), undefined)
})

test('resolveCoremlModelPath: returns original path for non-existent model', (t) => {
  const input = path.join(tmpDir, 'no_such_model.onnx')
  t.is(resolveCoremlModelPath(input), input)
})

// --- Integration: verify _load wires resolveCoremlModelPath for Chatterbox ---

const IS_DARWIN = os.platform() === 'darwin'

function createChatterboxModel (useGPU) {
  const model = new ONNXTTS({
    files: {
      tokenizer: path.join(tmpDir, 'tokenizer.json'),
      speechEncoder: path.join(tmpDir, 'speech_encoder_q4.onnx'),
      embedTokens: path.join(tmpDir, 'embed_tokens.onnx'),
      conditionalDecoder: path.join(tmpDir, 'big_model.onnx'),
      languageModel: path.join(tmpDir, 'speech_encoder_q4.onnx')
    },
    engine: 'chatterbox',
    config: { language: 'en', useGPU }
  })

  let capturedParams = null
  sinon.stub(model, '_createAddon').callsFake((params, outputCb) => {
    capturedParams = params
    return new TTSInterface(new MockedBinding(), params, outputCb)
  })

  return { model, getCapturedParams: () => capturedParams }
}

test('_load with useGPU:true resolves external-data paths on darwin', { skip: !IS_DARWIN }, async (t) => {
  const { model, getCapturedParams } = createChatterboxModel(true)
  await model.load()
  const params = getCapturedParams()

  t.is(
    params.speechEncoderPath,
    path.join(tmpDir, 'speech_encoder_q4_coreml.onnx'),
    'speechEncoder with external data + _coreml variant should resolve'
  )
  t.is(
    params.embedTokensPath,
    path.join(tmpDir, 'embed_tokens.onnx'),
    'embedTokens without external data should stay unchanged'
  )
  t.is(
    params.conditionalDecoderPath,
    path.join(tmpDir, 'big_model.onnx'),
    'conditionalDecoder with external data but no _coreml variant stays unchanged'
  )
  t.is(
    params.languageModelPath,
    path.join(tmpDir, 'speech_encoder_q4_coreml.onnx'),
    'languageModel should resolve to _coreml variant'
  )

  await model.destroy()
})

test('_load with useGPU:false does NOT resolve paths', async (t) => {
  const { model, getCapturedParams } = createChatterboxModel(false)
  await model.load()
  const params = getCapturedParams()

  t.is(
    params.speechEncoderPath,
    path.join(tmpDir, 'speech_encoder_q4.onnx'),
    'paths should not be altered when useGPU is false'
  )

  await model.destroy()
})

test('reload with useGPU:true resolves paths on darwin', { skip: !IS_DARWIN }, async (t) => {
  const { model, getCapturedParams } = createChatterboxModel(false)
  await model.load()

  t.is(
    getCapturedParams().speechEncoderPath,
    path.join(tmpDir, 'speech_encoder_q4.onnx'),
    'initial load with useGPU:false should not resolve'
  )

  await model.reload({ useGPU: true })

  t.is(
    getCapturedParams().speechEncoderPath,
    path.join(tmpDir, 'speech_encoder_q4_coreml.onnx'),
    'reload with useGPU:true should resolve to _coreml variant'
  )

  await model.destroy()
})

test('cleanup: remove temp fixtures', (t) => {
  try {
    const files = fs.readdirSync(tmpDir)
    for (const f of files) fs.unlinkSync(path.join(tmpDir, f))
    fs.rmdirSync(tmpDir)
  } catch (_) {}
  t.pass('cleaned up')
})
