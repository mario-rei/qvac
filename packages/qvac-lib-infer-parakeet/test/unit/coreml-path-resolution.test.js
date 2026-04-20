'use strict'

const path = require('bare-path')
const fs = require('bare-fs')
const os = require('bare-os')
const test = require('brittle')
const sinon = require('sinon')
const TranscriptionParakeet = require('../../index.js')
const { resolveCoremlModelPath } = require('../../index.js')
const { ParakeetInterface } = require('../../parakeet')
const MockedBinding = require('../mocks/MockedBinding.js')
const { transitionCb } = require('../mocks/utils.js')
const process = require('process')

global.process = process

const tmpDir = path.join(os.tmpdir(), 'coreml-parakeet-test-' + Date.now())

test('setup: create temp model fixtures', (t) => {
  fs.mkdirSync(tmpDir, { recursive: true })

  // TDT-style: .onnx.data (dot-separated) + coreml variant
  fs.writeFileSync(path.join(tmpDir, 'encoder-model.onnx'), 'fake')
  fs.writeFileSync(path.join(tmpDir, 'encoder-model.onnx.data'), 'fake')
  fs.writeFileSync(path.join(tmpDir, 'encoder-model_coreml.onnx'), 'fake')

  // CTC-style: .onnx_data (underscore) + coreml variant
  fs.writeFileSync(path.join(tmpDir, 'model.onnx'), 'fake')
  fs.writeFileSync(path.join(tmpDir, 'model.onnx_data'), 'fake')
  fs.writeFileSync(path.join(tmpDir, 'model_coreml.onnx'), 'fake')

  // External data but no coreml variant
  fs.writeFileSync(path.join(tmpDir, 'big_encoder.onnx'), 'fake')
  fs.writeFileSync(path.join(tmpDir, 'big_encoder.onnx.data'), 'fake')

  // Single-file model (no external data)
  fs.writeFileSync(path.join(tmpDir, 'decoder_joint-model.onnx'), 'fake')

  // Non-model files
  fs.writeFileSync(path.join(tmpDir, 'vocab.txt'), 'fake')
  fs.writeFileSync(path.join(tmpDir, 'tokenizer.json'), '{}')

  t.pass('fixtures created')
})

// --- Direct unit tests for resolveCoremlModelPath ---

test('resolveCoremlModelPath: resolves .onnx.data (TDT-style) to _coreml variant', (t) => {
  const input = path.join(tmpDir, 'encoder-model.onnx')
  const expected = path.join(tmpDir, 'encoder-model_coreml.onnx')
  t.is(resolveCoremlModelPath(input), expected)
})

test('resolveCoremlModelPath: resolves .onnx_data (CTC-style) to _coreml variant', (t) => {
  const input = path.join(tmpDir, 'model.onnx')
  const expected = path.join(tmpDir, 'model_coreml.onnx')
  t.is(resolveCoremlModelPath(input), expected)
})

test('resolveCoremlModelPath: returns original when external data exists but no _coreml variant', (t) => {
  const input = path.join(tmpDir, 'big_encoder.onnx')
  t.is(resolveCoremlModelPath(input), input)
})

test('resolveCoremlModelPath: returns original for single-file model', (t) => {
  const input = path.join(tmpDir, 'decoder_joint-model.onnx')
  t.is(resolveCoremlModelPath(input), input)
})

test('resolveCoremlModelPath: returns falsy input as-is', (t) => {
  t.is(resolveCoremlModelPath(''), '')
  t.is(resolveCoremlModelPath(null), null)
  t.is(resolveCoremlModelPath(undefined), undefined)
})

test('resolveCoremlModelPath: returns original for non-existent model', (t) => {
  const input = path.join(tmpDir, 'no_such_model.onnx')
  t.is(resolveCoremlModelPath(input), input)
})

// --- Integration: _buildConfigurationParams resolves paths on darwin with useGPU ---

const IS_DARWIN = os.platform() === 'darwin'

function createParakeetModel (useGPU, modelType) {
  // Restore any leftover prototype stub from other test files
  if (TranscriptionParakeet.prototype.validateModelFiles.restore) {
    TranscriptionParakeet.prototype.validateModelFiles.restore()
  }
  const validateStub = sinon.stub(TranscriptionParakeet.prototype, 'validateModelFiles').returns(undefined)

  const model = new TranscriptionParakeet({
    files: {
      encoder: path.join(tmpDir, 'encoder-model.onnx'),
      encoderData: path.join(tmpDir, 'encoder-model.onnx.data'),
      decoder: path.join(tmpDir, 'decoder_joint-model.onnx'),
      vocab: path.join(tmpDir, 'vocab.txt'),
      preprocessor: path.join(tmpDir, 'decoder_joint-model.onnx'),
      model: path.join(tmpDir, 'model.onnx'),
      modelData: path.join(tmpDir, 'model.onnx_data'),
      tokenizer: path.join(tmpDir, 'tokenizer.json')
    },
    config: {
      parakeetConfig: {
        modelType: modelType || 'tdt',
        maxThreads: 4,
        useGPU
      }
    }
  })

  let capturedParams = null
  sinon.stub(model, '_createAddon').callsFake((params) => {
    capturedParams = params
    return new ParakeetInterface(new MockedBinding(), params, () => {}, transitionCb)
  })

  model._validateStub = validateStub
  return { model, getCapturedParams: () => capturedParams }
}

test('_buildConfigurationParams with useGPU:true resolves ONNX paths on darwin', { skip: !IS_DARWIN }, async (t) => {
  const { model, getCapturedParams } = createParakeetModel(true, 'tdt')
  try {
    await model.load()
    const params = getCapturedParams()

    t.is(
      params.encoderPath,
      path.join(tmpDir, 'encoder-model_coreml.onnx'),
      'encoder with .onnx.data + _coreml variant should resolve'
    )
    t.is(
      params.decoderPath,
      path.join(tmpDir, 'decoder_joint-model.onnx'),
      'decoder without external data should stay unchanged'
    )
    t.is(
      params.encoderDataPath,
      path.join(tmpDir, 'encoder-model.onnx.data'),
      'encoderDataPath should never be resolved'
    )
    t.is(
      params.vocabPath,
      path.join(tmpDir, 'vocab.txt'),
      'vocabPath should never be resolved'
    )
    t.is(
      params.ctcModelPath,
      path.join(tmpDir, 'model_coreml.onnx'),
      'CTC model with .onnx_data + _coreml variant should resolve'
    )
    t.is(
      params.tokenizerPath,
      path.join(tmpDir, 'tokenizer.json'),
      'tokenizerPath should never be resolved'
    )
  } finally {
    model._validateStub.restore()
    await model.destroy()
  }
})

test('_buildConfigurationParams with useGPU:false does NOT resolve paths', async (t) => {
  const { model, getCapturedParams } = createParakeetModel(false, 'tdt')
  try {
    await model.load()
    const params = getCapturedParams()

    t.is(
      params.encoderPath,
      path.join(tmpDir, 'encoder-model.onnx'),
      'paths should not be altered when useGPU is false'
    )
    t.is(
      params.ctcModelPath,
      path.join(tmpDir, 'model.onnx'),
      'CTC model path should not be altered when useGPU is false'
    )
  } finally {
    model._validateStub.restore()
    await model.destroy()
  }
})

test('cleanup: remove temp fixtures', (t) => {
  try {
    const files = fs.readdirSync(tmpDir)
    for (const f of files) fs.unlinkSync(path.join(tmpDir, f))
    fs.rmdirSync(tmpDir)
  } catch (_) {}
  t.pass('cleaned up')
})
