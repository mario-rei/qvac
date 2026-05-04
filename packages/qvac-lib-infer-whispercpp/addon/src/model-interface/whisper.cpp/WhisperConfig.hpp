#ifndef WHISPERCONFIG_H
#define WHISPERCONFIG_H

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>

#include <whisper.h>
namespace qvac_lib_inference_addon_whisper {

class JSAdapter;

using JSValueVariant =
    std::variant<std::monostate, int, double, std::string, bool>;

/*
 Needs to handle both
 - whisper_full_params
 - whisper_context_params
 and probably more later.
*/

template <typename Params>
using HandlerFunction = std::function<void(Params&, const JSValueVariant&)>;

template <typename Params>
using HandlersMap = std::unordered_map<std::string, HandlerFunction<Params>>;

struct WhisperConfig {
  std::map<std::string, JSValueVariant> miscConfig;
  std::map<std::string, JSValueVariant> whisperMainCfg;
  std::map<std::string, JSValueVariant> vadCfg;
  std::map<std::string, JSValueVariant> whisperContextCfg;

  // Addon-level configuration consumed at backend-init time, not by
  // whisper_context_params / whisper_full_params. Both are optional and default
  // to empty strings so existing callers see no behavior change.

  // Writable directory used to cache OpenCL JIT-compiled kernels on Android.
  // When non-empty, "/opencl-cache" is appended and the resulting path is
  // exported as GGML_OPENCL_CACHE_DIR before the first ggml backend load.
  // Ignored on non-Android platforms.
  std::string openclCacheDir;

  // Directory containing prebuilt ggml backend shared libraries
  // (libggml-cpu*.so, libggml-vulkan.so, libggml-opencl.so, ...). When
  // non-empty it is forwarded to ggml_backend_load_all_from_path; otherwise
  // ggml_backend_load_all() default search is used.
  std::string backendsDir;
};

struct MiscConfig {
  bool captionModeEnabled;
  int seed; // this is an internal c++ calls that is going to be handled
            // seperately.
  // seed is not passed to the main functions but is set ahead of time before
  // the function call. use this struct for future options not passed into
  // whisper.cpp but nevertheless effects model input/output
};

MiscConfig defaultMiscConfig();

// HandlersMap are visible to the js side only.
// uses handlers map form js side.
whisper_full_params toWhisperFullParams(const WhisperConfig& whisperConfig);

// HandlersMap are visible to the js side only, so this function
// uses handlers
whisper_context_params
toWhisperContextParams(const WhisperConfig& whisperConfig);

MiscConfig toMiscConfig(const WhisperConfig& whisperConfig);

std::string convertVariantToString(const JSValueVariant& value);
} // namespace qvac_lib_inference_addon_whisper

#endif // WHISPERCONFIG_H
