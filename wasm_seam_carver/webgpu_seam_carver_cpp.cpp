// The WebGPU seam carver of webgpu_seam_carver.js in C++: the same WGSL, passes and buffers, driven
// through webgpu.h (https://github.com/webgpu-native/webgpu-headers), the C API that mirrors the
// JavaScript WebGPU API, here with Dawn's C++ bindings (webgpu_cpp.h). Emdawnwebgpu, Dawn's
// webgpu.h for Emscripten, turns the calls into JavaScript WebGPU calls in the browser; the same
// code could be built natively against Dawn. Exposed with embind; the page's adapter
// (CppWebGPUSeamCarver) turns the resolve/reject arguments into promises, so it has the same
// interface as the JavaScript engine and gives bit-identical results.
//
// Asynchrony: the browser's WebGPU never blocks, and without Asyncify or JSPI wgpuInstanceWaitAny
// can't wait either. So every asynchronous call uses CallbackMode::AllowSpontaneous: Emdawnwebgpu
// runs the callback when the JavaScript promise behind it settles, and the callback settles the
// page's promise in turn. WebGPU objects belong to the thread that created them: this module is
// single-threaded, like the JavaScript engine.
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using emscripten::val;

// The WGSL of webgpu_seam_carver.js, its only copy, extracted by build.sh as a raw string literal
const char kWgsl[] =
#include "seam_carver_wgsl.inc"
    ;

std::string text(wgpu::StringView view) { return std::string(std::string_view(view)); }

// The device and everything carvers share, created once by init()
struct Gpu {
    wgpu::Instance instance;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::ShaderModule module;
    wgpu::BindGroupLayout layout;
    wgpu::PipelineLayout pipeline_layout;
    struct {  // the full recompute's, BASE = 256
        wgpu::ComputePipeline diamonds, trace_seam, remove_seam, narrow;
    } full;
    std::map<int, wgpu::ComputePipeline> tiles;        // incremental tile width -> pipeline
    wgpu::Buffer distance_table;
    wgpu::Limits limits;
};
Gpu* gpu = nullptr;

wgpu::ComputePipeline pipeline(const char* entry_point, int base) {
    wgpu::ConstantEntry constant{.key = "BASE", .value = double(base)};
    wgpu::ComputePipelineDescriptor descriptor{
        .layout = gpu->pipeline_layout,
        .compute = {.module = gpu->module, .entryPoint = entry_point, .constantCount = 1, .constants = &constant},
    };
    return gpu->device.CreateComputePipeline(&descriptor);
}

const wgpu::ComputePipeline& incremental_pipeline(int tile) {
    if (tile < 2 || tile % 2 || uint32_t(tile) > gpu->limits.maxComputeInvocationsPerWorkgroup) {
        throw std::invalid_argument("the tile width must be even, from 2 to the workgroup size limit");
    }
    auto it = gpu->tiles.find(tile);
    if (it == gpu->tiles.end()) {
        it = gpu->tiles.emplace(tile, pipeline("dirty_diamonds", tile)).first;
    }
    return it->second;
}

void create_shared_objects(wgpu::Device device) {
    gpu->device = device;
    gpu->queue = device.GetQueue();
    device.GetLimits(&gpu->limits);
    wgpu::ShaderSourceWGSL wgsl;
    wgsl.code = kWgsl;
    wgpu::ShaderModuleDescriptor module_descriptor{.nextInChain = &wgsl};
    gpu->module = device.CreateShaderModule(&module_descriptor);

    auto storage = [](uint32_t binding, wgpu::BufferBindingType type) {
        return wgpu::BindGroupLayoutEntry{.binding = binding, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = type}};
    };
    const wgpu::BindGroupLayoutEntry entries[] = {
        {.binding = 0, .visibility = wgpu::ShaderStage::Compute,
         .buffer = {.type = wgpu::BufferBindingType::Uniform, .hasDynamicOffset = true}},
        storage(1, wgpu::BufferBindingType::Storage),
        storage(2, wgpu::BufferBindingType::Storage),
        storage(3, wgpu::BufferBindingType::Storage),
        storage(4, wgpu::BufferBindingType::Storage),
        storage(5, wgpu::BufferBindingType::ReadOnlyStorage),
        storage(6, wgpu::BufferBindingType::Storage),
        storage(7, wgpu::BufferBindingType::Storage),
    };
    wgpu::BindGroupLayoutDescriptor layout_descriptor{.entryCount = std::size(entries), .entries = entries};
    gpu->layout = device.CreateBindGroupLayout(&layout_descriptor);
    wgpu::PipelineLayoutDescriptor pipeline_layout_descriptor{.bindGroupLayoutCount = 1, .bindGroupLayouts = &gpu->layout};
    gpu->pipeline_layout = device.CreatePipelineLayout(&pipeline_layout_descriptor);
    gpu->full = {pipeline("diamonds", 256), pipeline("trace_seam", 256), pipeline("remove_seam", 256), pipeline("narrow", 256)};

    // Distances between two pixels: the f32 square root of every possible integer dr² + dg² + db²,
    // rounded from the double square root like JavaScript's Float32Array does
    std::vector<float> table(3 * 255 * 255 + 1);
    for (size_t i = 0; i < table.size(); ++i) table[i] = float(std::sqrt(double(i)));
    wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                      .size = table.size() * sizeof(float)};
    gpu->distance_table = device.CreateBuffer(&descriptor);
    gpu->queue.WriteBuffer(gpu->distance_table, 0, table.data(), descriptor.size);
}

// Requests an adapter and a device, then calls resolve() or reject(message). After a failure, the
// next call tries again.
void init(val resolve, val reject) {
    if (gpu && gpu->device) {
        resolve();
        return;
    }
    auto fail = [reject](std::string message) {
        delete gpu;
        gpu = nullptr;
        reject(val(message));
    };
    delete gpu;
    gpu = new Gpu;
    gpu->instance = wgpu::CreateInstance();
    wgpu::RequestAdapterOptions options{.powerPreference = wgpu::PowerPreference::HighPerformance};
    gpu->instance.RequestAdapter(
        &options, wgpu::CallbackMode::AllowSpontaneous,
        [resolve, fail](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                fail("no WebGPU adapter available: " + text(message));
                return;
            }
            // The largest buffers hold one 32-bit value per pixel: ask for what the adapter can do
            wgpu::Limits supported, required;
            adapter.GetLimits(&supported);
            required.maxStorageBufferBindingSize = supported.maxStorageBufferBindingSize;
            required.maxBufferSize = supported.maxBufferSize;
            wgpu::DeviceDescriptor descriptor;
            descriptor.requiredLimits = &required;
            descriptor.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
                std::printf("WebGPU error: %s\n", text(message).c_str());
            });
            descriptor.SetDeviceLostCallback(
                wgpu::CallbackMode::AllowSpontaneous,
                [](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView message) {
                    std::printf("WebGPU device lost: %s\n", text(message).c_str());
                });
            adapter.RequestDevice(
                &descriptor, wgpu::CallbackMode::AllowSpontaneous,
                [resolve, fail](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                    if (status != wgpu::RequestDeviceStatus::Success) {
                        fail("no WebGPU device: " + text(message));
                        return;
                    }
                    create_shared_objects(device);
                    resolve();
                });
        });
}

class SeamCarver {
public:
    // rgb is a row-major Uint8Array of width * height * 3 bytes
    SeamCarver(const val& rgb_array, int width, int height, int tile)
        : width_(width), height_(height), stride_(width) {
        if (!gpu || !gpu->device) throw std::logic_error("init() must succeed first");
        const std::vector<uint8_t> rgb = emscripten::convertJSArrayToNumberVector<uint8_t>(rgb_array);
        if (width <= 0 || height <= 0 || rgb.size() != size_t(width) * height * 3) {
            throw std::invalid_argument("rgb must hold width * height * 3 bytes");
        }
        const uint64_t cell_bytes = uint64_t(width) * height * 4;
        if (cell_bytes > gpu->limits.maxStorageBufferBindingSize || uint32_t(height) > gpu->limits.maxComputeWorkgroupsPerDimension) {
            throw std::range_error("too many pixels for this GPU");
        }
        tile_pipeline_ = &incremental_pipeline(tile);

        std::vector<uint32_t> packed(size_t(width) * height);
        for (size_t i = 0; i < packed.size(); ++i) {
            packed[i] = rgb[3 * i] | (rgb[3 * i + 1] << 8) | (rgb[3 * i + 2] << 16);
        }
        auto buffer = [](uint64_t size, wgpu::BufferUsage usage) {
            wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::Storage | usage, .size = size};
            return gpu->device.CreateBuffer(&descriptor);
        };
        pixels_ = buffer(cell_bytes, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        gpu->queue.WriteBuffer(pixels_, 0, packed.data(), cell_bytes);
        cost_ = buffer(cell_bytes, wgpu::BufferUsage::None);
        prev_ = buffer(cell_bytes, wgpu::BufferUsage::None);
        changed_ = buffer(cell_bytes, wgpu::BufferUsage::None);
        seam_ = buffer(uint64_t(height) * 4, wgpu::BufferUsage::None);
        state_ = buffer(8, wgpu::BufferUsage::CopyDst);
        const uint32_t state[] = {uint32_t(width), 0};
        gpu->queue.WriteBuffer(state_, 0, state, sizeof(state));

        align_ = gpu->limits.minUniformBufferOffsetAlignment;
        full_ = tiling(128);
        incremental_ = tiling(tile / 2);
    }

    int width() const { return width_; }
    int height() const { return height_; }

    // Removes pixels_to_shrink seams in one submission; settles once the GPU is done
    void shrink(int pixels_to_shrink, bool incremental, val resolve, val reject) {
        if (pixels_to_shrink < 0 || pixels_to_shrink >= width_) {
            reject(val("pixels_to_shrink must be in [0, width)"));
            return;
        }
        wgpu::CommandEncoder encoder = gpu->device.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        auto run = [&](const wgpu::ComputePipeline& pipeline, const Tiling& tiling, uint32_t groups, uint32_t wave = 0) {
            if (groups > 0) {
                const uint32_t offset = wave * align_;
                pass.SetPipeline(pipeline);
                pass.SetBindGroup(0, tiling.bind_group, 1, &offset);
                pass.DispatchWorkgroups(groups);
            }
        };
        for (int n = 0; n < pixels_to_shrink; ++n) {
            const bool update = incremental && has_removed_;
            const Tiling& tiling = update ? incremental_ : full_;
            for (uint32_t t = 0; t < tiling.waves; ++t) {
                run(update ? *tile_pipeline_ : gpu->full.diamonds, tiling, tiling.diamonds, t);
            }
            run(gpu->full.trace_seam, full_, 1);
            run(gpu->full.remove_seam, full_, uint32_t(height_));
            run(gpu->full.narrow, full_, 1);
            has_removed_ = incremental;
        }
        pass.End();
        wgpu::CommandBuffer commands = encoder.Finish();
        gpu->queue.Submit(1, &commands);
        width_ -= pixels_to_shrink;
        gpu->queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowSpontaneous,
            [resolve, reject](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                status == wgpu::QueueWorkDoneStatus::Success ? resolve() : reject(val(text(message)));
            });
    }

    // Settles with the shrunk picture as a Uint8Array of row-major RGB bytes
    void shrunk_rgb(val resolve, val reject) {
        const uint64_t size = pixels_.GetSize();
        wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, .size = size};
        wgpu::Buffer readback = gpu->device.CreateBuffer(&descriptor);
        wgpu::CommandEncoder encoder = gpu->device.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(pixels_, 0, readback, 0, size);
        wgpu::CommandBuffer commands = encoder.Finish();
        gpu->queue.Submit(1, &commands);
        const int width = width_, height = height_, stride = stride_;
        readback.MapAsync(
            wgpu::MapMode::Read, 0, size, wgpu::CallbackMode::AllowSpontaneous,
            [readback, width, height, stride, resolve, reject](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                if (status != wgpu::MapAsyncStatus::Success) {
                    reject(val(text(message)));
                    return;
                }
                // In WebAssembly this copies the mapped range into the module's memory
                const auto* packed = static_cast<const uint32_t*>(readback.GetConstMappedRange());
                std::vector<uint8_t> rgb;
                rgb.reserve(size_t(width) * height * 3);
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const uint32_t p = packed[size_t(y) * stride + x];
                        rgb.insert(rgb.end(), {uint8_t(p), uint8_t(p >> 8), uint8_t(p >> 16)});
                    }
                }
                readback.Unmap();
                readback.Destroy();
                resolve(val::global("Uint8Array").new_(emscripten::typed_memory_view(rgb.size(), rgb.data())));
            });
    }

    ~SeamCarver() {
        for (wgpu::Buffer* b : {&pixels_, &cost_, &prev_, &changed_, &seam_, &state_, &full_.uniforms, &incremental_.uniforms}) {
            if (*b) b->Destroy();
        }
    }

private:
    struct Tiling {
        uint32_t waves = 0, diamonds = 0;
        wgpu::Buffer uniforms;
        wgpu::BindGroup bind_group;
    };

    // Parameters per wave of diamonds 2 * half wide, at the offsets dynamic uniform bindings
    // require; entry 0 also serves the passes outside the waves, which only read stride and height
    Tiling tiling(int half) {
        Tiling t;
        t.waves = uint32_t((height_ - 1 + half - 1) / half + 1);
        std::vector<uint32_t> params(t.waves * align_ / 4);
        for (uint32_t w = 0; w < t.waves; ++w) {
            const uint32_t entry[] = {uint32_t(stride_), uint32_t(height_), w, 0};
            std::copy(std::begin(entry), std::end(entry), params.begin() + w * align_ / 4);
        }
        wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                          .size = params.size() * 4};
        t.uniforms = gpu->device.CreateBuffer(&descriptor);
        gpu->queue.WriteBuffer(t.uniforms, 0, params.data(), descriptor.size);
        const wgpu::BindGroupEntry entries_[] = {
            {.binding = 0, .buffer = t.uniforms, .size = 16},
            {.binding = 1, .buffer = state_},
            {.binding = 2, .buffer = pixels_},
            {.binding = 3, .buffer = cost_},
            {.binding = 4, .buffer = prev_},
            {.binding = 5, .buffer = gpu->distance_table},
            {.binding = 6, .buffer = seam_},
            {.binding = 7, .buffer = changed_},
        };
        wgpu::BindGroupDescriptor bind_group_descriptor{.layout = gpu->layout, .entryCount = std::size(entries_), .entries = entries_};
        t.bind_group = gpu->device.CreateBindGroup(&bind_group_descriptor);
        // Dispatch sizes cover the original width; workgroups beyond the current width do nothing
        t.diamonds = uint32_t((stride_ + half - 1) / (2 * half) + 1);
        return t;
    }

    int width_, height_, stride_;
    uint32_t align_ = 256;
    bool has_removed_ = false;
    const wgpu::ComputePipeline* tile_pipeline_ = nullptr;
    wgpu::Buffer pixels_, cost_, prev_, changed_, seam_, state_;
    Tiling full_, incremental_;
};

}  // namespace

EMSCRIPTEN_BINDINGS(webgpu_seam_carver_cpp) {
    emscripten::function("init", &init);
    emscripten::class_<SeamCarver>("SeamCarver")
        .constructor<const val&, int, int, int>()
        .function("shrink", &SeamCarver::shrink)
        .function("shrunk_rgb", &SeamCarver::shrunk_rgb)
        .property("width", &SeamCarver::width)
        .property("height", &SeamCarver::height);
}
