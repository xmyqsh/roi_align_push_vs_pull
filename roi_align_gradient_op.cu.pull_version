#include "roi_align_gradient_op.h"

#include <stdio.h>
#include <cfloat>
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__device__ T get_feature_gradient(
    T y,
    T x,
    const int h,
    const int w,
    const int height,
    const int width) {
  // deal with cases that inverse elements are out of feature map boundary
  // TODO:
  // int(-0.2) = ?
  // if (y < -0.5 || y > (height - 0.5) || x < -0.5 || x > (width - 0.5)) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y < 0) y = 0;
  if (x < 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T weight = 0;
  if (h == y_low) {
    if (w == x_low) {
      weight = (h + 1 - y) * (w + 1 - x);
    } else if (w == x_high) {
      weight = (h + 1 - y) * (x + 1 - w);
    }
  } else if (h == y_high) {
    if (w == x_low) {
      weight = (y + 1 - h) * (w + 1 - x);
    } else if (w == x_high) {
      weight = (y + 1 - h) * (x + 1 - w);
    }
  }
  return weight;
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__global__ void RoIAlignBackwardFeature(
    const int nthreads,
    const T* top_diff,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) is an element in the bottom input
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    // T gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const T* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      if (n != roi_batch_ind) continue;

      // Do not using rounding; this implementation detail is critical
      T roi_start_w = offset_bottom_rois[1] * spatial_scale;
      T roi_start_h = offset_bottom_rois[2] * spatial_scale;
      T roi_end_w = offset_bottom_rois[3] * spatial_scale;
      T roi_end_h = offset_bottom_rois[4] * spatial_scale;
      // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      // Force malformed ROIs to be 1x1
      T roi_width = max(roi_end_w - roi_start_w, (T)1.);
      T roi_height = max(roi_end_h - roi_start_h, (T)1.);
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      int top_offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const T* offset_top_diff = top_diff + top_offset;

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
          ? sampling_ratio
          : ceil(roi_height / pooled_height); // e.g., = 2
      int roi_bin_grid_w =
          (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

      // Calculate the range of pooled indexes corresponding to the bottom input
      // (ph_end - ph_start) * (pw_end - pw_start) belongs to [0,4]
      int ph_start = floor(static_cast<T>(h - roi_start_h - 1) / bin_size_h - 1);
      int ph_end = ceil(static_cast<T>(h - roi_start_h + 1) / bin_size_h);
      int pw_start = floor(static_cast<T>(w - roi_start_w - 1) / bin_size_w - 1);
      int pw_end = ceil(static_cast<T>(w - roi_start_w + 1) / bin_size_w);

      ph_start = min(max(ph_start, 0), pooled_height);
      ph_end = min(max(ph_end, 0), pooled_height);
      pw_start = min(max(pw_start, 0), pooled_width);
      pw_end = min(max(pw_end, 0), pooled_width);

      for (int ph = ph_start; ph < ph_end; ++ph) {
        for (int pw = pw_start; pw < pw_end; ++pw) {
          for (int iy = 0; iy < roi_bin_grid_h; iy++) { // e.g., iy = 0, 1
            const T y = roi_start_h + ph * bin_size_h +
                static_cast<T>(iy + .5f) * bin_size_h /
                    static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              const T x = roi_start_w + pw * bin_size_w +
                  static_cast<T>(ix + .5f) * bin_size_w /
                      static_cast<T>(roi_bin_grid_w);

              T weight = get_feature_gradient(y, x, h, w, height, width) / count;

              //gradient += weight * offset_top_diff[ph * pooled_width + pw];
              gpu_atomic_add(
                  static_cast<T>(weight * offset_top_diff[ph * pooled_width + pw]), bottom_diff + index);

            } // ix
          } // iy
        } // pw
      } // ph
    } // roi_n
    //bottom_diff[index] = gradient;
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward

} // namespace

template <>
bool RoIAlignGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to pool
  auto& R = Input(1); // RoIs
  auto& dY = Input(2); // Gradient of net w.r.t. output of "forward" op
                       // (aka "gradOutput")
  auto* dX = Output(0); // Gradient of net w.r.t. input to "forward" op
                        // (aka "gradInput")

  dX->ResizeLike(X);

  // Must zero-out dX before accumulating gradients
  // (TODO): Kaiming - is this safe?
  math::Set<float, CUDAContext>(
      dX->size(), 0.f, dX->mutable_data<float>(), &context_);

  if (dY.size() > 0) { // Handle possibly empty gradient if there were no rois
    RoIAlignBackwardFeature<float>
        <<<CAFFE_GET_BLOCKS(dX->size()),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            dX->size(),
            dY.data<float>(),
            R.dim32(0),
            spatial_scale_,
            X.dim32(1),
            X.dim32(2),
            X.dim32(3),
            pooled_height_,
            pooled_width_,
            sampling_ratio_,
            dX->mutable_data<float>(),
            R.data<float>());
  }
  return true;
}

REGISTER_CUDA_OPERATOR(
    RoIAlignGradient,
    RoIAlignGradientOp<float, CUDAContext>);
} // namespace caffe2
