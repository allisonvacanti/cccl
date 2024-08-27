/*
 * Compiles `cub::DeviceMergeSort` with a large custom comparator.
 *
 * This is a particularly impactful test case. RAPIDS actually patches
 * CUB's `cub/block/block_merge_sort.cuh` header to disable loop-unrolling
 * because compile time performance with large functors is so bad:
 *
 * https://github.com/rapidsai/cudf/blob/39de5a/cpp/cmake/thirdparty/patches/thrust_faster_sort_compile_times.diff
 *
 * On my system, applying the patch above reduces the compile time by 25% (21s unrolled, 16s no unrolling).
 *
 * nvcc -DNDEBUG -O3 \
 *   -I/path/to/cccl/cub/cub \
 *   -I/path/to/cccl/thrust/thrust \
 *   -I/path/to/cccl/libcudacxx/include \
 *   cub_merge_sort_case.cu
 */

#include <cub/device/device_merge_sort.cuh>
#include <cub/thread/thread_search.cuh>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/memory.h>
#include <thrust/sequence.h>

#include <iostream>

// Controls the size of the comparison functions:
static constexpr std::size_t NUM_LAYERS = 10;

struct comp_func_t
{
  int** d_layer_heads;
  std::size_t layer_size;

  __forceinline__ __device__ int apply_layer(int layer, int val) const
  {
    std::size_t offset = cub::LowerBound(d_layer_heads[layer], layer_size, val);
    return static_cast<int>(offset);
  }

  __forceinline__ __device__ bool operator()(int a, int b) const
  {
#pragma unroll
    for (int i = 0; i < NUM_LAYERS; ++i)
    {
      a = apply_layer(i, a);
      b = apply_layer(i, b);
    }

    return a < b;
  }
};

int main()
{
  constexpr std::size_t num_items = 1 << 20;

  thrust::device_vector<int> input(num_items);
  thrust::sequence(input.begin(), input.end(), static_cast<int>(num_items - 1), -1);
  thrust::device_vector<int> ref(num_items);
  thrust::sequence(ref.begin(), ref.end());

  std::cout << "Input/ref initialized...\n";

  thrust::device_vector<int*> layer_heads(NUM_LAYERS);
  thrust::device_vector<int> layer_data(num_items);
  thrust::sequence(layer_data.begin(), layer_data.end());
  for (std::size_t i = 0; i < NUM_LAYERS; ++i)
  {
    layer_heads[i] = thrust::raw_pointer_cast(layer_data.data());
  }
  comp_func_t comp_func{thrust::raw_pointer_cast(layer_heads.data()), num_items};

  std::cout << "Layers initialized...\n";

  std::size_t temp_storage_size{};
  cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_size, input.begin(), num_items, comp_func);
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);

  std::cout << "Temp storage allocated...\n";

  cub::DeviceMergeSort::SortKeys(
    thrust::raw_pointer_cast(temp_storage.data()), temp_storage_size, input.begin(), num_items, comp_func);

  std::cout << "Sort complete...\n";

  bool success = thrust::equal(input.begin(), input.end(), ref.begin());

  std::cout << "Validation complete...\n";

  if (!success)
  {
    std::cout << "Error: mismatch between input and reference output\n";
    return 1;
  }

  std::cout << "Success\n";
  return 0;
}
