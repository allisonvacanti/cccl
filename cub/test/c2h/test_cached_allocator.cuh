#pragma once

#include <cuda_runtime_api.h>

#include <cuda/__cccl_config>

#include <array>
#include <bit>
#include <cstdint>

namespace c2h
{
namespace detail
{

struct test_cached_allocator
{
  static constexpr std::array<std::size_t, 4> slab_sizes = {
    4 * 1024, // 4 KiB
    64 * 1024, // 64 KiB
    4 * 1024 * 1024, // 4 MiB
    64 * 1024 * 1024, // 64 MiB
  };
  static constexpr std::array<std::size_t, 4> slab_counts = {
    32, // 128 KiB
    32, // 2 MiB
    8, // 32 MiB
    4, // 256 MiB
  };
  static constexpr std::array<std::size_t, 5> slab_offsets = {
    0,
    slab_counts[0] * slab_sizes[0],
    slab_counts[0] * slab_sizes[0] + slab_counts[1] * slab_sizes[1],
    slab_counts[0] * slab_sizes[0] + slab_counts[1] * slab_sizes[1] + slab_counts[2] * slab_sizes[2],
    slab_counts[0] * slab_sizes[0] + slab_counts[1] * slab_sizes[1] + slab_counts[2] * slab_sizes[2]
      + slab_counts[3] * slab_sizes[3], // end
  };
  static constexpr std::size_t total_size = slab_offsets[4];

  void* m_pool{};
  std::array<void*, 4> m_slabs{};
  std::array<std::uint32_t, 4> m_slab_states{};

  void initialize()
  {
    cudaError_t error = cudaMalloc(&m_pool, total_size);
    if (error != cudaSuccess)
    {
      printf("Failed to allocate pool: %s\n", cudaGetErrorString(error));
      return;
    }

    for (std::size_t i = 0; i < 4; ++i)
    {
      m_slabs[i] = static_cast<char*>(m_pool) + slab_offsets[i];
    }
    printf("Initialized pool: %zu bytes (this=%p, pool=%p)\n", total_size, this, m_pool);
  }

  ~test_cached_allocator()
  {
    printf("Deallocating pool: %zu bytes (this=%p, pool=%p)\n", total_size, this, m_pool);
    cudaFree(m_pool);
  }

  cudaError_t allocate(void** ptr, std::size_t bytes)
  {
    if (!m_pool)
    {
      initialize();
    }
    for (std::size_t i = 0; i < 4; ++i)
    {
      if (bytes < slab_sizes[i])
      {
        const std::size_t first_free = static_cast<std::size_t>(std::countr_one(m_slab_states[i]));
        if (first_free < slab_counts[i]) [[likely]]
        {
          m_slab_states[i] |= (1 << first_free);
          *ptr = static_cast<char*>(m_slabs[i]) + first_free * slab_sizes[i];
          // printf("Fits in slab %zu: %zu bytes < %zu bytes (first_free=%zu) ptr=%p slab=%p slab_end=%p\n",
          //        i,
          //        bytes,
          //        slab_sizes[i],
          //        first_free,
          //        *ptr,
          //        m_slabs[i],
          //        static_cast<char*>(m_slabs[i]) + slab_sizes[i]);
          return cudaSuccess;
        }
        else
        {
          printf("Slab %zu is full (first_free=%zu)\n", i, first_free);
          break;
        }
      }
    }

    printf("Spilled to cudaMalloc: %zu bytes\n", bytes);
    return cudaErrorMemoryAllocation;
    // return cudaMalloc(ptr, bytes);
  }

  cudaError_t deallocate(void* ptr)
  {
    if (ptr >= m_pool && ptr < (static_cast<char*>(m_pool) + total_size)) [[likely]]
    {
      const std::size_t ptr_offset = static_cast<char*>(ptr) - static_cast<char*>(m_pool);
      for (std::size_t i = 0; i < 4; ++i)
      {
        if (ptr_offset < slab_offsets[i + 1])
        {
          // printf("Slab %d: Deallocating ptr=%p slab=%p slab_end=%p\n",
          //        (int) i,
          //        ptr,
          //        m_slabs[i],
          //        static_cast<char*>(m_slabs[i]) + slab_sizes[i]);
          const std::size_t slab_offset = ptr_offset - slab_offsets[i];
          const std::size_t slab_index  = slab_offset / slab_sizes[i];
          m_slab_states[i] &= ~(1 << slab_index);
          return cudaSuccess;
        }
      }
    }

    printf("Spilled to cudaFree: ptr=%p\n", ptr);
    return cudaErrorInvalidValue;
    // return cudaFree(ptr);
  }
};

} // namespace detail
} // namespace c2h
