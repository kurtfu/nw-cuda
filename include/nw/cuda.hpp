#ifndef NW_CUDA_HPP
#define NW_CUDA_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/aligner.hpp"

#include <cuda_runtime.h>
#include <memory>

/*****************************************************************************/
/*  TYPE ALIASES                                                             */
/*****************************************************************************/

template <typename T>
using nw_cuda_memory = std::unique_ptr<T, void (*)(void*)>;

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    class cuda : public aligner
    {
    public:
        cuda(int match, int miss, int gap);
        ~cuda() = default;

        trace& operator()(std::size_t rw, std::size_t cl) override;

        int fill(std::string const& ref, std::string const& src) override;
        int score(std::string const& ref, std::string const& src) override;

    private:
        std::pair<dim3, dim3> align_dimension(std::size_t n_vect);

        std::size_t find_submatrix_end(std::size_t start, std::size_t payload);
        std::size_t find_submatrix_size(std::size_t start, std::size_t end);
        std::size_t partition_payload();

        template <typename T>
        nw_cuda_memory<T> alloc_device_memory(std::size_t size)
        {
            return alloc_device_memory<T>(size, nullptr);
        }

        template <typename T>
        nw_cuda_memory<T> alloc_device_memory(std::size_t size, T const* val)
        {
            T* d_mem;

            cudaMalloc(&d_mem, size * sizeof(T));
            cudaMemcpy(d_mem, val, size, cudaMemcpyDefault);

            auto deleter = [](void* ptr)
            {
                cudaFree(ptr);
            };

            return std::unique_ptr<T, decltype(deleter)>(d_mem, deleter);
        }

        int warp_size;
        int multiprocessor_count;

        int max_thread_per_block;
        int max_thread_per_multiprocessor;
    };
}

#endif  // NW_CUDA_HPP
