/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/kernel.cuh"

#include <limits>
#include <vector>

#include <cooperative_groups.h>

#include <thrust/functional.h>
#include <thrust/swap.h>

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::kernel;

/*****************************************************************************/
/*  NAMESPACE ALIASES                                                        */
/*****************************************************************************/

namespace cg = cooperative_groups;

/*****************************************************************************/
/*  FREE FUNCTIONS                                                           */
/*****************************************************************************/

namespace
{
    __global__ void fill(kernel nw, std::size_t from, std::size_t to, bool tb)
    {
        cg::grid_group grid = cg::this_grid();

        for (std::size_t ad = from; ad < to; ++ad)
        {
            nw.swap_vectors();

            nw.score(ad, tb);
            nw.advance(ad);

            grid.sync();
        }

        std::size_t n_iter = to - from;
        nw.realign_vectors(n_iter);
    }
}

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

__host__ kernel::kernel(int match, int miss, int gap)
    : match{match}
    , miss{miss}
    , gap{gap}
{}

__host__ __device__ kernel::~kernel()
{
    cudaFree(submatrix);

    cudaFree(src);
    cudaFree(ref);

    cudaFree(diag);
    cudaFree(hv);
    cudaFree(curr);
}

__host__ void kernel::init(nw::input const& ref, nw::input const& src)
{
    n_row = src.length();
    n_col = ref.length();

    load(ref, src);
    allocate_vectors();
}

__host__ void kernel::allocate_traceback_matrix(std::size_t payload)
{
    cudaFree(submatrix);
    cudaMalloc(&submatrix, payload * sizeof(trace));
}

__host__ int kernel::launch(std::size_t from, std::size_t to, bool traceback)
{
    void* args[] = {this, &from, &to, &traceback};
    void* kernel = fill;

    std::size_t n_vect = n_row;

    dim3 grid;
    dim3 block;

    std::tie(grid, block) = align_dimension(n_vect);

    cudaLaunchCooperativeKernel(kernel, grid, block, args);
    cudaDeviceSynchronize();

    int score;
    cudaMemcpy(&score, &curr[n_vect], sizeof(int), cudaMemcpyDefault);

    return score;
}

__host__ void kernel::transfer(trace* to, std::size_t size)
{
    cudaMemcpy(to, submatrix, size, cudaMemcpyDefault);
}

__device__ void kernel::score(std::size_t ad, bool traceback)
{
    thrust::minimum<std::size_t> min;
    thrust::maximum<int> max;

    std::size_t rw = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t cl = (ad < n_col) ? ad : n_col - 1;

    std::size_t pos = thread_rank() + rw + 1;
    std::size_t end = min(n_row - rw, cl + 1) + rw + 1;

    std::size_t iter = thread_rank();

    for (; pos < end; pos += grid_size())
    {
        rw += thread_rank();
        cl -= thread_rank();

        int pair = diag[pos - 1] + ((ref[cl] == src[rw]) ? match : miss);
        int insert = hv[pos - 1] + gap;
        int remove = hv[pos] + gap;

        curr[pos] = max(pair, max(insert, remove));

        if (traceback)
        {
            submatrix[iter] = find_trace(pair, insert, remove);
            iter += grid_size();
        }
    }
}

__device__ void kernel::advance(std::size_t ad)
{
    thrust::minimum<std::size_t> min;

    std::size_t rw = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t cl = (ad < n_col) ? ad : n_col - 1;

    submatrix += min(n_row - rw, cl + 1);
}

__device__ void kernel::swap_vectors()
{
    thrust::swap(diag, hv);
    thrust::swap(hv, curr);
}

__device__ void kernel::realign_vectors(std::size_t n_iter)
{
    constexpr std::size_t device_vect_count = 3;

    if ((n_iter % device_vect_count) == 1)
    {
        copy_vector(diag, hv);
        copy_vector(hv, curr);
    }
    else if ((n_iter % device_vect_count) == 2)
    {
        copy_vector(diag, curr);
        copy_vector(curr, hv);
    }
}

/*****************************************************************************/
/*  PRIVATE METHODS                                                          */
/*****************************************************************************/

__host__ std::pair<dim3, dim3> kernel::align_dimension(std::size_t n_vect)
{
    int dev;
    cudaGetDevice(&dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    std::size_t n_block = (n_vect % prop.maxThreadsPerBlock) ? 1 : 0;
    n_block += n_vect / prop.maxThreadsPerBlock;

    if (n_block > prop.multiProcessorCount)
    {
        n_block = prop.multiProcessorCount;
    }

    std::size_t n_thread = (n_vect % n_block) ? 1 : 0;
    n_thread += n_vect / n_block;

    if (n_thread % prop.warpSize)
    {
        n_thread = ((n_thread / prop.warpSize) + 1) * prop.warpSize;
    }

    if (n_thread > prop.maxThreadsPerMultiProcessor)
    {
        n_thread = prop.maxThreadsPerMultiProcessor;
    }

    return std::make_pair(dim3(n_block), dim3(n_thread));
}

__host__ void kernel::load(nw::input const& ref, nw::input const& src)
{
    cudaMalloc(&this->ref, ref.length());
    cudaMemcpy(this->ref, &ref[0], ref.length(), cudaMemcpyDefault);

    cudaMalloc(&this->src, src.length());
    cudaMemcpy(this->src, &src[0], src.length(), cudaMemcpyDefault);
}

__host__ void kernel::allocate_vectors()
{
    std::size_t n_vect = n_row + 1;

    cudaMalloc(&this->curr, n_vect * sizeof(int));
    cudaMalloc(&this->hv, n_vect * sizeof(int));
    cudaMalloc(&this->diag, n_vect * sizeof(int));

    int val = std::numeric_limits<int>::min() - std::min({match, miss, gap});
    std::vector<int> vect(n_vect, val);

    cudaMemcpy(curr, &vect[0], n_vect * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(hv, &vect[0], n_vect * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(diag, &vect[0], n_vect * sizeof(int), cudaMemcpyDefault);

    cudaMemset(&curr[1], 0, sizeof(int));
}

__device__ void kernel::copy_vector(int* dst, int const* src)
{
    std::size_t pos = thread_rank();
    std::size_t end = n_row + 1;

    while (pos < end)
    {
        dst[pos] = src[pos];
        pos += grid_size();
    }
}

__device__ nw::trace kernel::find_trace(int pair, int insert, int remove)
{
    if (pair > insert)
    {
        return (pair > remove) ? nw::trace::pair : nw::trace::remove;
    }
    else
    {
        return (insert > remove) ? nw::trace::insert : nw::trace::remove;
    }
}

__device__ std::size_t kernel::thread_rank() const
{
    std::size_t thread_id = (threadIdx.y * blockDim.x) + threadIdx.x;
    std::size_t block_id = (blockIdx.y * gridDim.x) + blockIdx.x;

    return (block_id * (blockDim.x * blockDim.y)) + thread_id;
}

__device__ std::size_t kernel::grid_size() const
{
    return (blockDim.y * gridDim.y) * (blockDim.x * gridDim.x);
}
