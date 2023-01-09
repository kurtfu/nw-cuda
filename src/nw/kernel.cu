// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
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
/*  KERNEL FUNCTIONS                                                         */
/*****************************************************************************/

namespace
{
    __global__ void align(kernel nw, std::size_t from, std::size_t to)
    {
        cg::grid_group const grid = cg::this_grid();

        for (std::size_t ad = from; ad < to; ++ad)
        {
            nw.swap_vectors();

            nw.align(ad);
            nw.advance(ad);

            grid.sync();
        }
    }

    __global__ void score(kernel nw, std::size_t from, std::size_t to)
    {
        cg::grid_group const grid = cg::this_grid();

        for (std::size_t ad = from; ad < to; ++ad)
        {
            nw.swap_vectors();

            nw.score(ad);
            nw.advance(ad);

            grid.sync();
        }
    }
}

__device__ void kernel::swap_vectors()
{
    thrust::swap(diag, hv);
    thrust::swap(hv, curr);
}

__device__ void kernel::align(std::size_t ad)
{
    thrust::minimum<std::size_t> min;
    thrust::maximum<int> max;

    std::size_t row = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t col = (ad < n_col) ? ad : n_col - 1;

    std::size_t pos = thread_rank() + row + 1;
    std::size_t end = min(n_row - row, col + 1) + row + 1;

    std::size_t iter = thread_rank();

    for (; pos < end; pos += grid_size())
    {
        row += thread_rank();
        col -= thread_rank();

        int pair = diag[pos - 1] + ((ref[col] == src[row]) ? match : miss);
        int insert = hv[pos - 1] + gap;
        int remove = hv[pos] + gap;

        curr[pos] = max(pair, max(insert, remove));
        submatrix[iter] = find_trace(pair, insert, remove);

        iter += grid_size();
    }
}

__device__ void kernel::score(std::size_t ad)
{
    thrust::minimum<std::size_t> min;
    thrust::maximum<int> max;

    std::size_t row = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t col = (ad < n_col) ? ad : n_col - 1;

    std::size_t pos = thread_rank() + row + 1;
    std::size_t end = min(n_row - row, col + 1) + row + 1;

    for (; pos < end; pos += grid_size())
    {
        row += thread_rank();
        col -= thread_rank();

        int pair = diag[pos - 1] + ((ref[col] == src[row]) ? match : miss);
        int insert = hv[pos - 1] + gap;
        int remove = hv[pos] + gap;

        curr[pos] = max(pair, max(insert, remove));
    }
}

__device__ std::size_t kernel::thread_rank()
{
    std::size_t const thread_id = (threadIdx.y * blockDim.x) + threadIdx.x;
    std::size_t const block_id = (blockIdx.y * gridDim.x) + blockIdx.x;

    return (block_id * (static_cast<std::size_t>(blockDim.x) * blockDim.y)) + thread_id;
}

__device__ std::size_t kernel::grid_size()
{
    auto dimension_size_x = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto dimension_size_y = static_cast<std::size_t>(blockDim.y) * gridDim.y;

    return dimension_size_x * dimension_size_y;
}

__device__ nw::trace kernel::find_trace(int pair, int insert, int remove)
{
    if (pair > insert)
    {
        return (pair > remove) ? nw::trace::pair : nw::trace::remove;
    }

    return (insert > remove) ? nw::trace::insert : nw::trace::remove;
}

__device__ void kernel::advance(std::size_t ad)
{
    thrust::minimum<std::size_t> min;

    std::size_t const row = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t const col = (ad < n_col) ? ad : n_col - 1;

    submatrix += min(n_row - row, col + 1);
}

/*****************************************************************************/
/*  MEMBER FUNCTIONS                                                         */
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

__host__ void kernel::load(nw::input const& ref, nw::input const& src)
{
    cudaMalloc(&this->ref, ref.length());
    cudaMemcpy(this->ref, &ref[0], ref.length(), cudaMemcpyDefault);

    cudaMalloc(&this->src, src.length());
    cudaMemcpy(this->src, &src[0], src.length(), cudaMemcpyDefault);
}

__host__ void kernel::allocate_vectors()
{
    std::size_t const n_vect = n_row + 1;

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

__host__ void kernel::allocate_traceback_matrix(std::size_t payload)
{
    cudaFree(submatrix);
    cudaMalloc(&submatrix, payload * sizeof(trace));
}

__host__ void kernel::calculate_similarity()
{
    std::size_t from = 1;
    std::size_t to = n_row + n_col - 1;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* kernel = reinterpret_cast<void*>(::score);
    auto args = pack_kernel_args<void*>(this, &from, &to);

    launch(kernel, static_cast<void**>(args.data()));

    std::size_t const n_iter = to - from;
    realign_vectors(n_iter);
}

__host__ void kernel::align_sequences(std::size_t from, std::size_t to)
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* kernel = reinterpret_cast<void*>(::align);
    auto args = pack_kernel_args<void*>(this, &from, &to);

    launch(kernel, static_cast<void**>(args.data()));

    std::size_t const n_iter = to - from;
    realign_vectors(n_iter);
}

__host__ void kernel::launch(void* kernel, void** args)
{
    auto dimensions = calculate_kernel_dimensions();

    dim3 const grid = dimensions.first;
    dim3 const block = dimensions.second;

    cudaLaunchCooperativeKernel(kernel, grid, block, args);
}

__host__ void kernel::realign_vectors(std::size_t n_iter)
{
    constexpr std::size_t device_vect_count = 3;

    if ((n_iter % device_vect_count) == 1)
    {
        std::swap(diag, hv);
        std::swap(hv, curr);
    }
    else if ((n_iter % device_vect_count) == 2)
    {
        std::swap(diag, curr);
        std::swap(curr, hv);
    }
}

__host__ std::pair<dim3, dim3> kernel::calculate_kernel_dimensions() const
{
    int dev = 0;
    cudaGetDevice(&dev);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    std::size_t const n_vect = n_row;

    std::size_t n_block = ((n_vect % prop.maxThreadsPerBlock) != 0) ? 1 : 0;
    n_block += n_vect / prop.maxThreadsPerBlock;

    if (n_block > prop.multiProcessorCount)
    {
        n_block = prop.multiProcessorCount;
    }

    std::size_t n_thread = ((n_vect % n_block) != 0) ? 1 : 0;
    n_thread += n_vect / n_block;

    if ((n_thread % prop.warpSize) != 0)
    {
        n_thread = ((n_thread / prop.warpSize) + 1) * prop.warpSize;
    }

    if (n_thread > prop.maxThreadsPerMultiProcessor)
    {
        n_thread = prop.maxThreadsPerMultiProcessor;
    }

    return std::make_pair(dim3(n_block), dim3(n_thread));
}

__host__ void kernel::transfer(trace* to, std::size_t size)
{
    cudaMemcpy(to, submatrix, size, cudaMemcpyDefault);
}

__host__ int kernel::read_similarity_score() const
{
    int score = 0;
    cudaMemcpy(&score, &curr[n_row], sizeof(int), cudaMemcpyDefault);

    return score;
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
