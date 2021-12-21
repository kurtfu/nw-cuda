/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "cuda.hpp"

#include <algorithm>
#include <cooperative_groups.h>
#include <thrust/swap.h>

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::cuda;

/*****************************************************************************/
/*  NAMESPACE ALIASES                                                        */
/*****************************************************************************/

namespace cg = cooperative_groups;

/*****************************************************************************/
/*  DEVICE SYMBOLS                                                           */
/*****************************************************************************/

namespace
{
    __constant__ int nw_cuda_match;
    __constant__ int nw_cuda_miss;
    __constant__ int nw_cuda_gap;

    __constant__ std::size_t nw_cuda_n_row;
    __constant__ std::size_t nw_cuda_n_col;
}

/*****************************************************************************/
/*  DEVICE KERNELS                                                           */
/*****************************************************************************/

__device__ static void nw_cuda_fill_cell(std::size_t rw,
                                         std::size_t cl,
                                         int*        curr,
                                         int*        hv,
                                         int*        diag,
                                         char const* ref,
                                         char const* src)
{
    cg::grid_group grid   = cg::this_grid();
    std::size_t    n_vect = std::min(nw_cuda_n_row - rw, cl + 1);

    std::size_t top_row = rw;

    rw += grid.thread_rank();
    cl -= grid.thread_rank();

    if (rw - top_row >= n_vect)
    {
        return;
    }

    if (rw == 0 || cl == 0)
    {
        curr[rw] = (rw + cl) * nw_cuda_gap;
    }
    else
    {
        int eps = (ref[cl - 1] == src[rw - 1]) ? nw_cuda_match : nw_cuda_miss;

        curr[rw] = std::max({diag[rw - 1] + eps,
                             hv[rw - 1] + nw_cuda_gap,
                             hv[rw] + nw_cuda_gap});
    }
}

__device__ static void nw_cuda_fill_subad(std::size_t ad,
                                          int*        curr,
                                          int*        hv,
                                          int*        diag,
                                          char const* ref,
                                          char const* src)
{
    cg::grid_group grid = cg::this_grid();

    std::size_t rw = (ad < nw_cuda_n_col) ? 0 : ad - nw_cuda_n_col + 1;
    std::size_t cl = (ad < nw_cuda_n_col) ? ad : nw_cuda_n_col - 1;

    std::size_t n_vect = std::min(nw_cuda_n_row - rw, cl + 1);

    std::size_t n_iter = (n_vect % grid.size()) ? 1 : 0;
    n_iter += n_vect / grid.size();

    for (std::size_t iter = 0; iter < n_iter; ++iter)
    {
        nw_cuda_fill_cell(rw, cl, curr, hv, diag, ref, src);

        rw += grid.size();
        cl -= grid.size();
    }
}

__global__ static void nw_cuda_fill(std::size_t ad,
                                    int*        curr,
                                    int*        hv,
                                    int*        diag,
                                    char const* ref,
                                    char const* src)
{
    std::size_t rw = (ad < nw_cuda_n_col) ? 0 : ad - nw_cuda_n_col + 1;
    std::size_t cl = (ad < nw_cuda_n_col) ? ad : nw_cuda_n_col - 1;

    std::size_t n_vect = std::min(nw_cuda_n_row - rw, cl + 1);

    std::size_t top_row = rw;

    rw += (blockIdx.x * blockDim.x + threadIdx.x);
    cl -= (blockIdx.x * blockDim.x + threadIdx.x);

    if (rw - top_row >= n_vect)
    {
        return;
    }

    if (rw == 0 || cl == 0)
    {
        curr[rw] = (rw + cl) * nw_cuda_gap;
    }
    else
    {
        int eps = (ref[cl - 1] == src[rw - 1]) ? nw_cuda_match : nw_cuda_miss;

        curr[rw] = std::max({diag[rw - 1] + eps,
                             hv[rw - 1] + nw_cuda_gap,
                             hv[rw] + nw_cuda_gap});
    }
}

__global__ static void nw_cuda_score(int*        curr,
                                     int*        hv,
                                     int*        diag,
                                     char const* ref,
                                     char const* src)
{
    cg::grid_group grid   = cg::this_grid();
    std::size_t    n_diag = nw_cuda_n_row + nw_cuda_n_col - 1;

    for (std::size_t ad = 0; ad < n_diag; ++ad)
    {
        cg::sync(grid);

        thrust::swap(diag, hv);
        thrust::swap(hv, curr);

        nw_cuda_fill_subad(ad, curr, hv, diag, ref, src);
    }
}

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

cuda::cuda(int match, int miss, int gap)
{
    int dev;
    cudaGetDevice(&dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    warp_size            = prop.warpSize;
    multiprocessor_count = prop.multiProcessorCount;

    max_thread_per_block          = prop.maxThreadsPerBlock;
    max_thread_per_multiprocessor = prop.maxThreadsPerMultiProcessor;

    this->match = match;
    this->miss  = miss;
    this->gap   = gap;

    cudaMemcpyToSymbol(nw_cuda_match, &match, sizeof(int));
    cudaMemcpyToSymbol(nw_cuda_miss, &miss, sizeof(int));
    cudaMemcpyToSymbol(nw_cuda_gap, &gap, sizeof(int));
}

int& cuda::operator()(std::vector<int>::size_type rw, std::vector<int>::size_type cl)
{
    std::size_t upper_line = std::min(n_row, n_col);
    std::size_t lower_line = std::max(n_row, n_col) - 1;

    std::size_t ad = rw + cl;

    std::size_t pos;
    std::size_t offset;

    if (ad < upper_line)
    {
        pos    = ad * (ad + 1) / 2;
        offset = rw;
    }
    else if (ad < lower_line)
    {
        std::size_t n_vect = std::min(n_row, n_col);

        pos = upper_line * (upper_line + 1) / 2;
        pos += (ad - upper_line) * n_vect;

        offset = (n_row < n_col) ? rw : n_col - cl - 1;
    }
    else
    {
        std::size_t n_diag = n_row + n_col - 1;

        ad = n_diag - ad;

        pos = (n_row * n_col) - 1;
        pos -= (ad * (ad + 1) / 2);

        offset = (n_row < n_col) ? rw : n_col - cl;
    }

    return matrix[pos + offset];
}

std::size_t cuda::row_count() const
{
    return n_row;
}

std::size_t cuda::col_count() const
{
    return n_col;
}

void cuda::fill(std::string const& ref, std::string const& src)
{
    std::size_t n_row = src.size() + 1;
    std::size_t n_col = ref.size() + 1;

    if (n_row * n_col > this->n_row * this->n_col)
    {
        matrix.reserve(n_row * n_col);
    }
    else
    {
        matrix.resize(n_row * n_col);
        matrix.shrink_to_fit();
    }

    this->n_row = n_row;
    this->n_col = n_col;

    cudaMemcpyToSymbol(nw_cuda_n_row, &n_row, sizeof(std::size_t));
    cudaMemcpyToSymbol(nw_cuda_n_col, &n_col, sizeof(std::size_t));

    std::size_t n_vect = std::min(n_row, n_col);

    int* d_curr;
    int* d_hv;
    int* d_diag;

    cudaMallocHost(&d_curr, n_vect * sizeof(int));
    cudaMallocHost(&d_hv, n_vect * sizeof(int));
    cudaMallocHost(&d_diag, n_vect * sizeof(int));

    char* d_ref;
    char* d_src;

    cudaMalloc(&d_ref, ref.size());
    cudaMemcpy(d_ref, ref.c_str(), ref.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_src, src.size());
    cudaMemcpy(d_src, src.c_str(), src.size(), cudaMemcpyHostToDevice);

    std::size_t n_block = (n_vect % max_thread_per_block) ? 1 : 0;
    n_block += n_vect / max_thread_per_block;

    std::size_t n_thread = (n_vect % n_block) ? 1 : 0;
    n_thread += n_vect / n_block;

    if (n_thread % warp_size)
    {
        n_thread = ((n_thread / warp_size) + 1) * warp_size;
    }

    std::size_t n_diag = n_row + n_col - 1;

    for (std::size_t ad = 0; ad < n_diag; ++ad)
    {
        nw_cuda_fill<<<n_block, n_thread>>>(ad, d_curr, d_hv, d_diag, d_ref, d_src);
        cudaDeviceSynchronize();

        copy_diag(ad, d_curr);

        std::swap(d_diag, d_hv);
        std::swap(d_hv, d_curr);
    }

    cudaFree(d_src);
    cudaFree(d_ref);

    cudaFreeHost(d_diag);
    cudaFreeHost(d_hv);
    cudaFreeHost(d_curr);
}

int cuda::score(std::string const& ref, std::string const& src)
{
    std::size_t n_row = src.size() + 1;
    std::size_t n_col = ref.size() + 1;

    cudaMemcpyToSymbol(nw_cuda_n_row, &n_row, sizeof(std::size_t));
    cudaMemcpyToSymbol(nw_cuda_n_col, &n_col, sizeof(std::size_t));

    std::size_t n_vect = std::min(n_row, n_col);

    int* d_curr;
    int* d_hv;
    int* d_diag;

    cudaMalloc(&d_curr, n_vect * sizeof(int));
    cudaMalloc(&d_hv, n_vect * sizeof(int));
    cudaMalloc(&d_diag, n_vect * sizeof(int));

    char* d_ref;
    char* d_src;

    cudaMalloc(&d_ref, ref.size());
    cudaMemcpy(d_ref, ref.c_str(), ref.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_src, src.size());
    cudaMemcpy(d_src, src.c_str(), src.size(), cudaMemcpyHostToDevice);

    auto dimension = align_dimension(n_vect);

    std::size_t n_block  = dimension.first;
    std::size_t n_thread = dimension.second;

    void* args[] = {&d_curr, &d_hv, &d_diag, &d_ref, &d_src};

    cudaLaunchCooperativeKernel((void*)nw_cuda_score, n_block, n_thread, args);
    cudaDeviceSynchronize();

    int score;
    cudaMemcpy(&score, &d_curr[n_row - 1], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_ref);

    cudaFree(d_diag);
    cudaFree(d_hv);
    cudaFree(d_curr);

    return score;
}

/*****************************************************************************/
/*  PRIVATE METHODS                                                          */
/*****************************************************************************/

std::pair<std::size_t, std::size_t> cuda::align_dimension(std::size_t n_vect)
{
    std::size_t n_block = (n_vect % max_thread_per_block) ? 1 : 0;
    n_block += n_vect / max_thread_per_block;

    if (n_block > multiprocessor_count)
    {
        n_block = multiprocessor_count;
    }

    std::size_t n_thread = (n_vect % n_block) ? 1 : 0;
    n_thread += n_vect / n_block;

    if (n_thread % warp_size)
    {
        n_thread = ((n_thread / warp_size) + 1) * warp_size;
    }

    if (n_thread > max_thread_per_multiprocessor)
    {
        n_thread = max_thread_per_multiprocessor;
    }

    return std::make_pair(n_block, n_thread);
}

void cuda::copy_diag(std::size_t ad, int* diag)
{
    std::size_t rw = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t cl = (ad < n_col) ? ad : n_col - 1;

    std::size_t n_diag = std::min(n_row - rw, cl + 1);
    std::size_t pos    = (n_row <= n_col) ? rw : n_diag - cl - 1;

    cudaMemcpy(&(*this)(rw, cl), &diag[pos], n_diag * sizeof(int), cudaMemcpyDeviceToHost);
}
