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

__device__ static void nw_cuda_copy_ad(int*        dst,
                                       int const*  src,
                                       std::size_t size)
{
    cg::grid_group grid = cg::this_grid();
    std::size_t    pos  = grid.thread_rank();

    while (pos < size)
    {
        dst[pos] = src[pos];
        pos += grid.size();
    }
}

__device__ static void nw_cuda_fill_cell(std::size_t rw,
                                         std::size_t cl,
                                         int*        curr,
                                         int const*  hv,
                                         int const*  diag,
                                         char const* ref,
                                         char const* src)
{
    if (rw == 0)
    {
        *curr = cl * nw_cuda_gap;
        return;
    }

    if (cl == 0)
    {
        *curr = rw * nw_cuda_gap;
        return;
    }

    int pair   = (ref[cl - 1] == src[rw - 1]) ? nw_cuda_match : nw_cuda_miss;
    int insert = nw_cuda_gap;
    int remove = nw_cuda_gap;

    std::size_t line = (nw_cuda_n_col > nw_cuda_n_row)
                           ? std::max(nw_cuda_n_row, nw_cuda_n_col)
                           : std::min(nw_cuda_n_row, nw_cuda_n_col);

    if (rw + cl < line)
    {
        pair   += *(diag - 1);
        insert += *(hv - 1);
        remove += *hv;
    }
    else if (rw + cl == line)
    {
        pair   += *diag;
        insert += *hv;
        remove += *(hv + 1);
    }
    else
    {
        pair   += *(diag + 1);
        insert += *hv;
        remove += *(hv + 1);
    }

    *curr = std::max({pair, insert, remove});
}

__device__ static void nw_cuda_fill_ad(std::size_t ad,
                                       int*        curr,
                                       int const*  hv,
                                       int const*  diag,
                                       char const* ref,
                                       char const* src)
{
    cg::grid_group grid = cg::this_grid();

    std::size_t rw = (ad < nw_cuda_n_col) ? 0 : ad - nw_cuda_n_col + 1;
    std::size_t cl = (ad < nw_cuda_n_col) ? ad : nw_cuda_n_col - 1;

    std::size_t top_row = rw;
    std::size_t n_vect  = std::min(nw_cuda_n_row - rw, cl + 1);

    rw += grid.thread_rank();
    cl -= grid.thread_rank();

    while (rw - top_row < n_vect)
    {
        std::size_t pos = rw - top_row;

        nw_cuda_fill_cell(rw, cl, &curr[pos], &hv[pos], &diag[pos], ref, src);

        rw += grid.size();
        cl -= grid.size();
    }
}

__global__ static void nw_cuda_fill(std::size_t from,
                                    std::size_t to,
                                    int*        curr,
                                    int*        hv,
                                    int*        diag,
                                    char const* ref,
                                    char const* src)
{
    cg::grid_group grid = cg::this_grid();

    int* d_hv   = hv;
    int* d_diag = diag;

    auto ad_length = [](std::size_t ad)
    {
        std::size_t rw = (ad < nw_cuda_n_col) ? 0 : ad - nw_cuda_n_col + 1;
        std::size_t cl = (ad < nw_cuda_n_col) ? ad : nw_cuda_n_col - 1;

        return std::min(nw_cuda_n_row - rw, cl + 1);
    };

    for (std::size_t ad = from; ad < to; ++ad)
    {
        cg::sync(grid);

        nw_cuda_fill_ad(ad, curr, hv, diag, ref, src);

        diag = hv;
        hv   = curr;

        curr += ad_length(ad);
    }

    nw_cuda_copy_ad(d_diag, diag, ad_length(to - 2));
    nw_cuda_copy_ad(d_hv, hv, ad_length(to - 1));
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

        nw_cuda_fill_ad(ad, curr, hv, diag, ref, src);
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

int& cuda::operator()(std::size_t rw, std::size_t cl)
{
    std::size_t upper = std::min(n_row, n_col);
    std::size_t lower = std::max(n_row, n_col);

    std::size_t ad = rw + cl;

    std::size_t pos;
    std::size_t offset;

    if (ad < upper)
    {
        pos    = ad * (ad + 1) / 2;
        offset = rw;
    }
    else if (ad < lower)
    {
        pos = upper * (upper + 1) / 2;
        pos += (ad - upper) * std::min(n_row, n_col);

        offset = (n_row < n_col) ? rw : n_col - cl - 1;
    }
    else
    {
        std::size_t comp = n_row + n_col - ad - 1;

        pos = n_row * n_col;
        pos -= comp * (comp + 1) / 2;

        offset = n_col - cl - 1;
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

int cuda::fill(std::string const& ref, std::string const& src)
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

    cudaStream_t stream[2];

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    std::size_t n_vect  = std::min(n_row, n_col);
    std::size_t payload = partition_payload();

    auto buf = alloc_pinned(payload);

    std::unique_ptr<int, void (*)(void*)> d_curr[] = {alloc_pageable(payload),
                                                      alloc_pageable(payload)};

    auto d_hv   = alloc_pageable(n_vect);
    auto d_diag = alloc_pageable(n_vect);

    auto d_ref = alloc_sequence(ref);
    auto d_src = alloc_sequence(src);

    std::size_t start = 0;
    std::size_t end   = find_submatrix_end(start, payload);

    int* p_curr = d_curr[0].get();
    int* p_hv   = d_hv.get();
    int* p_diag = d_diag.get();

    char* p_ref = d_ref.get();
    char* p_src = d_src.get();

    void* args[] = {&start, &end, &p_curr, &p_hv, &p_diag, &p_ref, &p_src};
    void* kernel = nw_cuda_fill;

    auto[grid, block] = align_dimension(n_vect);

    cudaLaunchCooperativeKernel(kernel, grid, block, args, 0, stream[0]);

    int prev = 0;
    int next = 0;

    std::size_t n_diag = n_row + n_col - 1;

    for (std::size_t ad = end; ad < n_diag; ad = end)
    {
        std::size_t size = find_submatrix_size(start, end);

        std::size_t rw = (start < n_col) ? 0 : start - n_col + 1;
        std::size_t cl = (start < n_col) ? start : n_col - 1;

        cudaMemcpyAsync(buf.get(), p_curr, size, cudaMemcpyDefault, stream[prev]);

        prev = next;
        next = !next;

        start = ad;
        end   = find_submatrix_end(start, payload);

        p_curr = d_curr[next].get();

        cudaStreamSynchronize(stream[prev]);
        cudaLaunchCooperativeKernel(kernel, grid, block, args, 0, stream[next]);

        std::memcpy(&(*this)(rw, cl), buf.get(), size);
    }

    std::size_t rw = (start < n_col) ? 0 : start - n_col + 1;
    std::size_t cl = (start < n_col) ? start : n_col - 1;

    std::size_t size = find_submatrix_size(start, end);
    cudaMemcpy(&(*this)(rw, cl), p_curr, size, cudaMemcpyDefault);

    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[0]);

    return(*this)(n_row - 1, n_col - 1);
}

int cuda::score(std::string const& ref, std::string const& src)
{
    std::size_t n_row = src.size() + 1;
    std::size_t n_col = ref.size() + 1;

    cudaMemcpyToSymbol(nw_cuda_n_row, &n_row, sizeof(std::size_t));
    cudaMemcpyToSymbol(nw_cuda_n_col, &n_col, sizeof(std::size_t));

    std::size_t n_vect = std::min(n_row, n_col);

    auto d_curr = alloc_pageable(n_vect);
    auto d_hv   = alloc_pageable(n_vect);
    auto d_diag = alloc_pageable(n_vect);

    auto d_ref = alloc_sequence(ref);
    auto d_src = alloc_sequence(src);

    int* p_curr = d_curr.get();
    int* p_hv   = d_hv.get();
    int* p_diag = d_diag.get();

    char* p_ref = d_ref.get();
    char* p_src = d_src.get();

    void* args[] = {&p_curr, &p_hv, &p_diag, &p_ref, &p_src};
    void* kernel = nw_cuda_score;

    auto[grid, block] = align_dimension(n_vect);

    cudaLaunchCooperativeKernel(kernel, grid, block, args);
    cudaDeviceSynchronize();

    int score;
    cudaMemcpy(&score, d_curr.get(), sizeof(int), cudaMemcpyDefault);

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

nw_cuda_memory cuda::alloc_pageable(std::size_t size)
{
    int* d_mem;
    cudaMalloc(&d_mem, size * sizeof(int));

    auto deleter = [](void* ptr)
    {
        cudaFree(ptr);
    };

    return std::unique_ptr<int, decltype(deleter)>(d_mem, deleter);
}

nw_cuda_memory cuda::alloc_pinned(std::size_t size)
{
    int* d_mem;
    cudaMallocHost(&d_mem, size * sizeof(int));

    auto deleter = [](void* ptr)
    {
        cudaFreeHost(ptr);
    };

    return std::unique_ptr<int, decltype(deleter)>(d_mem, deleter);
}

nw_cuda_sequence cuda::alloc_sequence(std::string const& seq)
{
    char* d_seq;

    cudaMalloc(&d_seq, seq.size());
    cudaMemcpy(d_seq, seq.c_str(), seq.size(), cudaMemcpyDefault);

    auto deleter = [](void* ptr)
    {
        cudaFree(ptr);
    };

    return std::unique_ptr<char, decltype(deleter)>(d_seq, deleter);
}

std::size_t cuda::find_submatrix_end(std::size_t start, std::size_t payload)
{
    std::size_t n_diag = n_row + n_col - 1;

    std::size_t end   = start;
    std::size_t total = 0;

    while (end < n_diag && total < payload)
    {
        std::size_t rw = (end < n_col) ? 0 : end - n_col + 1;
        std::size_t cl = (end < n_col) ? end : n_col - 1;

        std::size_t n_vect = std::min(n_row - rw, cl + 1);

        total += n_vect;
        ++end;
    }

    return (total > payload) ? end - 1 : end;
}

std::size_t cuda::find_submatrix_size(std::size_t start, std::size_t end)
{
    std::size_t upper_line = std::min(n_row, n_col);
    std::size_t lower_line = std::max(n_row, n_col);

    std::size_t n_diag = n_row + n_col - 1;

    std::size_t size = n_row * n_col;

    if (start < upper_line)
    {
        size -= (start * (start + 1)) / 2;
    }
    else if (start < lower_line)
    {
        std::size_t n_vect = std::min(n_row, n_col);

        size -= (upper_line * (upper_line + 1) / 2);
        size -= (start - upper_line) * n_vect;
    }
    else
    {
        start = n_diag - start;
        size  = (start * (start + 1)) / 2;
    }

    if (end < upper_line)
    {
        size = (end * (end + 1)) / 2;
        size -= (start * (start + 1)) / 2;
    }
    else if (end < lower_line)
    {
        std::size_t n_vect = std::min(n_row, n_col);

        size -= (lower_line - end) * n_vect;

        end = n_diag - end;
        size -= (end * (end + 1)) / 2;
    }
    else
    {
        end = n_diag - end;
        size -= (end * (end + 1)) / 2;
    }

    return size * sizeof(int);
}

std::size_t cuda::partition_payload()
{
    static constexpr std::size_t threshold = 2'000'000;

    std::size_t max_vect = std::min(n_row, n_col);
    std::size_t capacity = n_row * n_col;

    std::size_t payload = (capacity < threshold) ? capacity : threshold;

    return (payload < max_vect) ? max_vect : payload;
}
