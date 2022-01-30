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
                                         nw::trace*  sub,
                                         int*        curr,
                                         int const*  hv,
                                         int const*  diag,
                                         char const* ref,
                                         char const* src)
{
    if (rw == 0)
    {
        *curr = cl * nw_cuda_gap;
        *sub  = nw::trace::remove;

        return;
    }

    if (cl == 0)
    {
        *curr = rw * nw_cuda_gap;
        *sub  = nw::trace::insert;

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
        pair += *(diag - 1);
        insert += *(hv - 1);
        remove += *hv;
    }
    else if (rw + cl == line)
    {
        pair += *diag;
        insert += *hv;
        remove += *(hv + 1);
    }
    else
    {
        pair += *(diag + 1);
        insert += *hv;
        remove += *(hv + 1);
    }

    if (insert > pair)
    {
        if (insert > remove)
        {
            *curr = insert;
            *sub  = nw::trace::insert;
        }
        else
        {
            *curr = remove;
            *sub  = nw::trace::remove;
        }
    }
    else
    {
        if (remove > pair)
        {
            *curr = remove;
            *sub  = nw::trace::remove;
        }
        else
        {
            *curr = pair;
            *sub  = nw::trace::pair;
        }
    }
}

__device__ static void nw_cuda_fill_ad(std::size_t ad,
                                       nw::trace*  sub,
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

        nw_cuda_fill_cell(rw, cl, &sub[pos], &curr[pos], &hv[pos], &diag[pos], ref, src);

        rw += grid.size();
        cl -= grid.size();
    }
}

__device__ static void nw_cuda_score_cell(std::size_t rw,
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
        pair += *(diag - 1);
        insert += *(hv - 1);
        remove += *hv;
    }
    else if (rw + cl == line)
    {
        pair += *diag;
        insert += *hv;
        remove += *(hv + 1);
    }
    else
    {
        pair += *(diag + 1);
        insert += *hv;
        remove += *(hv + 1);
    }

    *curr = std::max({pair, insert, remove});
}

__device__ static void nw_cuda_score_ad(std::size_t ad,
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

        nw_cuda_score_cell(rw, cl, &curr[pos], &hv[pos], &diag[pos], ref, src);

        rw += grid.size();
        cl -= grid.size();
    }
}

__global__ static void nw_cuda_fill(std::size_t from,
                                    std::size_t to,
                                    nw::trace*  sub,
                                    int*        curr,
                                    int*        hv,
                                    int*        diag,
                                    char const* ref,
                                    char const* src)
{
    cg::grid_group grid = cg::this_grid();

    for (std::size_t ad = from; ad < to; ++ad)
    {
        cg::sync(grid);

        thrust::swap(diag, hv);
        thrust::swap(hv, curr);

        nw_cuda_fill_ad(ad, sub, curr, hv, diag, ref, src);

        std::size_t rw = (ad < nw_cuda_n_col) ? 0 : ad - nw_cuda_n_col + 1;
        std::size_t cl = (ad < nw_cuda_n_col) ? ad : nw_cuda_n_col - 1;

        sub += std::min(nw_cuda_n_row - rw, cl + 1);
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

        nw_cuda_score_ad(ad, curr, hv, diag, ref, src);
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

nw::trace& cuda::operator()(std::size_t rw, std::size_t cl)
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

    if (n_row * n_col != this->n_row * this->n_col)
    {
        matrix.resize(n_row * n_col);
        matrix.shrink_to_fit();
    }

    this->n_row = n_row;
    this->n_col = n_col;

    cudaMemcpyToSymbol(nw_cuda_n_row, &n_row, sizeof(std::size_t));
    cudaMemcpyToSymbol(nw_cuda_n_col, &n_col, sizeof(std::size_t));

    std::size_t n_vect  = std::min(n_row, n_col);
    std::size_t payload = partition_payload();

    auto d_sub = alloc_trace(payload);

    auto d_curr = alloc_vect(n_vect);
    auto d_hv   = alloc_vect(n_vect);
    auto d_diag = alloc_vect(n_vect);

    auto d_ref = alloc_sequence(ref);
    auto d_src = alloc_sequence(src);

    int* p_curr = d_curr.get();
    int* p_hv   = d_hv.get();
    int* p_diag = d_diag.get();

    auto swap_vects = [&](std::size_t n_iter)
    {
        static constexpr std::size_t vect_count = 3;

        if (n_iter % vect_count == 1)
        {
            std::swap(p_diag, p_hv);
            std::swap(p_hv, p_curr);
        }
        else if (n_iter % vect_count == 2)
        {
            std::swap(p_curr, p_hv);
            std::swap(p_hv, p_diag);
        }
    };

    char* p_ref = d_ref.get();
    char* p_src = d_src.get();

    nw::trace* p_sub = d_sub.get();

    std::size_t from;
    std::size_t to;

    auto [grid, block] = align_dimension(n_vect);

    void* args[] = {&from, &to, &p_sub, &p_curr, &p_hv, &p_diag, &p_ref, &p_src};
    void* kernel = nw_cuda_fill;

    std::size_t n_diag = n_row + n_col - 1;

    for (std::size_t ad = 0; ad < n_diag; ad = to)
    {
        from = ad;
        to   = find_submatrix_end(ad, payload);

        cudaLaunchCooperativeKernel(kernel, grid, block, args);
        swap_vects(to - from);

        std::size_t rw = (ad < n_col) ? 0 : ad - n_col + 1;
        std::size_t cl = (ad < n_col) ? ad : n_col - 1;

        std::size_t size = find_submatrix_size(from, to);
        cudaMemcpy(&(*this)(rw, cl), p_sub, size, cudaMemcpyDefault);
    }

    int score;
    cudaMemcpy(&score, p_curr, sizeof(int), cudaMemcpyDefault);

    return score;
}

int cuda::score(std::string const& ref, std::string const& src)
{
    std::size_t n_row = src.size() + 1;
    std::size_t n_col = ref.size() + 1;

    cudaMemcpyToSymbol(nw_cuda_n_row, &n_row, sizeof(std::size_t));
    cudaMemcpyToSymbol(nw_cuda_n_col, &n_col, sizeof(std::size_t));

    std::size_t n_vect = std::min(n_row, n_col);

    auto d_curr = alloc_vect(n_vect);
    auto d_hv   = alloc_vect(n_vect);
    auto d_diag = alloc_vect(n_vect);

    auto d_ref = alloc_sequence(ref);
    auto d_src = alloc_sequence(src);

    int* p_curr = d_curr.get();
    int* p_hv   = d_hv.get();
    int* p_diag = d_diag.get();

    char* p_ref = d_ref.get();
    char* p_src = d_src.get();

    void* args[] = {&p_curr, &p_hv, &p_diag, &p_ref, &p_src};
    void* kernel = nw_cuda_score;

    auto [grid, block] = align_dimension(n_vect);

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

nw_cuda_vect cuda::alloc_vect(std::size_t size)
{
    int* d_mem;
    cudaMalloc(&d_mem, size * sizeof(int));

    auto deleter = [](void* ptr)
    {
        cudaFree(ptr);
    };

    return std::unique_ptr<int, decltype(deleter)>(d_mem, deleter);
}

nw_cuda_trace cuda::alloc_trace(std::size_t size)
{
    nw::trace* d_mem;
    cudaMalloc(&d_mem, size * sizeof(nw::trace));

    auto deleter = [](void* ptr)
    {
        cudaFree(ptr);
    };

    return std::unique_ptr<nw::trace, decltype(deleter)>(d_mem, deleter);
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
    std::size_t carry = 0;

    std::size_t rw = (start < n_col) ? 0 : start - n_col + 1;
    std::size_t cl = (start < n_col) ? start : n_col - 1;

    auto from = &(*this)(rw, cl);

    if (end != n_row + n_col - 1)
    {
        rw = (end < n_col) ? 0 : end - n_col + 1;
        cl = (end < n_col) ? end : n_col - 1;
    }
    else
    {
        rw = n_row - 1;
        cl = n_col - 1;

        carry = 1;
    }

    auto to = &(*this)(rw, cl);

    return (std::distance(from, to) + carry) * sizeof(nw::trace);
}

std::size_t cuda::partition_payload()
{
    static constexpr std::size_t threshold = 2'000'000;

    std::size_t max_vect = std::min(n_row, n_col);
    std::size_t capacity = n_row * n_col;

    std::size_t payload = (capacity < threshold) ? capacity : threshold;

    return (payload < max_vect) ? max_vect : payload;
}
