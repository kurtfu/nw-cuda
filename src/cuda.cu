/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "cuda.hpp"
#include <algorithm>

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::cuda;

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

namespace
{
    __global__ void nw_cuda_fill(std::size_t ad, int *matrix, char const *ref, char const *src)
    {
        std::size_t rw = (ad < nw_cuda_n_col) ? 0 : ad - nw_cuda_n_col + 1;
        std::size_t cl = (ad < nw_cuda_n_col) ? ad : nw_cuda_n_col - 1;

        std::size_t n_vect = std::min(nw_cuda_n_row - rw, cl + 1);

        std::size_t top_row = rw;

        rw += threadIdx.x + (blockIdx.x * blockDim.x);
        cl -= (blockIdx.x * blockDim.x + threadIdx.x);

        if (rw - top_row >= n_vect)
        {
            return;
        }

        std::size_t pos = rw * nw_cuda_n_col + cl;

        if (rw == 0 || cl == 0)
        {
            matrix[pos] = (rw + cl) * nw_cuda_gap;
        }
        else
        {
            int eps = (ref[cl - 1] == src[rw - 1]) ? nw_cuda_match : nw_cuda_miss;

            std::size_t diag = (rw - 1) * nw_cuda_n_col + (cl - 1);
            std::size_t horz = (rw - 1) * nw_cuda_n_col + cl;
            std::size_t vert = rw * nw_cuda_n_col + (cl - 1);

            matrix[pos] = std::max({matrix[diag] + eps,
                                    matrix[horz] + nw_cuda_gap,
                                    matrix[vert] + nw_cuda_gap});
        }
    }

    __global__ void nw_cuda_score(std::size_t ad,
                                  int *curr,
                                  int const *hv,
                                  int const *diag,
                                  char const *ref,
                                  char const *src)
    {
        std::size_t rw = (ad < nw_cuda_n_col) ? 0 : ad - nw_cuda_n_col + 1;
        std::size_t cl = (ad < nw_cuda_n_col) ? ad : nw_cuda_n_col - 1;

        std::size_t n_vect = std::min(nw_cuda_n_row - rw, cl + 1);

        std::size_t top_row = rw;

        rw += threadIdx.x + (blockIdx.x * blockDim.x);
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
}

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

cuda::cuda(int match, int miss, int gap)
{
    this->match = match;
    this->miss  = miss;
    this->gap   = gap;

    cudaMemcpyToSymbol(nw_cuda_match, &match, sizeof(int));
    cudaMemcpyToSymbol(nw_cuda_miss, &miss, sizeof(int));
    cudaMemcpyToSymbol(nw_cuda_gap, &gap, sizeof(int));
}

int& cuda::operator()(std::vector<int>::size_type rw, std::vector<int>::size_type cl)
{
    return matrix[rw * n_col + cl];
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
    std::size_t n_row = std::min(ref.size(), src.size()) + 1;
    std::size_t n_col = std::max(ref.size(), src.size()) + 1;

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

    int* d_matrix;

    cudaMalloc(&d_matrix, n_row * n_col * sizeof(int));

    char* d_ref;
    char* d_src;

    cudaMalloc(&d_ref, ref.size());
    cudaMemcpy(d_ref, ref.c_str(), ref.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_src, src.size());
    cudaMemcpy(d_src, src.c_str(), src.size(), cudaMemcpyHostToDevice);

    auto dimension = align_dimension(n_row);

    std::size_t n_block  = dimension.first;
    std::size_t n_thread = dimension.second;

    std::size_t n_diag = n_row + n_col - 1;

    for (std::size_t ad = 0; ad < n_diag; ++ad)
    {
        nw_cuda_fill<<<n_block, n_thread>>>(ad, d_matrix, d_ref, d_src);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&matrix[0], d_matrix, n_row * n_col * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_ref);

    cudaFree(d_matrix);
}

int cuda::score(std::string const& ref, std::string const& src)
{
    std::size_t n_row = std::min(ref.size(), src.size()) + 1;
    std::size_t n_col = std::max(ref.size(), src.size()) + 1;

    cudaMemcpyToSymbol(nw_cuda_n_row, &n_row, sizeof(std::size_t));
    cudaMemcpyToSymbol(nw_cuda_n_col, &n_col, sizeof(std::size_t));

    int* d_curr;
    int* d_hv;
    int* d_diag;

    cudaMalloc(&d_curr, n_row * sizeof(int));
    cudaMalloc(&d_hv, n_row * sizeof(int));
    cudaMalloc(&d_diag, n_row * sizeof(int));

    char* d_ref;
    char* d_src;

    cudaMalloc(&d_ref, ref.size());
    cudaMemcpy(d_ref, ref.c_str(), ref.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_src, src.size());
    cudaMemcpy(d_src, src.c_str(), src.size(), cudaMemcpyHostToDevice);

    auto dimension = align_dimension(n_row);

    std::size_t n_block  = dimension.first;
    std::size_t n_thread = dimension.second;

    std::size_t n_diag = n_row + n_col - 1;

    for (std::size_t ad = 0; ad < n_diag; ++ad)
    {
        std::swap(d_diag, d_hv);
        std::swap(d_hv, d_curr);

        nw_cuda_score<<<n_block, n_thread>>>(ad, d_curr, d_hv, d_diag, d_ref, d_src);
        cudaDeviceSynchronize();
    }

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
    constexpr std::size_t max_thread_per_block = 1024;
    constexpr std::size_t warp_size = 32;

    std::size_t n_block = n_vect / max_thread_per_block;
    n_block = (n_vect % max_thread_per_block) ? n_block + 1 : n_block;

    std::size_t n_thread = n_vect / n_block;
    n_thread = (n_vect % n_block) ? n_thread + 1 : n_thread;

    if (n_thread % warp_size)
    {
        n_thread = ((n_thread / warp_size) + 1) * warp_size;
    }

    return std::make_pair(n_block, n_thread);
}
