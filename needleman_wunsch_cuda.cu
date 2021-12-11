/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "needleman_wunsch_cuda.hpp"
#include <algorithm>

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
    __global__ void nw_cuda_score(int *curr,
                                  int const *hv,
                                  int const *diag,
                                  char const *src,
                                  char const *ref,
                                  std::size_t ad)
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

NeedlemanWunschCUDA::NeedlemanWunschCUDA(int match, int miss, int gap)
    : NeedlemanWunsch{match, miss, gap}
{
    cudaMemcpyToSymbol(nw_cuda_match, &match, sizeof(int));
    cudaMemcpyToSymbol(nw_cuda_miss, &miss, sizeof(int));
    cudaMemcpyToSymbol(nw_cuda_gap, &gap, sizeof(int));
}

int NeedlemanWunschCUDA::score(std::string ref, std::string src)
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

    std::size_t n_vect = n_row;

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

    std::size_t n_diag = n_row + n_col - 1;

    for (std::size_t ad = 0; ad < n_diag; ++ad)
    {
        nw_cuda_score<<<n_block, n_thread>>>(d_curr, d_hv, d_diag, d_ref, d_src, ad);
        cudaDeviceSynchronize();

        std::swap(d_diag, d_hv);
        std::swap(d_hv, d_curr);
    }

    int score;
    cudaMemcpy(&score, &d_hv[n_row - 1], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_ref);

    cudaFree(d_diag);
    cudaFree(d_hv);
    cudaFree(d_curr);

    return score;
}
