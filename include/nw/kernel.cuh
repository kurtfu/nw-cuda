#ifndef NW_KERNEL_CUH
#define NW_KERNEL_CUH

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/aligner.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    class kernel
    {
    public:
        __host__ kernel(int match, int miss, int gap);
        __host__ __device__ ~kernel();

        __host__ void init(nw::input const& ref, nw::input const& src);
        __host__ void allocate_traceback_matrix(std::size_t payload);

        __host__ int launch(std::size_t from, std::size_t to, bool traceback);
        __host__ void transfer(trace* to, std::size_t size);

        __device__ void score(std::size_t ad, bool traceback);
        __device__ void advance(std::size_t ad);
        __device__ void swap_vectors();

    private:
        __host__ std::pair<dim3, dim3> align_dimension(std::size_t n_vect);

        __host__ void load(nw::input const& ref, nw::input const& src);
        __host__ void allocate_vectors();
        __host__ void realign_vectors(std::size_t n_iter);

        __device__ trace find_trace(int pair, int insert, int remove);

        __device__ std::size_t thread_rank() const;
        __device__ std::size_t grid_size() const;

        int match;
        int miss;
        int gap;

        std::size_t n_row;
        std::size_t n_col;

        int* curr;
        int* hv;
        int* diag;

        char* ref;
        char* src;

        trace* submatrix;
    };
}

#endif  // NW_KERNEL_CUH
