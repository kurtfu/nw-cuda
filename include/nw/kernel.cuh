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

        __host__ kernel(kernel const& that) = delete;
        __host__ kernel(kernel&& that) = delete;

        __host__ kernel& operator=(kernel const& that) = delete;
        __host__ kernel& operator=(kernel&& that) = delete;

        __host__ __device__ ~kernel();

        __host__ void init(nw::input const& ref, nw::input const& src);
        __host__ void allocate_traceback_matrix(std::size_t payload);

        __host__ void calculate_similarity();
        __host__ void align_sequences(std::size_t from, std::size_t to);

        __host__ void transfer(trace* to, std::size_t size);
        __host__ int read_similarity_score() const;

        __device__ void align(std::size_t ad);
        __device__ void score(std::size_t ad);

        __device__ void advance(std::size_t ad);
        __device__ void swap_vectors();

    private:
        __host__ std::pair<dim3, dim3> calculate_kernel_dimensions() const;
        __host__ void launch(void* kernel, void* args[]);

        __host__ void load(nw::input const& ref, nw::input const& src);
        __host__ void allocate_vectors();
        __host__ void realign_vectors(std::size_t n_iter);

        __device__ trace find_trace(int pair, int insert, int remove) const;

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
