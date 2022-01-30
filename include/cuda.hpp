#ifndef NW_CUDA_HPP
#define NW_CUDA_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "aligner.hpp"
#include <memory>

/*****************************************************************************/
/*  TYPE ALIASES                                                             */
/*****************************************************************************/

using nw_cuda_sequence = std::unique_ptr<char, void (*)(void*)>;
using nw_cuda_trace    = std::unique_ptr<nw::trace, void (*)(void*)>;
using nw_cuda_vect     = std::unique_ptr<int, void (*)(void*)>;

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

        std::size_t row_count() const override;
        std::size_t col_count() const override;

        trace& operator()(std::size_t rw, std::size_t cl) override;

        int fill(std::string const& ref, std::string const& src) override;
        int score(std::string const& ref, std::string const& src) override;

    private:
        std::pair<std::size_t, std::size_t> align_dimension(std::size_t n_vect);

        nw_cuda_sequence alloc_sequence(std::string const& seq);
        nw_cuda_trace    alloc_trace(std::size_t size);
        nw_cuda_vect     alloc_vect(std::size_t size);

        std::size_t find_submatrix_end(std::size_t start, std::size_t payload);
        std::size_t find_submatrix_size(std::size_t start, std::size_t end);
        std::size_t partition_payload();

        int warp_size;
        int multiprocessor_count;

        int max_thread_per_block;
        int max_thread_per_multiprocessor;
    };
}

#endif  // NW_CUDA_HPP
