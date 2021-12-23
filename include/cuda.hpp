#ifndef NW_CUDA_HPP
#define NW_CUDA_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "aligner.hpp"

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

        virtual std::size_t row_count() const override;
        virtual std::size_t col_count() const override;

        int& operator()(std::vector<int>::size_type rw, std::vector<int>::size_type cl) override;

        void fill(std::string const& ref, std::string const& src) override;
        int  score(std::string const& ref, std::string const& src) override;

    private:
        std::pair<std::size_t, std::size_t> align_dimension(std::size_t n_vect);

        void copy_submatrix(int* matrix, std::size_t size, std::size_t start);

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
