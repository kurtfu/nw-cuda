#ifndef NW_CUDA_HPP
#define NW_CUDA_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/aligner.hpp"

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

        trace& operator()(std::size_t rw, std::size_t cl) override;

        int fill(nw::input const& ref, nw::input const& src) override;
        int score(nw::input const& ref, nw::input const& src) override;

    private:
        std::size_t partition_payload();
        std::size_t prior_element_count(std::size_t ad);

        std::size_t find_submatrix_end(std::size_t start, std::size_t payload);
        std::size_t find_submatrix_size(std::size_t start, std::size_t end);
    };
}

#endif  // NW_CUDA_HPP
