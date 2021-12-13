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
        int score(std::string const& ref, std::string const& src) override;
    };
}

#endif  // NW_CUDA_HPP
