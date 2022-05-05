#ifndef NW_ALIGNER_HPP
#define NW_ALIGNER_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/input.hpp"
#include <vector>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    enum class trace : unsigned char
    {
        pair,
        insert,
        remove
    };

    class aligner
    {
    public:
        virtual ~aligner() = default;

        virtual trace& operator()(std::size_t rw, std::size_t cl) = 0;

        virtual int fill(nw::input const& ref, nw::input const& src) = 0;
        virtual int score(nw::input const& ref, nw::input const& src) = 0;

    protected:
        int match;
        int miss;
        int gap;

        std::size_t n_row = 0;
        std::size_t n_col = 0;

        std::vector<trace> matrix;
    };
}

#endif  // NW_ALIGNER_HPP
