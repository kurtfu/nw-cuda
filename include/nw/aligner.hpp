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
        aligner(int match, int miss, int gap);
        virtual ~aligner() = default;

        virtual std::string align(nw::input const& ref, nw::input const& src) = 0;
        virtual int score(nw::input const& ref, nw::input const& src) = 0;

    protected:
        nw::trace find_trace(int pair, int insert, int remove);
        std::string traceback(nw::input const& ref, nw::input const& src) const;

        int match;
        int miss;
        int gap;

        std::size_t n_row = 0;
        std::size_t n_col = 0;

        std::vector<nw::trace> matrix;

    private:
        virtual nw::trace const& operator()(std::size_t rw, std::size_t cl) const = 0;
    };
}

#endif  // NW_ALIGNER_HPP
