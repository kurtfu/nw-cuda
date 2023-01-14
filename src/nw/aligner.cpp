/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/aligner.hpp"

#include <algorithm>

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::aligner;

/*****************************************************************************/
/*  MEMBER FUNCTIONS                                                         */
/*****************************************************************************/

nw::trace aligner::find_trace(int pair, int insert, int remove)
{
    if (pair > insert)
    {
        return (pair > remove) ? nw::trace::pair : nw::trace::remove;
    }

    return (insert > remove) ? nw::trace::insert : nw::trace::remove;
}

std::string aligner::traceback(nw::input const& ref, nw::input const& src) const
{
    std::string result;

    std::size_t row = src.length() - 1;
    std::size_t col = ref.length() - 1;

    while (row != 0 || col != 0)
    {
        nw::trace const trace = (*this)(row, col);

        if (trace == nw::trace::pair)
        {
            result += (ref[col] == src[row]) ? '*' : '!';

            --row;
            --col;
        }
        else if (trace == nw::trace::insert)
        {
            result += '-';
            --row;
        }
        else if (trace == nw::trace::remove)
        {
            result += '-';
            --col;
        }
    }

    std::reverse(result.begin(), result.end());
    return result;
}
