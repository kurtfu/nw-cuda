/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/aligner.hpp"

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::aligner;

/*****************************************************************************/
/*  MEMBER FUNCTIONS                                                         */
/*****************************************************************************/

aligner::aligner(int match, int miss, int gap)
    : match{match}
    , miss{miss}
    , gap{gap}
{}

nw::trace aligner::find_trace(int pair, int insert, int remove)
{
    if (pair > insert)
    {
        return (pair > remove) ? nw::trace::pair : nw::trace::remove;
    }
    else
    {
        return (insert > remove) ? nw::trace::insert : nw::trace::remove;
    }
}

std::string aligner::traceback(nw::input const& ref, nw::input const& src) const
{
    std::string result;

    std::size_t rw = n_row - 1;
    std::size_t cl = n_col - 1;

    while (rw != 0 || cl != 0)
    {
        nw::trace trace = (*this)(rw, cl);

        if (trace == nw::trace::pair)
        {
            result += (ref[cl] == src[rw]) ? '*' : '!';

            --rw;
            --cl;
        }
        else if (trace == nw::trace::insert)
        {
            result += '-';
            --rw;
        }
        else if (trace == nw::trace::remove)
        {
            result += '-';
            --cl;
        }
    }

    std::reverse(result.begin(), result.end());
    return result;
}
