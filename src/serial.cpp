/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "serial.hpp"

#include <algorithm>
#include <vector>

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::serial;

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

serial::serial(int match, int miss, int gap)
{
    this->match = match;
    this->miss  = miss;
    this->gap   = gap;
}

int serial::score(std::string const& ref, std::string const& src)
{
    std::size_t n_row = std::min(ref.size(), src.size()) + 1;
    std::size_t n_col = std::max(ref.size(), src.size()) + 1;

    std::vector<int> prev(n_col);
    std::vector<int> curr(n_col);

    for (std::size_t cl = 0; cl < n_col; ++cl)
    {
        curr[cl] = cl * gap;
    }

    for (std::size_t rw = 1; rw < n_row; ++rw)
    {
        std::swap(prev, curr);
        curr[0] = rw * gap;

        for (std::size_t cl = 1; cl < n_col; ++cl)
        {
            int eps = (ref[cl - 1] == src[rw - 1]) ? match : miss;

            curr[cl] = std::max({prev[cl - 1] + eps,
                                 prev[cl] + gap,
                                 curr[cl - 1] + gap});
        }
    }

    return curr.back();
}
