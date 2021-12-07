/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "needleman_wunsch.hpp"

#include <algorithm>
#include <vector>

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

NeedlemanWunsch::NeedlemanWunsch(int match, int miss, int gap)
    : match{match}
    , miss{miss}
    , gap{gap}
{}

int NeedlemanWunsch::score(std::string ref, std::string src)
{
    std::size_t n_row = std::min(ref.size(), src.size()) + 1;
    std::size_t n_col = std::max(ref.size(), src.size()) + 1;

    std::vector<int> prev(n_col);
    std::vector<int> curr(n_col);

    for (std::size_t rw = 0; rw < n_row; ++rw)
    {
        std::swap(prev, curr);

        for (std::size_t cl = 0; cl < n_col; ++cl)
        {
            if (rw == 0 || cl == 0)
            {
                curr[cl] = (rw + cl) * gap;
            }
            else
            {
                int eps = (ref[cl - 1] == src[rw - 1]) ? match : miss;

                curr[cl] = std::max({prev[cl - 1] + eps,
                                     prev[cl] + gap,
                                     curr[cl - 1] + gap});
            }
        }
    }

    return curr.back();
}
