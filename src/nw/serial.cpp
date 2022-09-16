/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/serial.hpp"
#include <algorithm>

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::serial;

/*****************************************************************************/
/*  MEMBER FUNCTIONS                                                         */
/*****************************************************************************/

serial::serial(int match, int miss, int gap)
    : aligner{match, miss, gap}
{}

int serial::score(nw::input const& ref, nw::input const& src)
{
    n_row = src.length();
    n_col = ref.length();

    int val = std::numeric_limits<int>::min() - std::min({match, miss, gap});

    std::vector<int> prev(n_col, val);
    std::vector<int> curr(n_col, val);

    for (std::size_t rw = 0; rw < n_row; ++rw)
    {
        std::swap(prev, curr);

        curr[0] = rw * gap;

        for (std::size_t cl = 1; cl < n_col; ++cl)
        {
            int pair = prev[cl - 1] + ((ref[cl] == src[rw]) ? match : miss);
            int insert = prev[cl] + gap;
            int remove = curr[cl - 1] + gap;

            curr[cl] = std::max({pair, insert, remove});
        }
    }

    return curr[n_col - 1];
}

std::string serial::align(nw::input const& ref, nw::input const& src)
{
    n_row = src.length();
    n_col = ref.length();

    matrix.resize(n_row * n_col);
    matrix.shrink_to_fit();

    int val = std::numeric_limits<int>::min() - std::min({match, miss, gap});

    std::vector<int> prev(n_col, val);
    std::vector<int> curr(n_col, val);

    for (std::size_t rw = 0; rw < n_row; ++rw)
    {
        std::swap(prev, curr);

        curr[0] = rw * gap;
        matrix[rw * n_col] = nw::trace::insert;

        for (std::size_t cl = 1; cl < n_col; ++cl)
        {
            int pair = prev[cl - 1] + ((ref[cl] == src[rw]) ? match : miss);
            int insert = prev[cl] + gap;
            int remove = curr[cl - 1] + gap;

            curr[cl] = std::max({pair, insert, remove});
            matrix[rw * n_col + cl] = find_trace(pair, insert, remove);
        }
    }

    return traceback(ref, src);
}

nw::trace const& serial::operator()(std::size_t rw, std::size_t cl) const
{
    return matrix[rw * n_col + cl];
}
