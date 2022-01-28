/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "serial.hpp"
#include <algorithm>

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

int& serial::operator()(std::size_t rw, std::size_t cl)
{
    return matrix[rw * n_col + cl];
}

std::size_t serial::row_count() const
{
    return n_row;
}

std::size_t serial::col_count() const
{
    return n_col;
}

int serial::fill(std::string const& ref, std::string const& src)
{
    std::size_t n_row = std::min(ref.size(), src.size()) + 1;
    std::size_t n_col = std::max(ref.size(), src.size()) + 1;

    if (n_row * n_col > this->n_row * this->n_col)
    {
        matrix.reserve(n_row * n_col);
    }
    else
    {
        matrix.resize(n_row * n_col);
        matrix.shrink_to_fit();
    }

    this->n_row = n_row;
    this->n_col = n_col;

    for (std::size_t cl = 0; cl < n_col; ++cl)
    {
        (*this)(0, cl) = cl * gap;
    }

    for (std::size_t rw = 1; rw < n_row; ++rw)
    {
        (*this)(rw, 0) = rw * gap;

        for (std::size_t cl = 1; cl < n_col; ++cl)
        {
            int eps = (ref[cl - 1] == src[rw - 1]) ? match : miss;

            (*this)(rw, cl) = std::max({(*this)(rw - 1, cl - 1) + eps,
                                        (*this)(rw - 1, cl) + gap,
                                        (*this)(rw, cl - 1) + gap});
        }
    }

    return (*this)(n_row - 1, n_col - 1);
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
