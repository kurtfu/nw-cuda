/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/cuda.hpp"
#include "nw/kernel.cuh"

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::cuda;

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

cuda::cuda(int match, int miss, int gap)
{
    this->match = match;
    this->miss = miss;
    this->gap = gap;
}

nw::trace& cuda::operator()(std::size_t rw, std::size_t cl)
{
    std::size_t upper = std::min(n_row, n_col);
    std::size_t lower = std::max(n_row, n_col);

    std::size_t ad = rw + cl;

    std::size_t pos;
    std::size_t offset;

    if (ad < upper)
    {
        pos = ad * (ad + 1) / 2;
        offset = rw;
    }
    else if (ad < lower)
    {
        pos = upper * (upper + 1) / 2;
        pos += (ad - upper) * std::min(n_row, n_col);

        offset = (n_row < n_col) ? rw : n_col - cl - 1;
    }
    else
    {
        std::size_t comp = n_row + n_col - ad - 1;

        pos = n_row * n_col;
        pos -= comp * (comp + 1) / 2;

        offset = n_col - cl - 1;
    }

    return matrix[pos + offset];
}

int cuda::fill(std::string const& ref, std::string const& src)
{
    n_row = src.size() + 1;
    n_col = ref.size() + 1;

    matrix.resize(n_row * n_col);
    matrix.shrink_to_fit();

    kernel nw(match, miss, gap);
    nw.init(ref, src);

    std::size_t payload = partition_payload();
    nw.allocate_traceback_matrix(payload);

    std::size_t n_diag = n_row + n_col - 1;
    int score = 0;

    for (std::size_t ad = 1; ad < n_diag;)
    {
        std::size_t from = ad;
        std::size_t to = find_submatrix_end(ad, payload);

        score = nw.launch(from, to, true);

        std::size_t rw = (ad < n_col) ? 0 : ad - n_col + 1;
        std::size_t cl = (ad < n_col) ? ad : n_col - 1;

        std::size_t size = find_submatrix_size(from, to);
        nw.transfer(ad, &(*this)(rw, cl), size);

        ad = to;
    }

    return score;
}

int cuda::score(std::string const& ref, std::string const& src)
{
    n_row = src.size() + 1;
    n_col = ref.size() + 1;

    kernel nw(match, miss, gap);
    nw.init(ref, src);

    std::size_t from = 1;
    std::size_t to = n_row + n_col - 1;

    return nw.launch(from, to, false);
}

/*****************************************************************************/
/*  PRIVATE METHODS                                                          */
/*****************************************************************************/

std::size_t cuda::partition_payload()
{
    static constexpr std::size_t threshold = 2'000'000;

    std::size_t max_vect = std::min(n_row, n_col);
    std::size_t capacity = n_row * n_col;

    std::size_t payload = (capacity < threshold) ? capacity : threshold;

    return (payload < max_vect) ? max_vect : payload;
}

std::size_t cuda::find_submatrix_end(std::size_t start, std::size_t payload)
{
    std::size_t n_diag = n_row + n_col - 1;

    std::size_t end = start;
    std::size_t total = 0;

    while (end < n_diag && total < payload)
    {
        std::size_t rw = (end < n_col) ? 0 : end - n_col + 1;
        std::size_t cl = (end < n_col) ? end : n_col - 1;

        std::size_t n_vect = std::min(n_row - rw, cl + 1);

        total += n_vect;
        ++end;
    }

    return (total > payload) ? end - 1 : end;
}

std::size_t cuda::find_submatrix_size(std::size_t start, std::size_t end)
{
    std::size_t carry = 0;

    std::size_t rw = (start < n_col) ? 0 : start - n_col + 1;
    std::size_t cl = (start < n_col) ? start : n_col - 1;

    auto from = &(*this)(rw, cl);

    if (end != n_row + n_col - 1)
    {
        rw = (end < n_col) ? 0 : end - n_col + 1;
        cl = (end < n_col) ? end : n_col - 1;
    }
    else
    {
        rw = n_row - 1;
        cl = n_col - 1;

        carry = 1;
    }

    auto to = &(*this)(rw, cl);

    return (std::distance(from, to) + carry) * sizeof(nw::trace);
}
