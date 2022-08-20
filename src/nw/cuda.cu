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

std::string cuda::align(nw::input const& ref, nw::input const& src)
{
    n_row = src.length();
    n_col = ref.length();

    matrix.resize(n_row * n_col);
    matrix.shrink_to_fit();

    kernel nw(match, miss, gap);
    nw.init(ref, src);

    std::size_t payload = partition_payload();
    nw.allocate_traceback_matrix(payload);

    std::size_t n_diag = n_row + n_col - 1;
    std::size_t pos = 1;

    for (std::size_t ad = 1; ad < n_diag;)
    {
        std::size_t from = ad;
        std::size_t to = find_submatrix_end(ad, payload);

        nw.launch(from, to, true);

        std::size_t size = find_submatrix_size(from, to);
        nw.transfer(&matrix[pos], size);

        pos += size;
        ad = to;
    }

    return traceback(ref, src);
}

int cuda::score(nw::input const& ref, nw::input const& src)
{
    n_row = src.length();
    n_col = ref.length();

    kernel nw(match, miss, gap);
    nw.init(ref, src);

    std::size_t from = 1;
    std::size_t to = n_row + n_col - 1;

    nw.launch(from, to, false);

    return nw.read_similarity_score();
}

/*****************************************************************************/
/*  PRIVATE METHODS                                                          */
/*****************************************************************************/

nw::trace const& cuda::operator()(std::size_t rw, std::size_t cl) const
{
    std::size_t ad = rw + cl;

    std::size_t top = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t pos = prior_element_count(ad) + (rw - top);

    return matrix[pos];
}

std::size_t cuda::partition_payload() const
{
    static constexpr std::size_t threshold = 2'000'000;

    std::size_t max_vect = std::min(n_row, n_col);
    std::size_t capacity = n_row * n_col;

    std::size_t payload = (capacity < threshold) ? capacity : threshold;

    return (payload < max_vect) ? max_vect : payload;
}

std::size_t cuda::prior_element_count(std::size_t ad) const
{
    std::size_t rw = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t cl = (ad < n_col) ? ad : n_col - 1;

    std::size_t n_vect = std::min(n_row - rw, cl + 1);

    std::size_t start = cl;
    std::size_t end = start - n_vect + 1;

    std::size_t size = rw * n_col;

    size += (start * (start + 1) / 2);
    size -= (end * (end - 1) / 2);

    return size;
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
    return prior_element_count(end) - prior_element_count(start);
}

std::string cuda::traceback(nw::input const& ref, nw::input const& src) const
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
