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
/*  MEMBER FUNCTIONS                                                         */
/*****************************************************************************/

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
cuda::cuda(int match, int miss, int gap)
    : aligner{match, miss, gap}
{}

int cuda::score(nw::input const& ref, nw::input const& src)
{
    n_row = src.length();
    n_col = ref.length();

    kernel nw(match, miss, gap);
    nw.init(ref, src);

    nw.calculate_similarity();
    return nw.read_similarity_score();
}

std::string cuda::align(nw::input const& ref, nw::input const& src)
{
    n_row = src.length();
    n_col = ref.length();

    matrix.resize(n_row * n_col);
    matrix.shrink_to_fit();

    kernel nw(match, miss, gap);
    nw.init(ref, src);

    std::size_t const payload = calculate_payload();
    nw.allocate_traceback_matrix(payload);

    std::size_t const n_diag = n_row + n_col - 1;
    std::size_t pos = 1;

    for (std::size_t ad = 1; ad < n_diag;)
    {
        std::size_t const from = ad;
        std::size_t const to = find_submatrix_border_vector(ad);

        nw.align_sequences(from, to);

        std::size_t const size = find_submatrix_size(from, to);
        nw.transfer(&matrix[pos], size);

        pos += size;
        ad = to;
    }

    return traceback(ref, src);
}

std::size_t cuda::calculate_payload() const
{
    static constexpr std::size_t threshold = 2'000'000;

    std::size_t max_vect = std::min(n_row, n_col);
    std::size_t const capacity = n_row * n_col;

    std::size_t const payload = (capacity < threshold) ? capacity : threshold;

    return (payload < max_vect) ? max_vect : payload;
}

std::size_t cuda::find_submatrix_border_vector(std::size_t start) const
{
    std::size_t const n_diag = n_row + n_col - 1;
    std::size_t end = start;

    std::size_t current_payload = 0;
    std::size_t const maximum_payload = calculate_payload();

    while (end < n_diag && current_payload < maximum_payload)
    {
        std::size_t const row = (end < n_col) ? 0 : end - n_col + 1;
        std::size_t const col = (end < n_col) ? end : n_col - 1;

        std::size_t n_vect = std::min(n_row - row, col + 1);

        current_payload += n_vect;
        ++end;
    }

    return (current_payload > maximum_payload) ? end - 1 : end;
}

std::size_t cuda::find_submatrix_size(std::size_t from, std::size_t to) const
{
    return prior_element_count(to) - prior_element_count(from);
}

std::size_t cuda::prior_element_count(std::size_t ad) const
{
    std::size_t const row = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t const col = (ad < n_col) ? ad : n_col - 1;

    std::size_t n_vect = std::min(n_row - row, col + 1);

    std::size_t const start = col;
    std::size_t const end = start - n_vect + 1;

    std::size_t size = row * n_col;

    size += (start * (start + 1) / 2);
    size -= (end * (end - 1) / 2);

    return size;
}

nw::trace const& cuda::operator()(std::size_t row, std::size_t col) const
{
    std::size_t const ad = row + col;

    std::size_t const top = (ad < n_col) ? 0 : ad - n_col + 1;
    std::size_t const pos = prior_element_count(ad) + (row - top);

    return matrix[pos];
}
