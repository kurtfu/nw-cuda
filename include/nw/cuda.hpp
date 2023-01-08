#ifndef NW_CUDA_HPP
#define NW_CUDA_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/aligner.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/********************************************delete*********************************/

namespace nw
{
    class cuda : public aligner
    {
    public:
        cuda(int match, int miss, int gap);

        cuda(cuda const& that) = delete;
        cuda(cuda&& that) = delete;

        ~cuda() override = default;

        cuda& operator=(cuda const& that) = delete;
        cuda& operator=(cuda&& that) = delete;

        std::string align(nw::input const& ref, nw::input const& src) override;
        int score(nw::input const& ref, nw::input const& src) override;

    private:
        nw::trace const& operator()(std::size_t row, std::size_t col) const override;

        [[nodiscard]] std::size_t calculate_payload() const;
        [[nodiscard]] std::size_t prior_element_count(std::size_t ad) const;

        [[nodiscard]] std::size_t find_submatrix_border_vector(std::size_t start) const;
        [[nodiscard]] std::size_t find_submatrix_size(std::size_t from, std::size_t to) const;
    };
}

#endif  // NW_CUDA_HPP
