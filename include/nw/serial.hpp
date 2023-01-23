#ifndef NW_SERIAL_HPP
#define NW_SERIAL_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/aligner.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    class serial : public aligner
    {
    public:
        serial(int match, int miss, int gap);

        serial(serial const& that) = delete;
        serial(serial&& that) = delete;

        ~serial() override = default;

        serial& operator=(serial const& that) = delete;
        serial& operator=(serial&& that) = delete;

        std::string align(nw::input const& ref, nw::input const& src) override;
        int score(nw::input const& ref, nw::input const& src) override;

    private:
        nw::trace const& operator()(std::size_t row, std::size_t col) const override;

        int match;
        int miss;
        int gap;

        std::size_t n_row = 0;
        std::size_t n_col = 0;

        std::vector<nw::trace> matrix{};
    };
}

#endif  // NW_SERIAL_HPP
