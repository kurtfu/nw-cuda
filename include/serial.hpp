#ifndef NW_SERIAL_HPP
#define NW_SERIAL_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "aligner.hpp"

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    class serial : public aligner
    {
    public:
        serial(int match, int miss, int gap);
        ~serial() = default;

        std::size_t row_count() const override;
        std::size_t col_count() const override;

        int& operator()(std::size_t rw, std::size_t cl) override;

        void fill(std::string const& ref, std::string const& src) override;
        int  score(std::string const& ref, std::string const& src) override;
    };
}

#endif  // NW_SERIAL_HPP
