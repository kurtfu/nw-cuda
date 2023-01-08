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
        ~serial() override = default;

        std::string align(nw::input const& ref, nw::input const& src) override;
        int score(nw::input const& ref, nw::input const& src) override;

    private:
        nw::trace const& operator()(std::size_t rw, std::size_t cl) const override;
    };
}

#endif  // NW_SERIAL_HPP
