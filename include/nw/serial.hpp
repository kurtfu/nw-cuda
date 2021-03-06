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
        ~serial() = default;

        trace& operator()(std::size_t rw, std::size_t cl) override;

        int fill(nw::input const& ref, nw::input const& src) override;
        int score(nw::input const& ref, nw::input const& src) override;

    private:
        trace find_trace(int pair, int insert, int remove);
    };
}

#endif  // NW_SERIAL_HPP
