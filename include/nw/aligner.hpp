#ifndef NW_ALIGNER_HPP
#define NW_ALIGNER_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/input.hpp"

#include <vector>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    enum class trace : unsigned char
    {
        pair,
        insert,
        remove
    };

    class aligner
    {
    public:
        aligner() = default;

        aligner(aligner const& that) = delete;
        aligner(aligner&& that) = delete;

        virtual ~aligner() = default;

        aligner& operator=(aligner const& that) = delete;
        aligner& operator=(aligner&& that) = delete;

        virtual std::string align(nw::input const& ref, nw::input const& src) = 0;
        virtual int score(nw::input const& ref, nw::input const& src) = 0;

    protected:
        static nw::trace find_trace(int pair, int insert, int remove);
        [[nodiscard]] std::string traceback(nw::input const& ref, nw::input const& src) const;

    private:
        virtual nw::trace const& operator()(std::size_t row, std::size_t col) const = 0;
    };
}

#endif  // NW_ALIGNER_HPP
