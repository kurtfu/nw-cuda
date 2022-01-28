#ifndef NW_ALIGNER_HPP
#define NW_ALIGNER_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include <string>
#include <vector>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace nw
{
    class aligner
    {
    public:
        virtual ~aligner() = default;

        virtual int& operator()(std::size_t rw, std::size_t cl) = 0;

        virtual std::size_t row_count() const = 0;
        virtual std::size_t col_count() const = 0;

        virtual int fill(std::string const& ref, std::string const& src)  = 0;
        virtual int score(std::string const& ref, std::string const& src) = 0;

    protected:
        int match;
        int miss;
        int gap;

        std::size_t n_row = 0;
        std::size_t n_col = 0;

        std::vector<int> matrix;
    };
}

#endif  // NW_ALIGNER_HPP
