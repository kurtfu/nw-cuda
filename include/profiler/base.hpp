#ifndef PROFILER_BASE_HPP
#define PROFILER_BASE_HPP

/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/creator.hpp"
#include <fstream>

/*****************************************************************************/
/*  DATA TYPES                                                               */
/*****************************************************************************/

namespace profiler
{
    class base
    {
    public:
        base() = default;

        base(base const& that) = delete;
        base(base&& that) = delete;

        virtual ~base() = default;

        base& operator=(base const& that) = delete;
        base& operator=(base&& that) = delete;

        void assign_scoring_coefficients(int match, int miss, int gap);
        void assign_nw_approach(nw::approach type);

        void attach_input(std::string const& file);
        void attach_output(std::string const& file);

        void profile_samples();

        virtual std::string run(nw::input const& ref, nw::input const& src) = 0;

    protected:
        int match{};
        int miss{};
        int gap{};

        nw::approach nw_approach{};

    private:
        std::ifstream input;
        std::ofstream output;
    };
}

#endif  // PROFILER_BASE_HPP
