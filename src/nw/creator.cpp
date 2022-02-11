/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/creator.hpp"

#include "nw/cuda.hpp"
#include "nw/serial.hpp"

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::creator;

/*****************************************************************************/
/*  MODULE FUNCTIONS                                                         */
/*****************************************************************************/

namespace
{
    std::unique_ptr<nw::aligner> create_serial(int match, int miss, int gap)
    {
        return std::make_unique<nw::serial>(match, miss, gap);
    }

    std::unique_ptr<nw::aligner> create_cuda(int match, int miss, int gap)
    {
        return std::make_unique<nw::cuda>(match, miss, gap);
    }
}

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

creator::creator(approach type)
{
    switch (type)
    {
        case approach::serial:
            create = create_serial;
            break;

        case approach::cuda:
            create = create_cuda;
            break;

        default:
            create = nullptr;
            break;
    }
}
