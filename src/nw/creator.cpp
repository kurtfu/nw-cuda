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
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

creator::creator(approach type)
    : type{type}
{}

std::unique_ptr<nw::aligner> creator::create(int match, int miss, int gap)
{
    switch (type)
    {
        case approach::serial:
            return std::make_unique<nw::serial>(match, miss, gap);
        case approach::cuda:
            return std::make_unique<nw::cuda>(match, miss, gap);
        default:
            return nullptr;
    }
}
