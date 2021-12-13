/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "creator.hpp"

#include "cuda.hpp"
#include "serial.hpp"

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::algo;
using nw::aligner;
using nw::creator;

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

std::unique_ptr<aligner> creator::create(algo type, int match, int mismatch, int gap)
{
    switch (type)
    {
        case algo::serial:
            return std::make_unique<serial>(match, mismatch, gap);
        case algo::cuda:
            return std::make_unique<cuda>(match, mismatch, gap);
        default:
            return nullptr;
    }
}
