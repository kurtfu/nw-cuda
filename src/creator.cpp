/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "creator.hpp"

#include "cuda.hpp"
#include "serial.hpp"

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

std::unique_ptr<nw::aligner> nw::creator::create(nw::algo type, int match, int mismatch, int gap)
{
    switch (type)
    {
        case nw::algo::serial:
            return std::make_unique<nw::serial>(match, mismatch, gap);
        case nw::algo::cuda:
            return std::make_unique<nw::cuda>(match, mismatch, gap);
        default:
            return nullptr;
    }
}
