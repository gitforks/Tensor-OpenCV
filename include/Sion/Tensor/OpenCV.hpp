#pragma once

#include <Sion/Tensor/OpenCV/Mat2Tensor.hpp>
#include <Sion/Tensor/OpenCV/Tensor2Mat.hpp>

namespace Sion { // namespace Sion
namespace Tensor { // namespace Tensor
namespace OpenCV {

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth)
	{
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

} // namespace OpenCV
} // namespace Tensor
} // namespace Sion

