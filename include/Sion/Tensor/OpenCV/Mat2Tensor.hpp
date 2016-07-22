#pragma once

#include <Sion/Tensor.hpp>

#include "Exception.hpp"

#include <opencv2/core.hpp>

namespace Sion {
namespace Tensor {
namespace OpenCV {

namespace {

template<typename DType, typename SType>
void CopyMat2Tensor3(Tensor<cpu, DType> &ret, const cv::Mat &mat)
{
	int c = mat.channels();

	for (int i = 0; i < mat.rows; ++i)
	{
		auto row = (SType *)mat.ptr(i);

		for (int j = 0; j < mat.cols; ++j)
		{
			auto col = row + c * j;

			for (int k = 0; k < c; ++k)
			{
				ret.data[(i * ret.dims[1] + j) * ret.stride + k] = *(col + k);
			}
		}
	}
}

} // namespace {}

template<typename Engine, typename DType = float>
struct Mat2Tensor {
	Tensor<Engine, DType> operator()(const cv::Mat &mat)
	{
		static_assert(0, "Not Implemented");
	}
};

template<typename DType>
struct Mat2Tensor<cpu, DType> {
	Tensor<cpu, DType> operator()(const cv::Mat &mat)
	{
		if (mat.dims != 2)
		{
			throw DimensionMismatchException(2, mat.dims);
		}

		auto c = mat.channels();

		//printf("l: %d %d %d\n", mat.rows, mat.cols, mat.channels());
		auto ret = NewTensor<cpu, DType>(mat.rows, mat.cols, c);
		//printf("v: %d\n", ret.stride_);

		switch (mat.depth())
		{
			case CV_8U:  CopyMat2Tensor3<DType, uint8_t>(ret, mat); break;
			case CV_8S:  CopyMat2Tensor3<DType, int8_t>(ret, mat); break;
			case CV_16U: CopyMat2Tensor3<DType, uint16_t>(ret, mat); break;
			case CV_16S: CopyMat2Tensor3<DType, int16_t>(ret, mat); break;
			case CV_32S: CopyMat2Tensor3<DType, int32_t>(ret, mat); break;
			case CV_32F: CopyMat2Tensor3<DType, float>(ret, mat); break;
			case CV_64F: CopyMat2Tensor3<DType, double>(ret, mat); break;
			default:     throw std::runtime_error("Unsupported Type"); break;
		}

		return ret;
	}
};

} // namespace OpenCV
} // namespace Tensor
} // namespace Sion
