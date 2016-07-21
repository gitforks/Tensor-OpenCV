#pragma once

#include <Sion/Tensor.hpp>

#include "Exception.hpp"

#include <opencv2/core.hpp>

namespace Sion {
namespace Tensor {
namespace OpenCV {

namespace {

template<typename DType, typename SType>
void CopyMat2Tensor3(Tensor<cpu, 3, DType> &ret, const cv::Mat &mat)
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
				ret.dptr_[(i * ret.stride_ + j) * c + k] = *(col + k);
			}
		}
	}
}

} // namespace {}

template<typename Device, int dimension, typename DType = float>
struct Mat2Tensor {
	Tensor<Device, dimension, DType> operator()(const cv::Mat &mat)
	{
		static_assert(0, "Not Implemented");
	}
};

template<typename DType>
struct Mat2Tensor<cpu, 3, DType> {
	Tensor<cpu, 3, DType> operator()(const cv::Mat &mat)
	{

		if (mat.dims != 2)
		{
			throw DimensionMismatchException(2, mat.dims);
		}

		auto c = mat.channels();

		mshadow::Shape<3> Shape;

		Shape[0] = mat.rows;
		Shape[1] = mat.cols;
		Shape[2] = c;
		
		Tensor<cpu, 3, DType> ret = mshadow::NewTensor<cpu, DType, 3>(Shape, 0.f);

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
