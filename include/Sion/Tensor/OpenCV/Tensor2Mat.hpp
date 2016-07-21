#pragma once

#include <Sion/Tensor.hpp>

#include "Exception.hpp"

#include <opencv2/core.hpp>

namespace Sion {
namespace Tensor {
namespace OpenCV {

template<typename DType>
void CopyTensor2Mat3(cv::Mat &mat, const Tensor<cpu, 3, DType> &Tensor)
{
	int c = mat.channels();

	for (int i = 0; i < mat.rows; ++i)
	{
		auto row = (DType *)mat.ptr(i);

		for (int j = 0; j < mat.cols; ++j)
		{
			auto col = row + c * j;

			for (int k = 0; k < c; ++k)
			{
				*(col + k) = Tensor.dptr_[(i * Tensor.size(1)  + j) * Tensor.stride_ + k];
			}
		}
	}
}

template<typename DType>
void CopyTensor2Mat2(cv::Mat &mat, const Tensor<cpu, 2, DType> &Tensor)
{
	int c = mat.channels();

	for (int i = 0; i < mat.rows; ++i)
	{
		auto row = (DType *)mat.ptr(i);

		for (int j = 0; j < mat.cols; ++j)
		{
			auto col = row + c * j;

			*col = Tensor.dptr_[i * Tensor.stride_ + j];
		}
	}
}

template<typename Device, int dimension, typename DType = float>
struct Tensor2Mat {
	void operator()(const Tensor<Device, dimension, DType> &Tensor)
	{
		static_assert(0, "Not Implemented");
	}
};

template<typename DType>
struct Tensor2Mat<cpu, 2, DType> {
	cv::Mat operator()(const Tensor<cpu, 2, DType> &Tensor)
	{
		int type = cv::DataType<DType>::type;

		cv::Mat ret(Tensor.size(0), Tensor.size(1), type);

		CopyTensor2Mat2(ret, Tensor);

		return ret;
	}
};

template<typename DType>
struct Tensor2Mat<cpu, 3, DType> {
	cv::Mat operator()(const Tensor<cpu, 3, DType> &Tensor)
	{
		int type = CV_MAKE_TYPE(cv::DataType<DType>::depth, Tensor.size(2));

		cv::Mat ret(Tensor.size(0), Tensor.size(1), type);

		CopyTensor2Mat3(ret, Tensor);

		return ret;
	}
};


} // namespace OpenCV
} // namespace Tensor
} // namespace Sion
