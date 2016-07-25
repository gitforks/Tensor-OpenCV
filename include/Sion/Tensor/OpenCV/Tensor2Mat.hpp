#pragma once

#include <Sion/Tensor.hpp>

#include "Exception.hpp"

#include <opencv2/core.hpp>

namespace Sion {
namespace Tensor {
namespace OpenCV {

template<typename DType>
void CopyTensor2Mat3(cv::Mat &mat, const Tensor<cpu, DType> &T)
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
				*(col + k) = T.data[(i * T.dims[1] + j) * T.dims[2] + k];
			}
		}
	}
}

template<typename DType>
void CopyTensor2Mat2(cv::Mat &mat, const Tensor<cpu, DType> &T)
{
	int c = mat.channels();

	for (int i = 0; i < mat.rows; ++i)
	{
		auto row = (DType *)mat.ptr(i);

		for (int j = 0; j < mat.cols; ++j)
		{
			auto col = row + c * j;

			*col = T.data[i * T.dims[1] + j];
		}
	}
}

template<typename Engine, typename DType = float>
struct Tensor2Mat {
	void operator()(const Tensor<Engine, DType> &T)
	{
		static_assert(0, "Not implemented");
	}
};

template<typename DType>
struct Tensor2Mat<cpu, DType> {
	cv::Mat operator()(const Tensor<cpu, DType> &T)
	{
		int type = cv::DataType<DType>::type;

		cv::Mat ret;

		if (T.ndims == 2)
		{
			ret = cv::Mat(T.dims[0], T.dims[1], type);

			CopyTensor2Mat2(ret, T);
		}
		else if (T.ndims == 3)
		{
			int type = CV_MAKE_TYPE(cv::DataType<DType>::depth, T.dims[2]);

			ret = cv::Mat(T.dims[0], T.dims[1], type);

			CopyTensor2Mat3(ret, T);
		}
		else
		{
			throw std::runtime_error("Unsupported Size!!");
		}

		return ret;
	}
};

} // namespace OpenCV
} // namespace Tensor
} // namespace Sion
