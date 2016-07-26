#include <Sion/Tensor.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>

#include <Sion/Tensor/OpenCV.hpp>
#include <Sion/ImProc.hpp>

#include "tinyfiledialogs.h"

const char *exts[] = {
	"*.jpg",
	"*.png",
};

using BLAS = Sion::Tensor::blas<Sion::Tensor::cpu, float>;

int main(int argc, char **argv)
{
	namespace ImProc = Sion::ImProc;
	using namespace Sion::Tensor;

	auto fn = tinyfd_openFileDialog("Select a Image", "test.jpg", sizeof(exts) / sizeof(char *), exts, "", 0);

	if (!fn)
	{
		return 0;
	}

	//auto m = cv::imread(fn);
	//auto v = Sion::Tensor::OpenCV::Mat2Tensor<cpu, float>()(m);
	Tensor<cpu, float> v;
	ImProc::imread(fn, v);

	BLAS::scal(v, 1.0f / 255.0f);

	{
		auto n = Sion::Tensor::OpenCV::Tensor2Mat<cpu, float>()(v);
		cv::cvtColor(n, n, CV_RGB2BGR);
		cv::imshow("RGB", n);
	}

	Sion::ImProc::rgb2gray<cpu, float>()(v, v);
	Sion::ImProc::imresize<cpu, float, Sion::ImProc::ImResize::Bicubic<float>>(300, -1)(v, v);
	
	{
		auto n = Sion::Tensor::OpenCV::Tensor2Mat<cpu, float>()(v);
		cv::imshow("GrayScale", n);
	}

	{
		struct FloatToBool {
			bool operator()(float v)
			{
				if (v > 0.8f)
				{
					return true;
				}
				
				return false;
			}
		};

		struct BoolToFloat {
			float operator()(bool v)
			{
				if (v)
				{
					return 1.0f;
				}

				return 0.0f;
			}
		};

		bool _data[81] = {
			0, 0, 0, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 1, 0, 0, 0,
			0, 0, 1, 1, 1, 1, 1, 0, 0,
			0, 1, 1, 1, 1, 1, 1, 1, 0,
			1, 1, 1, 1, 1, 1, 1, 1, 1,
			0, 1, 1, 1, 1, 1, 1, 1, 0,
			0, 0, 1, 1, 1, 1, 1, 0, 0,
			0, 0, 0, 1, 1, 1, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0, 0,
		};

		auto s = NewTensor<cpu, bool>(9, 9);
		s.data = _data;

		auto m = NewTensor<cpu, bool>(v.dims[0], v.dims[1]);
		auto k = NewTensor<cpu, float>(v.dims[0], v.dims[1]);
		apply<cpu, FloatToBool>()(v, m);
		Sion::ImProc::imopen<cpu, bool, bool>()(m, s, m);
		apply<cpu, BoolToFloat>()(m, k);
		auto n = Sion::Tensor::OpenCV::Tensor2Mat<cpu, float>()(k);
		cv::imshow("Boolean Image", n);
	}

	{
		float filter_data_h[9] = {
			-1, -2, -1,
			 0,  0,  0,
			 1,  2,  1,
		};

		float filter_data_t[9] = {
			-1,  0,  1,
			-2,  0,  2,
			-1,  0,  1,
		};

		Tensor<cpu, float> filter;

		filter.ndims = 2;
		filter.dims[0] = 3;
		filter.dims[1] = 3;
		
		filter.data = filter_data_h;
		auto h = Sion::ImProc::imfilter<cpu, float>(filter)(v);
		filter.data = filter_data_t;
		auto t = Sion::ImProc::imfilter<cpu, float>(filter)(v);

		struct Pow {
			float v = 2.0f;

			float operator()(float x)
			{
				return std::pow(x, v);
			}
		} pow;

		struct Sqrt {
			float operator()(float x)
			{
				return std::sqrt(x);
			}
		};

		Sion::Tensor::apply<cpu, Pow>()(h);
		Sion::Tensor::apply<cpu, Pow>()(t);
		Sion::Tensor::apply<cpu, Sqrt>()(t);
		BLAS::axpy(1, h, t);

		auto n = Sion::Tensor::OpenCV::Tensor2Mat<cpu, float>()(t);
		cv::imshow("Sobel", n);
	}

	cv::waitKey();


	return 0;
}
