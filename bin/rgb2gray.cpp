#include <Sion/Tensor.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>

#include <Sion/Tensor/OpenCV.hpp>
#include <Sion/ImProc.hpp>

#include "tinyfiledialogs.h"

std::string type2str(int type) {
	std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
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
  r += (chans+'0');

  return r;
}

const char *exts[] = {
	"*.jpg",
	"*.png",
};

using BLAS = Sion::Tensor::blas<Sion::Tensor::cpu, float>;

int main(int argc, char **argv)
{
	using namespace Sion::Tensor;

	auto fn = tinyfd_openFileDialog("Select a Image", "test.jpg", sizeof(exts) / sizeof(char *), exts, "", 0);

	if (!fn)
	{
		return 0;
	}

	auto m = cv::imread(fn);

	auto v = Sion::Tensor::OpenCV::Mat2Tensor<cpu, float>()(m);
	BLAS::scal(v, 1.0f / 255.0f);

	{
		auto n = Sion::Tensor::OpenCV::Tensor2Mat<cpu, float>()(v);
		cv::imshow("RGB", n);
	}

	v = Sion::ImProc::rgb2gray<cpu, float>()(v);
	v = Sion::ImProc::imresize<cpu, float, Sion::ImProc::ImResize::Bicubic<float>>(1024, -1)(v);
	
	{
		auto n = Sion::Tensor::OpenCV::Tensor2Mat<cpu, float>()(v);
		cv::imshow("GrayScale", n);
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
		filter.stride = 3;
		
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

		Sion::Tensor::apply<cpu, float, Pow>()(h);
		Sion::Tensor::apply<cpu, float, Pow>()(t);
		Sion::Tensor::apply<cpu, float, Sqrt>()(t);
		BLAS::axpy(1, h, t);

		auto n = Sion::Tensor::OpenCV::Tensor2Mat<cpu, float>()(t);
		cv::imshow("Sobel", n);
	}

	cv::waitKey();


	return 0;
}
