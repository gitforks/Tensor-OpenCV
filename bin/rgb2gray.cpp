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
	//v /= 255.0f;
	auto n = Sion::Tensor::OpenCV::Tensor2Mat<cpu, float>()(v);
	cv::imshow("Test", n);

	//auto r = Sion::ImProc::rgb2gray<mshadow::cpu, 3, float>()(v);

	//{
	//	auto o = Sion::ImProc::imresize<mshadow::cpu, 3, float, Sion::ImProc::ImResize::Nearest<float>>(1024, -1)(v);
	//	auto n = Sion::Tensor::OpenCV::Tensor2Mat<mshadow::cpu, 3, float>()(o);
	//	cv::imshow("Nearest", n);
	//}

	//{
	//	auto o = Sion::ImProc::imresize<mshadow::cpu, 3, float, Sion::ImProc::ImResize::Bilinear<float>>(1024, -1)(v);
	//	auto n = Sion::Tensor::OpenCV::Tensor2Mat<mshadow::cpu, 3, float>()(o);
	//	cv::imshow("Bilinear", n);
	//}

	//{
	//	auto o = Sion::ImProc::imresize<mshadow::cpu, 3, float, Sion::ImProc::ImResize::Bicubic<float>>(1024, -1)(v);
	//	auto n = Sion::Tensor::OpenCV::Tensor2Mat<mshadow::cpu, 3, float>()(o);
	//	cv::imshow("Bicubic", n);
	//}

	cv::waitKey();


	return 0;
}
