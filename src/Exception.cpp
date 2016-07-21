#include <Sion/Tensor/OpenCV/Exception.hpp>

Sion::Tensor::OpenCV::DimensionMismatchException::DimensionMismatchException(int Requested, int Actual)
	: std::runtime_error(what_fn(Requested, Actual))
{
}

std::string Sion::Tensor::OpenCV::DimensionMismatchException::what_fn(int Requested, int Actual)
{
	char buf[1024];

	sprintf(buf, "Requested: %d Actual: %d", Requested, Actual);

	return buf;
}
