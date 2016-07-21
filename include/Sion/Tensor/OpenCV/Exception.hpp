#pragma once

#include <exception>
#include <stdexcept>

namespace Sion {
namespace Tensor {
namespace OpenCV {

struct DimensionMismatchException : public std::runtime_error {
private:
	std::string what_fn(int Requested, int Actual);

public:
	DimensionMismatchException(int Requested, int Actual);
};

} // namespace OpenCV
} // namespace Tensor
} // namespace Sion

