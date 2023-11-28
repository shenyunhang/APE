#include <cuda_runtime_api.h>

namespace ape {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace ape
