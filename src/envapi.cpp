#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class vec_trade_t {
public:
  vec_trade_t(int n_feature) : n_feature_(n_feature) {}
  void Load(std::string name,
            py::array_t<float, py::array::c_style | py::array::forcecast> arr) {

    int64_t n = arr.size();
    ctx[name] = std::vector<float>(n);
    std::memcpy(ctx[name].data(), arr.data(), n * sizeof(float));
  }

  int n_feature_;
  std::map<std::string, std::vector<float>> ctx;
};

namespace py = pybind11;

PYBIND11_MODULE(trade_env, m) {
  py::class_<vec_trade_t>(m, "VecTrade")
      .def(py::init<int>())
      .def("Load", &vec_trade_t::Load);
}
