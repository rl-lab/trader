#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

using feature_t = std::pair<std::vector<float>, std::vector<float>>;

class vec_trade_t {
public:
  vec_trade_t(int n_feature, int bs) : n_feature_(n_feature), bs_(bs) {}
  void Load(std::string name,
            py::array_t<float, py::array::c_style | py::array::forcecast> raw,
            py::array_t<float, py::array::c_style | py::array::forcecast> fea) {

    int64_t n = raw.size();
    int64_t m = raw.size();
    ctx[name] = {std::vector<float>(n), std::vector<float>(m)};
    std::memcpy(ctx[name].first.data(), raw.data(), n * sizeof(float));
    std::memcpy(ctx[name].second.data(), fea.data(), m * sizeof(float));
  };

  py::array_t<float> /*obs*/ Reset() {
    std::vector<float> obs;
    return py::array({bs_, n_feature_}, obs.data());
  };

  std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<int>>
  Step(std::vector<int> actions) {
    std::vector<float> obs;
    std::vector<float> reward;
    std::vector<int> done;

    return {py::array({bs_, n_feature_}, obs.data()),
            py::array({bs_}, reward.data()), py::array({bs_}, done.data())};
  }

  int n_feature_;
  int bs_;
  std::map<std::string, feature_t> ctx;
};

namespace py = pybind11;

PYBIND11_MODULE(trade_env, m) {
  py::class_<vec_trade_t>(m, "VecTrade")
      .def(py::init<int, int>())
      .def("Reset", &vec_trade_t::Reset)
      .def("Step", &vec_trade_t::Step)
      .def("Load", &vec_trade_t::Load);
}
