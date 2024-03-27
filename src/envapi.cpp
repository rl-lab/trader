#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <string>

namespace py = pybind11;
static std::random_device rd;
static std::mt19937 gen(rd());

using feature_t = std::pair<std::vector<float>, std::vector<float>>;
/* code, start, cursor, position*/
using status_t = std::tuple<std::string, int, int, float>;

class vec_trade_t {
public:
  vec_trade_t(int n_feature, int bs, int T)
      : n_feature_(n_feature), bs_(bs), T_(T), status_(bs) {}
  void Load(std::string name,
            py::array_t<float, py::array::c_style | py::array::forcecast> raw,
            py::array_t<float, py::array::c_style | py::array::forcecast> fea) {

    int64_t n = raw.size();
    int64_t m = raw.size();
    ctx_[name] = {std::vector<float>(n), std::vector<float>(m)};
    std::memcpy(ctx_[name].first.data(), raw.data(), n * sizeof(float));
    std::memcpy(ctx_[name].second.data(), fea.data(), m * sizeof(float));
  };

  std::vector<float> ResetImpl(int i) {
    std::uniform_int_distribution<> dis(0, ctx_.size() - 1);
    int randomIndex = dis(gen);
    auto it = ctx_.begin();
    std::advance(it, randomIndex);
    int tot = ctx_[it->first].second.size() / n_feature_;
    std::uniform_int_distribution<int> dis_t(0, tot - T_ - 1);
    int start = dis_t(gen);
    status_[i] = {it->first, start, 0, 0};
    std::vector<float> ob(ctx_[it->first].second.begin() + start * n_feature_,
                          ctx_[it->first].second.begin() +
                              (1 + start) * n_feature_);
    return ob;
  }

  py::array_t<float> /*obs*/ Reset() {
    std::vector<float> obs(bs_ * n_feature_);
    for (int i = 0; i < bs_; i++) {
      auto ob = ResetImpl(i);
      std::copy(ob.begin(), ob.end(), obs.data() + i * n_feature_);
    }
    return py::array({bs_, n_feature_}, obs.data());
  };

  std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<int>>
  Step(std::vector<int> actions) {

    std::vector<float> obs(bs_ * n_feature_);
    std::vector<float> reward(bs_);
    std::vector<int> done(bs_);

    for (int i = 0; i < bs_; i++) {
      const auto &[code, start, cursor, position] = status_[i];
      status_[i] = {code, start, cursor + 1, position + actions[i] * 0.05};
    }

    return {py::array({bs_, n_feature_}, obs.data()),
            py::array({bs_}, reward.data()), py::array({bs_}, done.data())};
  }

  int n_feature_;
  int bs_;
  int T_;

  std::unordered_map<std::string, feature_t> ctx_;
  std::vector<status_t> status_;
};

namespace py = pybind11;

PYBIND11_MODULE(trade_env, m) {
  py::class_<vec_trade_t>(m, "VecTrade")
      .def(py::init<int, int, int>())
      .def("Reset", &vec_trade_t::Reset)
      .def("Step", &vec_trade_t::Step)
      .def("Load", &vec_trade_t::Load);
}
