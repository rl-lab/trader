#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <string>

namespace py = pybind11;
static std::random_device rd;
static std::mt19937 gen(rd());

using feature_t = std::pair<std::vector<float>, std::vector<float>>;
/* code, start, cursor, position, costprice */
using status_t = std::tuple<std::string, int, int, float, float>;

class vec_trade_t {
public:
  vec_trade_t(int n_feature, int bs, int T, float amt)
      : n_feature_(n_feature), bs_(bs), T_(T), amt_(amt), status_(bs) {}
  void Load(std::string name,
            py::array_t<float, py::array::c_style | py::array::forcecast> raw,
            py::array_t<float, py::array::c_style | py::array::forcecast> fea) {

    int64_t n = raw.size();
    int64_t m = raw.size();
    ctx_[name] = {std::vector<float>(n), std::vector<float>(m)};
    std::memcpy(ctx_[name].first.data(), raw.data(), n * sizeof(float));
    std::memcpy(ctx_[name].second.data(), fea.data(), m * sizeof(float));
  };

  std::vector<float> ToObs(int i) {
    auto [code, start, cursor, position, cost] = status_[i];

    std::vector<float> ob(ctx_[code].second.begin() + cursor * n_feature_,
                          ctx_[code].second.begin() +
                              (1 + cursor) * n_feature_);
    ob.push_back(position);
    return ob;
  }

  std::vector<float> ResetImpl(int i) {
    std::uniform_int_distribution<> dis(0, ctx_.size() - 1);
    int randomIndex = dis(gen);
    auto it = ctx_.begin();
    std::advance(it, randomIndex);
    int tot = ctx_[it->first].second.size() / n_feature_;
    std::uniform_int_distribution<int> dis_t(0, tot - T_ - 1);
    int start = dis_t(gen);
    status_[i] = {it->first, start, 0, 0, 0};
    return ToObs(i);
  }

  py::array_t<float> /*obs*/ Reset() {
    int d_obs = n_feature_ + 1;
    std::vector<float> obs(bs_ * d_obs);
    for (int i = 0; i < bs_; i++) {
      auto ob = ResetImpl(i);
      std::copy(ob.begin(), ob.end(), obs.data() + i * d_obs);
    }
    return py::array({bs_, d_obs}, obs.data());
  };

  std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<int>>
  Step(std::vector<int> actions) {
    int d_obs = n_feature_ + 1;

    std::vector<float> obs(bs_ * d_obs);
    std::vector<float> reward(bs_);
    std::vector<int> done(bs_);

    for (int i = 0; i < bs_; i++) {
      auto [code, start, cursor, position, cost] = status_[i];
      cursor++;
      position += actions[i] * amt_;
      auto spent = actions[i] * ctx_[code].first[5 * cursor + 2] * amt_;
      done[i] = (cursor >= T_) || position >= 1;
      if (cursor >= T_ && position < 1) {
        spent += ctx_[code].first[5 * cursor + 2] * (1 - position);
        position = 1;
      }
      cost += spent;
      status_[i] = {code, start, cursor, position, cost};
      if (done[i]) {
        int m = 1 / amt_;
        size_t step = T_ / m;
        float base = 0;
        float po = 0;
        for (size_t k = 0; k < m - 1; ++i) {
          base += amt_ * ctx_[code].first[5 * (start + k * step) + 2];
          po += amt_;
        }
        base += (1 - po) * ctx_[code].first[5 * (start + T_) + 2];
        reward[i] = cost / base - 1;

        auto ob = ResetImpl(i);
        std::copy(ob.begin(), ob.end(), obs.data() + i * d_obs);
      } else {
        reward[i] = 0;
        auto ob = ToObs(i);
        std::copy(ob.begin(), ob.end(), obs.data() + i * d_obs);
      }
    }

    return {py::array({bs_, d_obs}, obs.data()),
            py::array({bs_}, reward.data()), py::array({bs_}, done.data())};
  }

  int n_feature_;
  int bs_;
  int T_;
  float amt_;

  std::unordered_map<std::string, feature_t> ctx_;
  std::vector<status_t> status_;
};

namespace py = pybind11;

PYBIND11_MODULE(trade_env, m) {
  py::class_<vec_trade_t>(m, "VecTrade")
      .def(py::init<int, int, int, float>())
      .def("Reset", &vec_trade_t::Reset)
      .def("Step", &vec_trade_t::Step)
      .def("Load", &vec_trade_t::Load);
}
