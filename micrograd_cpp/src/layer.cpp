#include "../include/layer.hpp"

#include <memory>

#include "../include/value.hpp"

Layer::Layer(Graph& graph, const int& nin, const int& nout,
             const bool non_linear)
    : Neuron(graph), nout_(nout) {
  for (int _ = 0; _ < nin; ++_) {
    in_nodes.emplace_back(std::make_shared<Neuron>(graph, nin, non_linear));
  }
}

std::vector<std::shared_ptr<Value>>& Layer::operator()(
    const std::vector<std::shared_ptr<Value>>& x) {
  for (int i = 0; i < nout_; ++i) {
    out_values.emplace_back(((*in_nodes.at(i))(x)).get_shared_ptr());
  }
  return out_values;
}
