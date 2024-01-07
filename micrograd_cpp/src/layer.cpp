#include "../include/layer.hpp"

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr

#include "../include/neuron.hpp"
#include "../include/value.hpp"

Layer::Layer(Graph& graph) : Neuron(graph) {}

Layer::Layer(Graph& graph, const int& n_in, const int& n_out,
             const bool non_linear)
    : Neuron(graph) {
  for (int _ = 0; _ < n_out; ++_) {
    in_nodes.emplace_back(std::make_shared<Neuron>(graph, n_in, non_linear));
  }
}

std::vector<std::shared_ptr<Value>>
Layer::operator()(  // cppcheck-suppress duplInheritedMember
    const std::vector<std::shared_ptr<Value>>& x) {
  std::vector<std::shared_ptr<Value>> out_values;
  for (std::size_t i = 0; i < in_nodes.size(); ++i) {
    // NOTE: This creates new Values
    out_values.emplace_back(((*in_nodes.at(i))(x)).get_shared_ptr());
  }
  return out_values;
}
