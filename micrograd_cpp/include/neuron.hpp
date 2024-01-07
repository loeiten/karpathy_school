#ifndef MICROGRAD_CPP_INCLUDE_NEURON_HPP_
#define MICROGRAD_CPP_INCLUDE_NEURON_HPP_

#include <memory>  // for shared_ptr
#include <vector>  // for vector

#include "../include/module.hpp"  // for Module

class Value;
class Graph;

// We have public inheritance as we want get_parameters to be publicly
// accessible in the children
class Neuron : public Module {
 public:
  // Layer is a child of Neuron and hence need to call a ctor
  // The default ctor of Neuron of Neuron is implicitly deleted (however here we
  // do it explicitly) since Module doesn't have a default ctor
  // Module doesn't have a default ctor as it takes in graph by reference
  Neuron() = delete;
  explicit Neuron(Graph& graph);                                       // NOLINT
  Neuron(Graph& graph, const int& nin, const bool non_linear = true);  // NOLINT
  virtual ~Neuron() = default;
  // Input Weight   Bias Activation
  //                 b
  //   x1 - w1   \    \
  //   x2 - w2   -       o1
  //   x3 - w3   /
  Value& operator()(const std::vector<std::shared_ptr<Value>>& x);

 protected:
  bool non_linear_;
  std::shared_ptr<Value> b;
  // We use vector as the order matter
  std::vector<std::shared_ptr<Value>> w;
  std::shared_ptr<Value> activation_ptr;
};

#endif  // MICROGRAD_CPP_INCLUDE_NEURON_HPP_
