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
  Neuron(Graph& graph, const int& nin, const bool non_linear = true);  // NOLINT
  virtual ~Neuron() = default;

  Value& operator()(const std::vector<std::shared_ptr<Value>>& x);

 protected:
  bool non_linear_;
  std::shared_ptr<Value> b;
  // We use vector as the order matter
  std::vector<std::shared_ptr<Value>> w;
  std::shared_ptr<Value> activation_ptr;
};

#endif  // MICROGRAD_CPP_INCLUDE_NEURON_HPP_
