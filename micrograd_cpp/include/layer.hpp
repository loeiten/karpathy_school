#ifndef MICROGRAD_CPP_INCLUDE_LAYER_HPP_
#define MICROGRAD_CPP_INCLUDE_LAYER_HPP_

#include <memory>  // for shared_ptr
#include <vector>  // for vector

#include "../include/neuron.hpp"  // for Neuron

class Graph;
class Value;

// We have public inheritance as we want get_parameters to be publicly
// accessible in the children
class Layer : public Neuron {
 public:
  // See neuron.hpp on why we need this
  explicit Layer(Graph& graph);                           // NOLINT
  Layer(Graph& graph, const int& n_in, const int& n_out,  // NOLINT
        const bool non_linear = true);
  Layer(const Layer& layer) = delete;
  virtual ~Layer() = default;

  // NOTE: The out nodes shares the input, it's the weights and biases of the
  // out nodes which differs
  // The situation can be depicted like this
  // For illustrative purposes x1, x2 and x3 are depicted twice
  //
  // Input Weight   Bias Activation
  //                 b1
  //   x1 - w11   \    \
  //   x2 - w12   -       o1
  //   x3 - w13   /
  //                 b2
  //   x1 - w21   \    \
  //   x2 - w22   -       o2
  //   x3 - w23   /
  std::vector<std::shared_ptr<Value>>
  operator()(  // cppcheck-suppress duplInheritedMember
      const std::vector<std::shared_ptr<Value>>& x);

 protected:
  // We use vector as the order matter
  std::vector<std::shared_ptr<Neuron>> in_nodes;
};

#endif  // MICROGRAD_CPP_INCLUDE_LAYER_HPP_
