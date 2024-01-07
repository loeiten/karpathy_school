#ifndef MICROGRAD_CPP_INCLUDE_MLP_HPP_
#define MICROGRAD_CPP_INCLUDE_MLP_HPP_

#include <memory>  // for shared_ptr
#include <vector>  // for vector

#include "../include/layer.hpp"  // for Layer

class Layer;
class Graph;

// We have public inheritance as we want get_parameters to be publicly
// accessible in the children
class MLP : public Layer {
 public:
  MLP(Graph& graph, const int& n_in,  // NOLINT
      const std::vector<int>& n_outs);

  std::vector<std::shared_ptr<Value>> operator()(
      const std::vector<std::shared_ptr<Value>>& x);

  std::vector<std::vector<std::shared_ptr<Value>>> Inference(
      const std::vector<std::vector<std::shared_ptr<Value>>>& examples);
  Value& Loss(const std::vector<std::vector<std::shared_ptr<Value>>>& examples,
              const std::vector<std::shared_ptr<Value>>& ground_truth);
  void Train(const std::vector<std::vector<std::shared_ptr<Value>>>& examples,
             const std::vector<std::shared_ptr<Value>>& ground_truth,
             const int& epochs);

 protected:
  int n_out;
  // We use vector as the order matter
  std::vector<std::shared_ptr<Layer>> layers;
};

#endif  // MICROGRAD_CPP_INCLUDE_MLP_HPP_
