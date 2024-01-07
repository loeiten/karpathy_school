#ifndef MICROGRAD_CPP_INCLUDE_MODULE_HPP_
#define MICROGRAD_CPP_INCLUDE_MODULE_HPP_

#include <memory>
#include <vector>

class Value;
class Graph;

class Module {
 public:
  explicit Module(Graph& graph);  // NOLINT (non-const reference)
  virtual ~Module() = default;

  std::vector<std::shared_ptr<Value>>& get_parameters();

  void ZeroGrad();

 protected:
  // We use vector as the order matter
  std::vector<std::shared_ptr<Value>> parameters;
  Graph& graph_;
};

#endif  // MICROGRAD_CPP_INCLUDE_MODULE_HPP_
