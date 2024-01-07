#ifndef MICROGRAD_CPP_INCLUDE_MODULE_HPP_
#define MICROGRAD_CPP_INCLUDE_MODULE_HPP_

#include <memory>  // for shared_ptr
#include <vector>  // for vector

class Value;
class Graph;

class Module {
 public:
  explicit Module(Graph& graph);  // NOLINT (non-const reference)
  virtual ~Module() = default;

  std::vector<std::shared_ptr<Value>>& get_parameters();

  void ZeroGrad();

 protected:
  Graph& graph_;
};

// As Neuron inherits from Module, Neuron from Module etc., because derived
// classes needs to explicitly instantiate parent classes and because a Layer
// consist of several Neurons and so on, we need a common ground for the
// parameters
class ParametersSingleton {
 public:
  static ParametersSingleton& get_instance() {
    static ParametersSingleton instance;  // Guaranteed to be destroyed.
                                          // Instantiated on first use.
    return instance;
  }
  ParametersSingleton(ParametersSingleton const&) = delete;
  void operator=(ParametersSingleton const&) = delete;

  std::vector<std::shared_ptr<Value>>& get_parameters() { return parameters; }
  void add_parameter(const std::shared_ptr<Value>& value) {
    parameters.push_back(value);
  }

 private:
  ParametersSingleton() {}  // Constructor

  // We use vector as the order matter
  std::vector<std::shared_ptr<Value>> parameters;
};

#endif  // MICROGRAD_CPP_INCLUDE_MODULE_HPP_
