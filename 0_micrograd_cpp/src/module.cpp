#include "../include/module.hpp"

#include "../include/graph.hpp"
#include "../include/value.hpp"

Module::Module(Graph& graph) : graph_(graph) {}

std::vector<std::shared_ptr<Value>>& Module::get_parameters() {
  auto& parameter_singleton = ParametersSingleton::get_instance();
  return parameter_singleton.get_parameters();
}

void Module::ZeroGrad() {
  for (auto& parameter : get_parameters()) {
    parameter->set_grad(0.0);
  }
}
