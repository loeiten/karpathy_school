#include "../include/module.hpp"

#include "../include/graph.hpp"

Module::Module(Graph& graph) : graph_(graph) {}

std::vector<std::shared_ptr<Value>>& Module::get_parameters() {
  return parameters;
}
