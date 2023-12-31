#include "../include/neuron.hpp"

#include <memory>     // for shared_ptr
#include <random>     // for random_device, mt19937, uniform_real_distribution
#include <sstream>    // for stringstream
#include <stdexcept>  // for length_error

#include "../include/graph.hpp"     // for Graph
#include "../include/module.hpp"    // for module
#include "../include/ops/tanh.hpp"  // for Tanh
#include "../include/value.hpp"     // for Value

Neuron::Neuron(Graph& graph) : Module(graph), non_linear_(true) {}

Neuron::Neuron(Graph& graph, const int& n_in, const bool non_linear)
    : Module(graph), non_linear_(non_linear) {
  auto& parameter_singleton = ParametersSingleton::get_instance();

  // Start the random generator
  std::random_device rd;   // Generates an integer
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  // Create the bias
  b = graph.CreateValue(0).get_shared_ptr();
  parameter_singleton.add_parameter(b);
  std::stringstream ss;
  ss << "b_" << b->get_id();
  b->set_label(ss.str());
  ss.str("");
  ss.clear();

  // Create the weights
  for (int _ = 0; _ < n_in; ++_) {
    w.push_back(graph.CreateValue(dis(gen)).get_shared_ptr());
    auto w_ptr = w.back();
    parameter_singleton.add_parameter(w_ptr);
    ss << "w_" << w_ptr->get_id();
    w_ptr->set_label(ss.str());
    ss.str("");
    ss.clear();
  }
}

Value& Neuron::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  std::stringstream ss;
  if (x.size() != w.size()) {
    ss << "Size mismatch: x(" << x.size() << ") != w(" << w.size() << ")";
    throw std::length_error(ss.str());
  }

  activation_ptr = b;
  for (unsigned int i = 0; i < x.size(); ++i) {
    // NOTE: We are creating new Values for each operation we are doing
    activation_ptr =
        ((*activation_ptr) + (*w.at(i)) * (*x.at(i))).get_shared_ptr();
  }

  ss.str("");
  ss.clear();
  ss << "activation_" << activation_ptr->get_id();
  activation_ptr->set_label(ss.str());

  if (non_linear_) {
    activation_ptr = tanh(*(activation_ptr)).get_shared_ptr();
  }

  return *(activation_ptr);
}
