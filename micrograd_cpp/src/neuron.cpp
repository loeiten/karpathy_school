#include "../include/neuron.hpp"

#include <exception>
#include <memory>
#include <random>
#include <sstream>

#include "../include/graph.hpp"
#include "../include/ops/tanh.hpp"
#include "../include/value.hpp"

Neuron::Neuron(Graph& graph, const int& nin, const bool non_linear)
    : Module(graph), non_linear_(non_linear) {
  // Start the random generator
  std::random_device rd;   // Generates an integer
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  // Create the bias
  b = graph.CreateValue(0).get_shared_ptr();
  parameters.push_back(b);
  std::stringstream ss;
  ss << "b_" << b->get_id();
  b->set_label(ss.str());
  ss.str("");
  ss.clear();

  // Create the weights
  for (int _ = 0; _ < nin; ++_) {
    w.push_back(graph.CreateValue(dis(gen)).get_shared_ptr());
    auto w_ptr = w.back();
    parameters.push_back(w_ptr);
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
    activation_ptr =
        ((*activation_ptr) + (*w.at(i)) * (*x.at(i))).get_shared_ptr();
  }

  ss.str("");
  ss.clear();
  ss << "activation_" << activation_ptr->get_id();
  activation_ptr->set_label(ss.str());

  if (non_linear_) {
    activation_ptr = tanh(*(activation_ptr.get())).get_shared_ptr();
  }

  return *(activation_ptr.get());
}
