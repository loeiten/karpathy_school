#include "../include/mlp.hpp"

#include <stddef.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "../include/graph.hpp"
#include "../include/ops/pow.hpp"
#include "../include/value.hpp"

MLP::MLP(Graph& graph, const int& n_in, const std::vector<int>& n_outs)
    : Layer(graph), n_out(n_outs.back()) {
  std::vector<int> sizes(n_outs);
  sizes.insert(sizes.begin(), n_in);
  bool non_linear = false;
  for (int i = 0; i < n_out; ++i) {
    layers.emplace_back(std::make_shared<Layer>(graph, sizes.at(i),
                                                sizes.at(i + 1), non_linear));
    // The last layer will have no non-linearities
    if (i == (n_out - 1)) {
      non_linear = true;
    }
  }
}

std::vector<std::shared_ptr<Value>> MLP::operator()(
    const std::vector<std::shared_ptr<Value>>& input) {
  auto x{input};
  for (const auto& layer : layers) {
    x = (*layer)(x);
  }
  return x;
}

std::vector<std::vector<std::shared_ptr<Value>>> MLP::Inference(
    const std::vector<std::vector<std::shared_ptr<Value>>>& examples) {
  std::vector<std::vector<std::shared_ptr<Value>>> output;

  for (const auto& example : examples) {
    output.push_back((*this)(example));
  }

  return output;
}

Value& MLP::Loss(
    const std::vector<std::vector<std::shared_ptr<Value>>>& examples,
    const std::vector<std::shared_ptr<Value>>& ground_truth) {
  std::stringstream ss;
  if (n_out != 1) {
    ss << "Loss only implemented for output size of 1, received output size of"
       << n_out;
    throw std::length_error(ss.str());
  }
  if (examples.size() != ground_truth.size()) {
    ss << "Size mismatch: examples (" << examples.size()
       << ") != ground_truth (" << ground_truth.size() << ")";
    throw std::length_error(ss.str());
  }
  auto output = Inference(examples);
  auto loss_ptr = graph_.CreateValue(0.0, "loss").get_shared_ptr();

  for (size_t i = 0; i < ground_truth.size(); ++i) {
    auto lhs_ptr = output.at(i).at(0);
    auto rhs_ptr = ground_truth.at(i);
    auto sub_ptr = ((*lhs_ptr) + (-(*rhs_ptr))).get_shared_ptr();
    loss_ptr = ((*loss_ptr) + pow((*sub_ptr), 2.0)).get_shared_ptr();
  }

  return (*loss_ptr);
}

void MLP::Train(
    const std::vector<std::vector<std::shared_ptr<Value>>>& examples,
    const std::vector<std::shared_ptr<Value>>& ground_truth,
    const int& epochs) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Forward pass
    auto loss_ptr = Loss(examples, ground_truth).get_shared_ptr();

    // Backward pass
    ZeroGrad();
    loss_ptr->Backward();

    // Update
    for (auto& parameter_ptr : get_parameters()) {
      // NOTE: We are only updating the parameters (that is weights and biases)
      //       Activations, intermediate Values etc. are not updated by this
      //       procedure
      parameter_ptr->set_data(parameter_ptr->get_data() -
                              0.05 * parameter_ptr->get_grad());
    }

    std::cout << "Epoch: " << epoch << " loss: " << std::scientific
              << std::setprecision(5) << loss_ptr->get_data() << std::endl;
  }
}
