#include "../include/mlp.hpp"

#include <cstddef>    // for size_t
#include <iomanip>    // for operator<<, setprecision
#include <iostream>   // for basic_ostream, char_traits, operator<<
#include <memory>     // for shared_ptr, make_shared
#include <sstream>    // for stringstream
#include <stdexcept>  // for length_error
#include <vector>     // for vector

#include "../include/graph.hpp"    // for Graph
#include "../include/layer.hpp"    // for Layer
#include "../include/ops/pow.hpp"  // for Pow
#include "../include/value.hpp"    // for Value, pow

MLP::MLP(Graph& graph, const int& n_in, const std::vector<int>& n_outs)
    : Layer(graph), n_out(n_outs.back()) {
  std::vector<int> sizes(n_outs);
  sizes.insert(sizes.begin(), n_in);
  bool non_linear = true;
  for (std::size_t i = 0; i < n_outs.size(); ++i) {
    layers.emplace_back(std::make_shared<Layer>(graph, sizes.at(i),
                                                sizes.at(i + 1), non_linear));
    // The last layer will have no non-linearities
    if (i == (n_outs.size() - 1)) {
      non_linear = false;
    }
  }
}

std::vector<std::shared_ptr<Value>>
MLP::operator()(  // cppcheck-suppress duplInheritedMember
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
    // NOTE: Each time we are calling inference we are creating new Values
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
  // NOTE: Calculating the Loss is creating new Values
  auto outputs = Inference(examples);
  auto loss_ptr = graph_.CreateValue(0.0, "loss").get_shared_ptr();

  for (std::size_t i = 0; i < ground_truth.size(); ++i) {
    auto lhs_ptr = outputs.at(i).at(0);
    auto rhs_ptr = ground_truth.at(i);
    auto sub_ptr = ((*lhs_ptr) - (*rhs_ptr)).get_shared_ptr();
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
    // NOTE: We create a new loss Value each time this is called
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
