#include <cstdlib>     // for EXIT_FAILURE, EXIT_SUCCESS
#include <filesystem>  // for path
#include <ios>
#include <iostream>  // for operator<<, basic_ostream, endl, cha...
#include <memory>
#include <sstream>
#include <string>  // for char_traits, string
#include <vector>

#include "../include/graph.hpp"   // for Graph
#include "../include/neuron.hpp"  // for Value, cos, exp, operator*, operator+
#include "../include/value.hpp"   // for Value, cos, exp, operator*, operator+

void SimpleNeuron(const bool non_linear) {
  // Create graph where Values reside
  auto graph = Graph();
  int nin = 2;
  Neuron neuron{graph, nin, non_linear};

  // Manually change the values to get reproducibility
  auto &parameters = neuron.get_parameters();
  // b, w1, w2
  std::vector<double> own_params{0.0, 0.2355, 0.0655};
  std::stringstream ss;
  // +1 due to the bias
  for (int i = 0; i < (nin + 1); ++i) {
    parameters.at(i)->set_data(own_params.at(i));
  }

  std::vector<std::shared_ptr<Value>> x;
  x.push_back(graph.CreateValue(1.0, "x1").get_shared_ptr());
  x.push_back(graph.CreateValue(-2.0, "x2").get_shared_ptr());

  auto &y = neuron(x);
  y.Backward();

  std::cout << graph.ReturnDot(y) << std::endl;
}

int main(int argc, char **argv) {
  std::string filename = std::filesystem::path(argv[0]).filename();
  if (argc < 2) {
    std::cout << "Usage: ./" << filename << " test args" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::string(argv[1]).compare("LinearNeuron") == 0) {
    bool non_linear = false;
    SimpleNeuron(non_linear);
  } else if (std::string(argv[1]).compare("NonLinearNeuron") == 0) {
    bool non_linear = true;
    SimpleNeuron(non_linear);
  } else {
    std::cerr << "No test named " << argv[1] << " in " << filename << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
