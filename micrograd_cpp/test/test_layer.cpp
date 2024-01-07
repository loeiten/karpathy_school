#include <cstdlib>     // for EXIT_FAILURE, EXIT_SUCCESS
#include <filesystem>  // for path
#include <iostream>    // for operator<<, basic_ostream, endl, cha...
#include <memory>      // for shared_ptr
#include <string>      // for char_traits, string
#include <vector>      // for vector

#include "../include/graph.hpp"  // for Graph
#include "../include/layer.hpp"  // for Layer
#include "../include/value.hpp"  // for Value

void SimpleLayer(const bool non_linear) {
  // Create graph where Values reside
  auto graph = Graph();
  int n_in = 2;
  int n_out = 2;
  Layer layer{graph, n_in, n_out, non_linear};

  // Manually change the values to get reproducibility
  auto &parameters = layer.get_parameters();
  // b1, w12, w12, b2, w22, w22
  std::vector<double> own_params{0.0, 0.2355, 0.0655, 0.0, 0.04, -0.3};
  // +1 due to the bias
  for (int i = 0; i < (n_in + 1) * n_out; ++i) {
    parameters.at(i)->set_data(own_params.at(i));
  }

  std::vector<std::shared_ptr<Value>> x;
  x.push_back(graph.CreateValue(0.5, "x1").get_shared_ptr());
  x.push_back(graph.CreateValue(-2.0, "x2").get_shared_ptr());

  const auto &y = layer(x);

  for (const auto &output : y) {
    output->Backward();
  }

  std::cout << graph.ReturnDot(y) << std::endl;
}

int main(int argc, char **argv) {
  std::string filename = std::filesystem::path(argv[0]).filename();
  if (argc < 2) {
    std::cout << "Usage: ./" << filename << " test args" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::string(argv[1]).compare("LinearLayer") == 0) {
    bool non_linear = false;
    SimpleLayer(non_linear);
  } else if (std::string(argv[1]).compare("NonLinearLayer") == 0) {
    bool non_linear = true;
    SimpleLayer(non_linear);
  } else {
    std::cerr << "No test named " << argv[1] << " in " << filename << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
