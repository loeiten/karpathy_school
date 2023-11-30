#include <cstdlib>     // for EXIT_FAILURE, EXIT_SUCCESS
#include <filesystem>  // for path
#include <iostream>    // for operator<<, basic_ostream, endl
#include <string>      // for char_traits, string

#include "../include/print_graph.hpp"  // for ReturnDot
#include "../include/value.hpp"        // for Value

void SimpleGraph() {
  auto a = Value(2.0, "a");
  auto b = Value(-3.0, "b");
  auto c = Value(10.0, "c");
  auto e = a * b;
  e.set_label("e");
  auto d = e + c;
  d.set_label("d");

  std::cout << ReturnDot(d) << std::endl;
}

void BackProp() {
  // Inputs x1, x2
  auto x1 = Value(2.0, "x1");
  auto x2 = Value(0.0, "x2");
  // Weights w1, w2
  auto w1 = Value(-3.0, "w1");
  auto w2 = Value(1.0, "w2");
  // Bias of the neuron
  auto b = Value(6.8813735870195432, "b");
  // x1w1 + x2w2 + b
  auto x1w1 = x1 * w1;
  x1w1.set_label("x1w1");
  auto x2w2 = x2 * w2;
  x2w2.set_label("x2w2");
  auto x1w1x2w2 = x1w1 * x2w2;
  x1w1x2w2.set_label("x1w1 + x2w2");
  auto n = x1w1x2w2 + b;
  n.set_label("n");
  // Output
  auto o = n.tanh();
  o.set_label("o");

  o.Backward();

  std::cout << ReturnDot(o) << std::endl;
}

int main(int argc, char** argv) {
  std::string filename = std::filesystem::path(argv[0]).filename();
  if (argc < 2) {
    std::cout << "Usage: ./" << filename << " test args" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::string(argv[1]).compare("SimpleGraph") == 0) {
    SimpleGraph();
  } else if (std::string(argv[1]).compare("BackProp") == 0) {
    BackProp();
  } else {
    std::cerr << "No test named " << argv[1] << " in " << filename << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
