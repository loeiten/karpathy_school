#include <cstdlib>     // for EXIT_FAILURE, EXIT_SUCCESS
#include <filesystem>  // for path
#include <iostream>    // for operator<<, basic_ostream, endl, cha...
#include <string>      // for char_traits, string

#include "../include/graph.hpp"  // for Graph
#include "../include/value.hpp"  // for Value, cos, exp, operator*, operator+

void SimpleGraph() {
  // Create graph where Values reside
  auto graph = Graph();
  auto &a = graph.CreateValue(2.0, "a");
  auto &b = graph.CreateValue(-3.0, "b");
  auto &c = graph.CreateValue(10.0, "c");
  auto &e = a * b;
  e.set_label("e");
  auto &d = e + c;
  d.set_label("d");

  std::cout << graph.ReturnDot(d) << std::endl;
}

void BackProp() {
  // Create graph where Values reside
  auto graph = Graph();
  // Inputs x1, x2
  auto &x1 = graph.CreateValue(2.0, "x1");
  auto &x2 = graph.CreateValue(0.0, "x2");
  // Weights w1, w2
  auto &w1 = graph.CreateValue(-3.0, "w1");
  auto &w2 = graph.CreateValue(1.0, "w2");
  // Bias of the neuron
  auto &b = graph.CreateValue(6.8813735870195432, "b");
  // x1w1 + x2w2 + b
  auto &x1w1 = x1 * w1;
  x1w1.set_label("x1w1");
  auto &x2w2 = x2 * w2;
  x2w2.set_label("x2w2");
  auto &x1w1x2w2 = x1w1 + x2w2;
  x1w1x2w2.set_label("x1w1 + x2w2");
  auto &n = x1w1x2w2 + b;
  n.set_label("n");
  // Output
  auto &o = tanh(n);
  o.set_label("o");

  o.Backward();

  std::cout << graph.ReturnDot(o) << std::endl;
}

void GraphWithTemporaries() {
  // Create graph where Values reside
  auto graph = Graph();
  auto &t1 = graph.CreateValue(4, "t1");
  auto &t2 = graph.CreateValue(1, "t2");
  auto &t4 = graph.CreateValue(3, "t4");
  auto &o = cos((t1 + t2) * t4);
  o.set_label("o");

  o.Backward();

  std::cout << graph.ReturnDot(o) << std::endl;
}

void ReuseVariable() {
  // Create graph where Values reside
  auto graph = Graph();
  auto &a = graph.CreateValue(-2.0, "a");
  auto &b = graph.CreateValue(3.0, "b");
  auto &d = a * b;
  d.set_label("d");
  auto &e = a + b;
  e.set_label("e");
  auto &f = d * e;
  f.set_label("f");

  f.Backward();

  std::cout << graph.ReturnDot(f) << std::endl;
}

void CompoundOps() {
  // Create graph where Values reside
  auto graph = Graph();
  auto &a = graph.CreateValue(4.0, "a");
  auto &b = graph.CreateValue(2.0, "b");
  auto &o = a / b;
  o.set_label("o");

  o.Backward();

  std::cout << graph.ReturnDot(o) << std::endl;
}

void TanhSpelledOut() {
  // Create graph where Values reside
  auto graph = Graph();
  // Inputs x1, x2
  auto &x1 = graph.CreateValue(2.0, "x1");
  auto &x2 = graph.CreateValue(0.0, "x2");
  // Weights w1, w2
  auto &w1 = graph.CreateValue(-3.0, "w1");
  auto &w2 = graph.CreateValue(1.0, "w2");
  // Bias of the neuron
  auto &b = graph.CreateValue(6.8813735870195432, "b");
  // x1w1 + x2w2 + b
  auto &x1w1 = x1 * w1;
  x1w1.set_label("x1w1");
  auto &x2w2 = x2 * w2;
  x2w2.set_label("x2w2");
  auto &x1w1x2w2 = x1w1 + x2w2;
  x1w1x2w2.set_label("x1w1 + x2w2");
  auto &n = x1w1x2w2 + b;
  n.set_label("n");
  // Output
  auto &e = exp(2.0f * n);
  e.set_label("e");
  auto &o = (e - 1.0f) / (e + 1.0f);
  o.set_label("o");

  o.Backward();

  std::cout << graph.ReturnDot(o) << std::endl;
}

int main(int argc, char **argv) {
  std::string filename = std::filesystem::path(argv[0]).filename();
  if (argc < 2) {
    std::cout << "Usage: ./" << filename << " test args" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::string(argv[1]).compare("SimpleGraph") == 0) {
    SimpleGraph();
  } else if (std::string(argv[1]).compare("BackProp") == 0) {
    BackProp();
  } else if (std::string(argv[1]).compare("GraphWithTemporaries") == 0) {
    GraphWithTemporaries();
  } else if (std::string(argv[1]).compare("ReuseVariable") == 0) {
    ReuseVariable();
  } else if (std::string(argv[1]).compare("CompoundOps") == 0) {
    CompoundOps();
  } else if (std::string(argv[1]).compare("TanhSpelledOut") == 0) {
    TanhSpelledOut();
  } else {
    std::cerr << "No test named " << argv[1] << " in " << filename << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
