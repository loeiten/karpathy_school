#include <filesystem>
#include <iostream>

#include "../include/print_graph.hpp"
#include "../include/value.hpp"

void SimpleGraph() {
  auto a = Value(2.0);
  auto b = Value(-3.0);
  auto c = Value(10.0);
  auto e = a * b;
  auto d = e + c;

  std::cout << ReturnDot(d) << std::endl;
}

int main(int argc, char** argv) {
  std::string filename = std::filesystem::path(argv[0]).filename();
  if (argc < 2) {
    std::cout << "Usage: ./" << filename << " test args" << std::endl;
    return EXIT_FAILURE;
  }

  if (std::string(argv[1]).compare("SimpleGraph") == 0) {
    SimpleGraph();
  } else {
    std::cerr << "No test named " << argv[1] << " in " << filename << std::endl;
    return EXIT_FAILURE;
  }
}
