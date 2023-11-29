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

  return EXIT_SUCCESS;
}
