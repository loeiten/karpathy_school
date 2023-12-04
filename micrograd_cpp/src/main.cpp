#include <iostream>  // for char_traits, operator<<, basic...

#include "../include/print_graph.hpp"  // for ReturnDot
#include "../include/value.hpp"        // for Value

int main() {
  auto n = Value(1.0f, "n");
  auto e = 2.0f * n;
  e.set_label("e");

  e.Backward();

  std::cout << ReturnDot(e) << std::endl;

  return 0;
}
