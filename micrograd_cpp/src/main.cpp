#include <iostream>  // for char_traits, operator<<, basic...

#include "../include/print_graph.hpp"  // for ReturnDot
#include "../include/value.hpp"        // for Value

int main() {
  auto n = Value(1.0f, "n");
  auto tmp = (2.0f * n);
  auto e = tmp.exp();
  e.set_label("e");

  e.Backward();

  std::cout << ReturnDot(e) << std::endl;

  return 0;
}
