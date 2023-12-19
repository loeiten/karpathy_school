#include <iostream>  // for char_traits, operator<<, basic...

#include "../include/graph.hpp"  // for ReturnDot
#include "../include/value.hpp"        // for Value

int main() {
  auto graph = Graph();
  auto n = Value(1.0f, "n");
  auto tmp = (2.0f * n);
  auto e = exp(tmp);
  e.set_label("e");

  e.Backward();

  std::cout << graph.ReturnDot(e) << std::endl;

  return 0;
}
