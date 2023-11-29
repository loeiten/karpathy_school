#include <iostream>  // for char_traits, operator<<, basic...

#include "../include/print_graph.hpp"  // for ReturnDot
#include "../include/value.hpp"        // for Value

int main() {
  auto a = Value(2.0, "a");
  auto b = Value(-3.0, "b");
  auto c = Value(10.0, "c");
  auto e = a * b;
  e.set_label("e");
  auto d = e + c;
  d.set_label("d");

  std::cout << ReturnDot(d) << std::endl;

  return 0;
}
