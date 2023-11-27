#include <memory>         // for shared_ptr
#include <sstream>        // for operator<<, basic_ostream, char_traits
#include <string>         // for string, operator!=
#include <unordered_set>  // for unordered_set, operator!=, __hash_co...
#include <utility>        // for pair, make_pair

#include "../include/value.hpp"  // for Value, hash, equal_to, operator<<

void Trace(const Value &value, std::unordered_set<Value> *nodes,
           std::unordered_set<std::pair<Value, Value>> *edges) {
  auto iterator_not_exist = nodes->emplace(value);
  // If the emplace succeeds (an insert happened), then the return returns true
  // which means that the element didn't existed in the past
  if (iterator_not_exist.second == true) {
    for (const auto &child : value.get_children()) {
      edges->emplace(std::make_pair(*child, value));
      Trace(*child, nodes, edges);
    }
  }
}

std::string ReturnDot(const Value &root) {
  const int indent = 4;
  // FIXME: Here the Values are "hard copies", would it make sense to convert
  //        them to pointers?
  std::unordered_set<Value> nodes;
  std::unordered_set<std::pair<Value, Value>> edges;
  Trace(root, &nodes, &edges);

  std::stringstream dot_stream;
  dot_stream << "digraph {\n"
             << std::string(indent, ' ') << "graph [rankdir=LR]\n";

  // Create the nodes and connect the op to it's children
  for (const auto &node : nodes) {
    // Create the node
    dot_stream << std::string(indent, ' ') << "\"" << node.get_id() << "\""
               << " [label=\"{" << node << "}\" shape=record]\n";
    if (node.get_op() != "") {
      // Connect op to the children
      dot_stream << std::string(indent, ' ') << "\"" << node.get_id()
                 << node.get_op() << "\" [label=\"" << node.get_op() << "\"]\n";
      dot_stream << std::string(indent, ' ') << "\"" << node.get_id()
                 << node.get_op() << "\" -> \"" << node.get_id() << "\""
                 << "\n";
    }
  }

  // Create the rest of the edges
  for (const auto &child_parent : edges) {
    auto &child = child_parent.first;
    auto &parent = child_parent.second;
    dot_stream << std::string(indent, ' ') << "\"" << child.get_id()
               << "\" -> \"" << parent.get_id() << parent.get_op() << "\"\n";
  }

  dot_stream << "}";
  return dot_stream.str();
}
