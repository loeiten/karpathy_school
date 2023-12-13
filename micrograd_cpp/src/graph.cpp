#include "../include/graph.hpp"  // for Value, hash, equal_to, operator<<

#include <memory>
#include <set>      // for unordered_set, operator!=, __hash_co...
#include <sstream>  // for operator<<, basic_ostream, stringstream
#include <string>   // for char_traits, string, operator!=
#include <utility>  // for pair, make_pair

#include "../include/value.hpp"  // for Value, hash, equal_to, operator<<

std::shared_ptr<Value> Graph::CreateValue(const double &value) {
  // FIXME: Call pImpl in ctor
  auto it_existing = values.emplace(value);
  return *(it_existing.first);
}

// FIXME: Fix these stray dogs
void Graph::Trace(const Value &value, std::set<const Value *> *nodes,
                  std::set<std::pair<const Value *, const Value *>> *edges) {
  auto iterator_not_exist = nodes->insert(&value);
  // If the emplace succeeds (an insert happened), then the return returns true
  // which means that the element didn't existed in the past
  if (iterator_not_exist.second == true) {
    for (const auto &child : value.get_children()) {
      edges->insert(std::make_pair(child, &value));
      Trace(*child, nodes, edges);
    }
  }
}

std::string Graph::ReturnDot(const Value &root) {
  const int indent = 4;
  std::set<const Value *> nodes;
  std::set<std::pair<const Value *, const Value *>> edges;
  Trace(root, &nodes, &edges);

  std::stringstream dot_stream;
  dot_stream << "digraph {\n"
             << std::string(indent, ' ') << "graph [rankdir=LR]\n";

  // Create the nodes and connect the op to it's children
  for (const auto &node : nodes) {
    // Create the node
    dot_stream << std::string(indent, ' ') << "\"" << node->get_id() << "\""
               << " [label=\"{" << *node << "}\" shape=record]\n";
    if (node->get_op() != "") {
      // Connect op to the children
      dot_stream << std::string(indent, ' ') << "\"" << node->get_id()
                 << node->get_op() << "\" [label=\"" << node->get_op()
                 << "\"]\n";
      dot_stream << std::string(indent, ' ') << "\"" << node->get_id()
                 << node->get_op() << "\" -> \"" << node->get_id() << "\""
                 << "\n";
    }
  }

  // Create the rest of the edges
  for (const auto &child_parent : edges) {
    auto &child = child_parent.first;
    auto &parent = child_parent.second;
    dot_stream << std::string(indent, ' ') << "\"" << child->get_id()
               << "\" -> \"" << parent->get_id() << parent->get_op() << "\"\n";
  }

  dot_stream << "}";
  return dot_stream.str();
}
