#include "../include/graph.hpp"  // for Value, hash, equal_to, operator<<

#include <memory>   // for shared_ptr, make_shared
#include <set>      // for set
#include <sstream>  // for operator<<, basic_ostream, stringstream
#include <string>   // for char_traits, string, operator!=
#include <utility>  // for pair, make_pair
#include <vector>   // for vector

#include "../include/value.hpp"  // for Value, operator<<

Value &Graph::CreateValue(const double value) {
  auto it_existing = values.emplace(std::make_shared<Value>(*this, value));
  auto value_shared_ptr = *(it_existing.first);
  value_shared_ptr->set_shared_ptr(value_shared_ptr);
  // In order to chain, we need to return references
  return *(value_shared_ptr);
}

Value &Graph::CreateValue(const double value, const std::string &label) {
  auto it_existing =
      values.emplace(std::make_shared<Value>(*this, value, label));
  auto value_shared_ptr = *(it_existing.first);
  value_shared_ptr->set_shared_ptr(value_shared_ptr);
  // In order to chain, we need to return references
  return *(value_shared_ptr);
}

void Graph::TopologicalSort(const Value &value) {
  if (visited.find(value.get_id()) == visited.end()) {
    visited.insert(value.get_id());
    for (const auto &producer : value.get_producers()) {
      TopologicalSort(*producer);
    }
    topology.push_back(value.get_shared_ptr());
  }
}

void Graph::Trace(const Value &value,
                  std::set<const std::shared_ptr<Value>> *nodes,
                  std::set<std::pair<const std::shared_ptr<Value>,
                                     const std::shared_ptr<Value>>> *edges) {
  auto iterator_not_exist = nodes->insert(value.get_shared_ptr());
  // If the emplace succeeds (an insert happened), then the return returns true
  // which means that the element didn't existed in the past
  if (iterator_not_exist.second == true) {
    for (const auto &producer : value.get_producers()) {
      edges->insert(std::make_pair(producer, value.get_shared_ptr()));
      Trace(*producer, nodes, edges);
    }
  }
}

std::string Graph::ReturnDot(const Value &root) {
  std::set<const std::shared_ptr<Value>> nodes;
  std::set<
      std::pair<const std::shared_ptr<Value>, const std::shared_ptr<Value>>>
      edges;
  Trace(root, &nodes, &edges);
  return CreateDotString(nodes, edges);
}

std::string Graph::ReturnDot(const std::vector<std::shared_ptr<Value>> &x) {
  std::set<const std::shared_ptr<Value>> nodes;
  std::set<
      std::pair<const std::shared_ptr<Value>, const std::shared_ptr<Value>>>
      edges;
  for (const auto &root : x) {
    Trace(*(root->get_shared_ptr()), &nodes, &edges);
  }
  return CreateDotString(nodes, edges);
}

std::string Graph::CreateDotString(
    const std::set<const std::shared_ptr<Value>> &nodes,
    const std::set<std::pair<const std::shared_ptr<Value>,
                             const std::shared_ptr<Value>>> &edges) {
  const int indent = 4;
  std::stringstream dot_stream;
  dot_stream << "digraph {\n"
             << std::string(indent, ' ') << "graph [rankdir=LR]\n";

  // Create the nodes and connect the op to it's producers
  for (const auto &node : nodes) {
    // Create the node
    dot_stream << std::string(indent, ' ') << "\"" << node->get_id() << "\""
               << " [label=\"{" << *node << "}\" shape=record]\n";
    if (node->get_op() != "") {
      // Connect op to the producers
      dot_stream << std::string(indent, ' ') << "\"" << node->get_id()
                 << node->get_op() << "\" [label=\"" << node->get_op()
                 << "\"]\n";
      dot_stream << std::string(indent, ' ') << "\"" << node->get_id()
                 << node->get_op() << "\" -> \"" << node->get_id() << "\""
                 << "\n";
    }
  }

  // Create the rest of the edges
  for (const auto &producer_consumer : edges) {
    auto &producer = producer_consumer.first;
    auto &consumer = producer_consumer.second;
    dot_stream << std::string(indent, ' ') << "\"" << producer->get_id()
               << "\" -> \"" << consumer->get_id() << consumer->get_op()
               << "\"\n";
  }

  dot_stream << "}";
  return dot_stream.str();
}
