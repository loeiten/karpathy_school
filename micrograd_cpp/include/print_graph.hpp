#ifndef MICROGRAD_CPP_INCLUDE_PRINT_GRAPH_HPP_
#define MICROGRAD_CPP_INCLUDE_PRINT_GRAPH_HPP_

#include <string>
#include <unordered_set>
#include <utility>

class Value;

void Trace(const Value &value, std::unordered_set<Value> *nodes,
           std::unordered_set<std::pair<Value, Value>> *edges);

std::string ReturnDot(const Value &root);

#endif  // MICROGRAD_CPP_INCLUDE_PRINT_GRAPH_HPP_
