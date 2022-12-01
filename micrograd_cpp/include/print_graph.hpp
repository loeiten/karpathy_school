#ifndef MICROGRAD_CPP_INCLUDE_PRINT_GRAPH_HPP_
#define MICROGRAD_CPP_INCLUDE_PRINT_GRAPH_HPP_

#include <set>      // for set
#include <string>   // for string
#include <utility>  // for pair

class Value;

void Trace(const Value &value, std::set<Value *> *nodes,
           std::set<std::pair<Value *, Value *>> *edges);

std::string ReturnDot(const Value &root);

#endif  // MICROGRAD_CPP_INCLUDE_PRINT_GRAPH_HPP_
