#include "../../include/ops/add.hpp"
#include <memory>

Add::Add(std::shared_ptr<Value> rhs, std::shared_ptr<Value> lhs) : Op(rhs->get_graph()), rhs_(rhs), lhs_(lhs){}
Add::Add(const float &rhs, std::shared_ptr<Value> lhs) : Op(lhs->get_graph()), lhs_(lhs){
  // Create the rhs in the graph
  rhs_ = graph.CreateValue(rhs);
}
Add::Add(std::shared_ptr<Value> rhs, const float &lhs) : Op(rhs->get_graph()), rhs_(rhs)
{
  // Create the lhs in the graph
  lhs_ = graph.CreateValue(lhs);
}

std::shared_ptr<Value> Add::Forward() {
  
}