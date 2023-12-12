#include "../../include/ops/add.hpp"
#include "../../include/value.hpp"
#include "../../include/graph.hpp"
#include <memory>

Add::Add(std::shared_ptr<Value> rhs, std::shared_ptr<Value> lhs) : Op(rhs), rhs_(rhs), lhs_(lhs){}
Add::Add(const float &rhs, std::shared_ptr<Value> lhs) : Op(lhs), lhs_(lhs){
  // Create the rhs in the graph
  rhs_ = graph.CreateValue(rhs);
}
Add::Add(std::shared_ptr<Value> rhs, const float &lhs) : Op(rhs), rhs_(rhs)
{
  // Create the lhs in the graph
  lhs_ = graph.CreateValue(lhs);
}

Value& Add::Forward() {
    // FIXME: You are here: CreateValue could output a reference to Value (it's the implementation that will be stored anyways)
    auto out = graph.CreateValue(rhs_->get_data() + lhs_->get_data());
    out->AddProducer(lhs_);
    out->AddProducer(rhs_);
    out->SetOp("+");
    return out;
}