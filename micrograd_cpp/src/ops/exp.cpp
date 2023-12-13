#include "../../include/ops/exp.hpp"

#include <memory>

#include "../../include/graph.hpp"
#include "../../include/value.hpp"

Exp::Exp(std::shared_ptr<Value> exponent) : Op(exponent), exponent_(exponent) {}

Value &Exp::Forward() {
  auto &out = graph.CreateValue(std::exp(exponent_->get_data()));
  out_ = out.get_shared_ptr();
  out.AddProducer(exponent_);
  out.set_op("exp");
  return out;
}

void Exp::Backward() { out_->UpdateGrad(out_->get_data() * out_->get_grad()); }