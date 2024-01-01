#include "../../include/ops/cos.hpp"

#include <memory>

#include "../../include/graph.hpp"
#include "../../include/value.hpp"

Cos::Cos(std::shared_ptr<Value> arg) : Op(arg), arg_(arg) {}

Value &Cos::Forward() {
  auto &out = graph.CreateValue(std::cos(arg_->get_data()));
  out_ = out.get_shared_ptr();
  out.AddProducer(arg_);
  out.set_op("cos");
  return out;
}

void Cos::Backward() { arg_->UpdateGrad(-std::sin(arg_->get_data()) * out_->get_grad()); }