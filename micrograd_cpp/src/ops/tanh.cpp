#include "../../include/ops/tanh.hpp"

#include <memory>

#include "../../include/graph.hpp"
#include "../../include/value.hpp"

Tanh::Tanh(std::shared_ptr<Value> arg) : Op(arg), arg_(arg) {}

Value &Tanh::Forward() {
  const double &x = arg_->get_data();
  t_ = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
  auto &out = graph.CreateValue(t_);
  out_ = out.get_shared_ptr();
  out.AddProducer(arg_);
  out.set_op("tanh");
  return out;
}

void Tanh::Backward() {
  arg_->UpdateGrad((1 - std::pow(t_, 2)) * out_->get_grad()); 
}