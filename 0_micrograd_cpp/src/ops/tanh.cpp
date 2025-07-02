#include "../../include/ops/tanh.hpp"

#include <cmath>    // for exp, expm1, pow
#include <memory>   // for shared_ptr
#include <sstream>  // for char_traits, basic_ostream, oper...

#include "../../include/graph.hpp"   // for Graph
#include "../../include/ops/op.hpp"  // for Op
#include "../../include/value.hpp"   // for Value

Tanh::Tanh(std::shared_ptr<Value> arg) : Op(arg), arg_(arg) {}

Value &Tanh::Forward() {
  const double &x = arg_->get_data();
  // NOTE: We use expm1(x) instead of exp(x-1) to avoid loss of precision
  t_ = std::expm1(2 * x) / (std::exp(2 * x) + 1);
  auto &out = graph.CreateValue(t_);
  out_ = out.get_shared_ptr();
  out.AddProducer(arg_);
  out.set_op("tanh");
  std::stringstream ss;
  ss << "tanh_out_id_" << out.get_id();
  out.set_label(ss.str());
  return out;
}

void Tanh::Backward() {
  arg_->UpdateGrad((1 - std::pow(t_, 2)) * out_->get_grad());
}
