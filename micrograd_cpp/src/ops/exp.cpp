#include "../../include/ops/exp.hpp"

#include <cmath>
#include <memory>
#include <sstream>

#include "../../include/graph.hpp"
#include "../../include/ops/op.hpp"
#include "../../include/value.hpp"

Exp::Exp(std::shared_ptr<Value> exponent) : Op(exponent), exponent_(exponent) {}

Value &Exp::Forward() {
  auto &out = graph.CreateValue(std::exp(exponent_->get_data()));
  out_ = out.get_shared_ptr();
  out.AddProducer(exponent_);
  out.set_op("exp");
  std::stringstream ss;
  ss << "tmp_exp_out_id_" << out.get_id();
  out.set_label(ss.str());
  return out;
}

void Exp::Backward() {
  exponent_->UpdateGrad(out_->get_data() * out_->get_grad());
}
