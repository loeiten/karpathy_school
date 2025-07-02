#include "../../include/ops/exp.hpp"

#include <cmath>    // for exp
#include <memory>   // for shared_ptr
#include <sstream>  // for char_traits, basic_ostream, oper...

#include "../../include/graph.hpp"   // for Graph
#include "../../include/ops/op.hpp"  // for Op
#include "../../include/value.hpp"   // for Value

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
