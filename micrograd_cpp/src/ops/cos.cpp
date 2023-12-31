#include "../../include/ops/cos.hpp"

#include <cmath>    // for cos, sin
#include <memory>   // for shared_ptr
#include <sstream>  // for char_traits, basic_ostream, oper...

#include "../../include/graph.hpp"   // for Graph
#include "../../include/ops/op.hpp"  // for Op
#include "../../include/value.hpp"   // for Value

Cos::Cos(std::shared_ptr<Value> arg) : Op(arg), arg_(arg) {}

Value &Cos::Forward() {
  auto &out = graph.CreateValue(std::cos(arg_->get_data()));
  out_ = out.get_shared_ptr();
  out.AddProducer(arg_);
  out.set_op("cos");
  std::stringstream ss;
  ss << "cos_out_id_" << out.get_id();
  out.set_label(ss.str());
  return out;
}

void Cos::Backward() {
  arg_->UpdateGrad(-std::sin(arg_->get_data()) * out_->get_grad());
}
