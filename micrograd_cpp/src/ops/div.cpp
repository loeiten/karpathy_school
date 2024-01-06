#include "../../include/ops/div.hpp"

#include <functional>
#include <iomanip>  // for operator<<, setprecision
#include <memory>
#include <sstream>

#include "../../include/graph.hpp"
#include "../../include/ops/mul.hpp"
#include "../../include/ops/op.hpp"
#include "../../include/ops/pow.hpp"
#include "../../include/value.hpp"

Div::Div(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
    : Op(rhs), rhs_(rhs), lhs_(lhs) {}

Div::Div(std::shared_ptr<Value> lhs, const double &rhs) : Op(lhs), lhs_(lhs) {
  // Create the rhs in the graph
  auto &tmp = graph.CreateValue(rhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << rhs;
  tmp.set_label(ss.str());
  // Store the pointer
  rhs_ = tmp.get_shared_ptr();
}

Div::Div(const double &lhs, std::shared_ptr<Value> rhs) : Op(rhs), rhs_(rhs) {
  // Create the lhs in the graph
  auto &tmp = graph.CreateValue(lhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << lhs;
  tmp.set_label(ss.str());
  // Store the pointer
  lhs_ = tmp.get_shared_ptr();
}

Value &Div::Forward() {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto pow_op = std::make_shared<Pow>(rhs_, -1.0);
  auto &tmp = pow_op->Forward();
  auto tmpPtr = tmp.get_shared_ptr();
  tmpPtr->set_backward(std::bind(&Pow::Backward, pow_op));
  tmpPtr->AddProducer(rhs_);
  std::stringstream ss;
  ss << "div_tmp_id_" << tmpPtr->get_id();
  tmpPtr->set_label(ss.str());

  auto mul_op = std::make_shared<Mul>(lhs_, tmpPtr);
  auto &out = mul_op->Forward();
  out.set_backward(std::bind(&Mul::Backward, mul_op));
  out.AddProducer(lhs_);
  out.AddProducer(tmpPtr);
  // Reset the sstream
  ss.str("");
  ss.clear();
  ss << "tmp_div_out_id_" << out.get_id();
  out.set_label(ss.str());
  out_ = out.get_shared_ptr();
  return out;
}

void Div::Backward() { out_->Backward(); }
