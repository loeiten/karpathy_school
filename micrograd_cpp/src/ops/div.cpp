#include "../../include/ops/div.hpp"

#include <memory>
#include <sstream>
#include <iomanip>  // for operator<<, setprecision

#include "../../include/graph.hpp"
#include "../../include/ops/mul.hpp"
#include "../../include/ops/pow.hpp"
#include "../../include/value.hpp"

Div::Div(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
    : Op(rhs), rhs_(rhs), lhs_(lhs) {}

Div::Div(std::shared_ptr<Value> lhs, const double &rhs) : Op(lhs), lhs_(lhs) {
  // Create the rhs in the graph
  auto& tmp = graph.CreateValue(rhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << rhs;
  tmp.set_label(ss.str());
  // Store the pointer
  rhs_ = tmp.get_shared_ptr();
}

Div::Div(const double &lhs, std::shared_ptr<Value> rhs) : Op(rhs), rhs_(rhs) {
  // Create the lhs in the graph
  auto& tmp = graph.CreateValue(lhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << lhs;
  tmp.set_label(ss.str());
  // Store the pointer
  lhs_ = tmp.get_shared_ptr();
}

Value &Div::Forward() {
  // FIXME: Should I have no producers?
  // FIXME:
  std::cout << "Div FWD:" << std::endl;
  std::cout << "  rhs_: id " << rhs_->get_id() << " | " << *rhs_ << std::endl;
  std::cout << "  lhs_: id " << lhs_->get_id() << " | " << *lhs_ << std::endl;
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto pow_op = std::make_shared<Pow>(rhs_, -1.0);
  auto &tmp = pow_op->Forward();
  auto tmpPtr = tmp.get_shared_ptr();
  tmpPtr->set_backward(std::bind(&Pow::Backward, pow_op));
  // FIXME:
  tmpPtr->AddProducer(rhs_);
  auto mul_op = std::make_shared<Mul>(lhs_, tmpPtr);
  auto &out = mul_op->Forward();
  out.set_backward(std::bind(&Mul::Backward, mul_op));
  // FIXME:
  out.AddProducer(lhs_);
  out.AddProducer(tmpPtr);
  // FIXME:
  std::cout << "  out: id " << out.get_id() << " | " << out << std::endl;
  // FIXME:
  out_ = out.get_shared_ptr();
  return out;
}

void Div::Backward() {
  std::cout << "        Div BWD (crickets chirping)" << std::endl;
  // NOTE: The back-propagation is controlled by the the other backward passes
  // FIXME: Test:
  //out_->Backward();
}