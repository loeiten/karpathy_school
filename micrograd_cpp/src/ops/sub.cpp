#include "../../include/ops/sub.hpp"

#include <iomanip>  // for operator<<, setprecision
#include <memory>
#include <sstream>

#include "../../include/graph.hpp"
#include "../../include/ops/add.hpp"
#include "../../include/ops/neg.hpp"
#include "../../include/value.hpp"

Sub::Sub(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
    : Op(rhs), rhs_(rhs), lhs_(lhs) {}

Sub::Sub(std::shared_ptr<Value> lhs, const double &rhs) : Op(lhs), lhs_(lhs) {
  // Create the rhs in the graph
  auto &tmp = graph.CreateValue(rhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << rhs;
  tmp.set_label(ss.str());
  // Store the pointer
  rhs_ = tmp.get_shared_ptr();
}

Sub::Sub(const double &lhs, std::shared_ptr<Value> rhs) : Op(rhs), rhs_(rhs) {
  // Create the lhs in the graph
  auto &tmp = graph.CreateValue(lhs);
  // Add a label to the tmp
  std::stringstream ss;
  ss << "literal " << std::fixed << std::setprecision(2) << lhs;
  tmp.set_label(ss.str());
  // Store the pointer
  lhs_ = tmp.get_shared_ptr();
}

Value &Sub::Forward() {
  auto neg_op = std::make_shared<Neg>(rhs_);
  auto &tmp = neg_op->Forward();
  auto tmpPtr = tmp.get_shared_ptr();
  // NOTE: We don't set backward as this is done in the op
  tmpPtr->AddProducer(rhs_);
  std::stringstream ss;
  ss << "neg_tmp_id_" << tmpPtr->get_id();
  tmpPtr->set_label(ss.str());

  auto add_op = std::make_shared<Add>(lhs_, tmpPtr);
  auto &out = add_op->Forward();
  out.set_backward(std::bind(&Add::Backward, add_op));
  out.AddProducer(lhs_);
  out.AddProducer(tmpPtr);
  // Reset the sstream
  ss.str("");
  ss.clear();
  ss << "sub_out_id_" << out.get_id();
  out.set_label(ss.str());
  out_ = out.get_shared_ptr();
  return out;
}

void Sub::Backward() { out_->Backward(); }