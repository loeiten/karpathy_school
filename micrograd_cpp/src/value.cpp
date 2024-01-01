#include "../include/value.hpp"

#include <memory.h>

#include <cmath>
#include <functional>
#include <iomanip>  // for operator<<, setprecision
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>  // for unordered_set

#include "../include/ops/add.hpp"
#include "../include/ops/cos.hpp"
#include "../include/ops/div.hpp"
#include "../include/ops/exp.hpp"
#include "../include/ops/mul.hpp"
#include "../include/ops/neg.hpp"
#include "../include/ops/pow.hpp"
#include "../include/ops/sub.hpp"
#include "../include/ops/tanh.hpp"

int Value::instance_count = 0;

// Friend functions
// =============================================================================
std::ostream &operator<<(std::ostream &os, const Value &value) {
  os << std::fixed << std::setprecision(2) << value.label_ << " | data "
     << value.data_ << " | grad " << value.grad_;
  return os;
}

Value &pow(Value &a, const double &n) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  // FIXME: Can we make this a unique pointer instead? Are we copying it
  // anywhere?
  auto pow_op = std::make_shared<Pow>(a.get_shared_ptr(), n);
  auto &out = pow_op->Forward();
  out.Backward_ = std::bind(&Pow::Backward, pow_op);
  return out;
}

Value &operator+(const double &lhs, Value &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto add_op = std::make_shared<Add>(lhs, rhs.get_shared_ptr());
  auto &out = add_op->Forward();
  out.Backward_ = std::bind(&Add::Backward, add_op);
  return out;
}

Value &operator+(Value &lhs, const double &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto add_op = std::make_shared<Add>(lhs.get_shared_ptr(), rhs);
  auto &out = add_op->Forward();
  out.Backward_ = std::bind(&Add::Backward, add_op);
  return out;
}

Value &operator-(const double &lhs, Value &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto sub_op = std::make_shared<Sub>(lhs, rhs.get_shared_ptr());
  auto &out = sub_op->Forward();
  out.Backward_ = std::bind(&Sub::Backward, sub_op);
  return out;
}

Value &operator-(Value &lhs, const double &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto sub_op = std::make_shared<Sub>(lhs.get_shared_ptr(), rhs);
  auto &out = sub_op->Forward();
  out.Backward_ = std::bind(&Sub::Backward, sub_op);
  return out;
}

Value &operator*(const double &lhs, Value &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto mul_op = std::make_shared<Mul>(lhs, rhs.get_shared_ptr());
  auto &out = mul_op->Forward();
  out.Backward_ = std::bind(&Mul::Backward, mul_op);
  return out;
}

Value &operator*(Value &lhs, const double &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto mul_op = std::make_shared<Mul>(lhs.get_shared_ptr(), rhs);
  auto &out = mul_op->Forward();
  out.Backward_ = std::bind(&Mul::Backward, mul_op);
  return out;
}

Value &operator/(const double &lhs, Value &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto div_op = std::make_shared<Div>(lhs, rhs.get_shared_ptr());
  auto &out = div_op->Forward();
  out.Backward_ = std::bind(&Div::Backward, div_op);
  return out;
}

Value &operator/(Value &lhs, const double &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto div_op = std::make_shared<Div>(lhs.get_shared_ptr(), rhs);
  auto &out = div_op->Forward();
  out.Backward_ = std::bind(&Div::Backward, div_op);
  return out;
}

Value &tanh(Value &value) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto tanh_op = std::make_shared<Tanh>(value.get_shared_ptr());
  auto &out = tanh_op->Forward();
  out.Backward_ = std::bind(&Tanh::Backward, tanh_op);
  return out;
}

Value &cos(Value &value) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto cos_op = std::make_shared<Cos>(value.get_shared_ptr());
  auto &out = cos_op->Forward();
  out.Backward_ = std::bind(&Cos::Backward, cos_op);
  return out;
}

Value &exp(Value &value) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto exp_op = std::make_shared<Exp>(value.get_shared_ptr());
  auto &out = exp_op->Forward();
  out.Backward_ = std::bind(&Exp::Backward, exp_op);
  return out;
}
// =============================================================================

// Constructors
// =============================================================================
Value::Value(Graph &graph, const double &data)
    : graph_(graph), data_(data), grad_(0) {
  ++instance_count;
  id_ = instance_count;
}

Value::Value(Graph &graph, const double &data, const std::string &label)
    : graph_(graph), data_(data), grad_(0), label_(label) {
  ++instance_count;
  id_ = instance_count;
}
// =============================================================================

// Member functions: Operator overloads
// =============================================================================
Value &Value::operator+(Value &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto add_op = std::make_shared<Add>(get_shared_ptr(), rhs.get_shared_ptr());
  auto &out = add_op->Forward();
  out.Backward_ = std::bind(&Add::Backward, add_op);
  return out;
}

Value &Value::operator*(Value &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto mul_op = std::make_shared<Mul>(get_shared_ptr(), rhs.get_shared_ptr());
  auto &out = mul_op->Forward();
  out.Backward_ = std::bind(&Mul::Backward, mul_op);
  return out;
}

Value &Value::operator/(Value &rhs) {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto div_op = std::make_shared<Div>(get_shared_ptr(), rhs.get_shared_ptr());
  auto &out = div_op->Forward();
  out.Backward_ = std::bind(&Div::Backward, div_op);
  return out;
}

Value &Value::operator-() {
  // NOTE: We need to dynamically allocate the op for it to be in scope when
  //       out.Backward_ is called
  auto neg_op = std::make_shared<Neg>(get_shared_ptr());
  auto &out = neg_op->Forward();
  out.Backward_ = std::bind(&Neg::Backward, neg_op);
  return out;
}
// =============================================================================

// Member functions: Accessors and mutators
// =============================================================================
const std::set<std::shared_ptr<Value>> &Value::get_producers() const {
  return producers;
}

const std::string &Value::get_op() const { return op_; }

const double &Value::get_data() const { return data_; }

const double &Value::get_grad() const { return grad_; }

int Value::get_id() const { return id_; }

Graph &Value::get_graph() const { return graph_; }

std::shared_ptr<Value> Value::get_shared_ptr() const {
  return value_shared_ptr_;
}

void Value::set_shared_ptr(const std::shared_ptr<Value> &value_shared_ptr) {
  value_shared_ptr_ = value_shared_ptr;
}

void Value::set_label(const std::string &label) { label_ = label; }

void Value::set_grad(const double &grad) { grad_ = grad; }

void Value::set_op(const std::string &op) { op_ = op; }
// =============================================================================

// Member functions: Other
// =============================================================================
void Value::AddProducer(std::shared_ptr<Value> producer) {
  producers.insert(producer);
}

void Value::UpdateGrad(const double &grad) { grad_ += grad; }
// =============================================================================

// FIXME: Move to graph?
void Value::Backward() {
  // Build the topology
  TopologicalSort(*this);
  this->grad_ = 1.0;

  for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
    if ((*it)->Backward_ != nullptr) {
      (*it)->Backward_();
    }
  }
}

void Value::TopologicalSort(const Value &value) {
  if (visited.find(value.get_id()) == visited.end()) {
    visited.insert(value.get_id());
    for (const auto &child : value.producers) {
      TopologicalSort(*child);
    }
    topology.push_back(value.get_shared_ptr());
  }
}
