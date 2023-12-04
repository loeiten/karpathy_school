#include "../include/value.hpp"

#include <memory.h>

#include <cmath>
#include <iomanip>  // for operator<<, setprecision
#include <iostream>
#include <sstream>
#include <unordered_set>  // for unordered_set

int Value::instance_count = 0;

std::ostream &operator<<(std::ostream &os, const Value &value) {
  os << std::fixed << std::setprecision(2) << value.label_ << " | data "
     << value.data_ << " | grad " << value.grad_;
  return os;
}

Value pow(const Value &a, const float &n) {
  std::stringstream ss;
  ss << "^" << a.data_;
  auto out = Value(std::pow(a.data_, n), ss.str());

  out.Backward_ = [n, &a, &out]() {
    out.grad_ += n * a.data_ * std::pow(a.data_, n - 1) * out.grad_;
  };
  return out;
}

Value::Value(const double &data, const std::string &label)
    : data_(data), grad_(0), label_(label) {
  ++instance_count;
  id_ = instance_count;
}

Value::Value(const double &data, std::set<Value *> &&children,
             const std::string &op)
    : data_(data), grad_(0), prev_(children), op_(op) {
  ++instance_count;
  id_ = instance_count;
}

Value Value::operator+(Value &rhs) {  // NOLINT
  std::set<Value *> children;
  children.insert(this);
  children.insert(&rhs);
  // We move the children to the out object, no reason for copy as the variable
  // in this function will go out of scope anyways
  Value out = Value(this->data_ + rhs.data_, std::move(children), "+");

  // Finally, we define the backward function

  // In a lambda: The compiler will create a "template" of the function
  //              During runtime parameters of the function will be filled
  out.Backward_ = [this, &out, &rhs]() {
    this->grad_ += out.grad_;
    rhs.grad_ += out.grad_;
  };
  // NOTE: It looks like out goes out of scope and is copied
  //       However, this is not the case due to named return value optimization
  //       See
  //       https://artificial-mind.net/blog/2021/10/23/return-moves#:~:text=Returning%20a%20Local%20Variable&text=No%20temporary%20object%20is%20created,return%20value%20optimization%E2%80%9D%20or%20NRVO.
  //       for details
  return out;
}

Value Value::operator*(Value &rhs) {  // NOLINT
  std::set<Value *> children;
  children.insert(this);
  children.insert(&rhs);
  Value out(this->data_ * rhs.data_, std::move(children), "*");
  out.Backward_ = [this, &out, &rhs]() {
    this->grad_ += rhs.data_ * out.grad_;
    rhs.grad_ += this->data_ * out.grad_;
  };
  return out;
}

Value Value::operator/(Value &rhs) {  // NOLINT
  auto tmp = pow(rhs, -1.0f);
  auto out = (*this) * tmp;
  return out;
}

Value Value::operator-() {
  auto out = (*this) * (-1.0f);
  return out;
}

const std::set<Value *> &Value::get_children() const { return prev_; }

const std::string &Value::get_op() const { return op_; }

int Value::get_id() const { return id_; }

void Value::set_label(const std::string &label) { label_ = label; }

void Value::set_grad(const double &grad) { grad_ = grad; }

Value Value::tanh() {
  const double &x = data_;
  const double &t = (::exp(2 * x) - 1) / (::exp(2 * x) + 1);
  std::set<Value *> children;
  children.insert(this);
  Value out(t, std::move(children), "tanh");
  // We copy t as this goes out of scope
  out.Backward_ = [this, &out, t]() {
    this->grad_ += (1 - pow(t, 2)) * out.grad_;
  };
  return out;
}

Value Value::exp() {
  std::set<Value *> children;
  children.insert(this);
  Value out(::exp(data_), std::move(children), "exp");
  out.Backward_ = [this, &out]() { this->grad_ += out.data_ * out.grad_; };
  return out;
}

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
  // FIXME:
  std::cout << "Start toposort" << std::endl;
  if (visited.find(&value) == visited.end()) {
    visited.insert(&value);
    // FIXME:
    std::cout << "Not visited" << std::endl;
    for (const auto &child : value.prev_) {
      std::cout << "  data: " << child->data_ << std::endl;
      std::cout << "Label: " << child->label_ << std::endl;
      TopologicalSort(*child);
    }
    topology.push_back(&value);
  }
}

Value operator+(const float &lhs, Value &rhs) {
  auto tmp = Value(lhs, "");
  auto out = tmp + rhs;
  return out;
}

Value operator+(Value &lhs, const float &rhs) {
  auto tmp = Value(rhs, "");
  auto out = lhs + tmp;
  return out;
}

Value operator-(const float &lhs, Value &rhs) {
  auto tmp = (-rhs);
  auto out = lhs + tmp;
  return out;
}

Value operator-(Value &lhs, const float &rhs) {
  auto out = lhs + (-rhs);
  return out;
}

Value operator*(const float &lhs, Value &rhs) {
  auto tmp = Value(lhs, "");
  auto out = tmp * rhs;
  return out;
}

Value operator*(Value &lhs, const float &rhs) {
  auto tmp = Value(rhs, "");
  auto out = lhs * tmp;
  return out;
}

Value operator/(const float &lhs, Value &rhs) {
  auto tmp = pow(rhs, -1.0f);
  auto out = lhs * tmp;
  return out;
}

Value operator/(Value &lhs, const float &rhs) {
  auto out = lhs * std::pow(rhs, -1.0f);
  return out;
}
