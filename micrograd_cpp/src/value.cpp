#include "../include/value.hpp"

#include <memory.h>

#include <cmath>
#include <iomanip>  // for operator<<, setprecision
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>  // for unordered_set

int Value::instance_count = 0;

std::ostream &operator<<(std::ostream &os, const Value &value) {
  os << std::fixed << std::setprecision(2) << value.label_ << " | data "
     << value.data_ << " | grad " << value.grad_;
  return os;
}

Value pow(Value &a, const float &n) {
  std::set<Value *> children;
  children.insert(&a);
  // FIXME: Verify this
  // We are moving the children, which is a set of pointer
  // Hence, we are not moving the members themselves
  auto out = Value(std::pow(a.data_, n), std::move(children), "^");

  // FIXME:
  std::cout << "Inside pow: out.label_=" << out.label_ << std::endl;

  out.Backward_ = [n, &a, &out]() {
    out.grad_ += n * a.data_ * std::pow(a.data_, n - 1) * out.grad_;
  };
  return out;
}

Value operator+(const float &lhs, Value &rhs) {
  auto tmp = std::make_unique<Value>(lhs, "literal " + std::to_string(lhs));
  auto out = (*tmp) + rhs;
  out.dynamic_values.push_back(std::move(tmp));
  return out;
}

Value operator+(Value &lhs, const float &rhs) {
  auto tmp = std::make_unique<Value>(rhs, "literal " + std::to_string(rhs));
  auto out = lhs + (*tmp);
  out.dynamic_values.push_back(std::move(tmp));
  return out;
}

Value operator-(const float &lhs, Value &rhs) {
  // operator-() will call
  // operator*(Value &lhs, const float &rhs)
  // This will create a tmp for -1.0f multiply it with (*this)
  // (*this) in this context is the reference of rhs
  // the out resulting from (*this) * (-1.0f) should also be
  // outputted as named return value optimization (copy elision)
  // However tmp does go out of scope
  // Hence we copy the object to a smart pointer
  auto tmp = (-rhs);
  // FIXME: Why does move work here? I didn't specify a move constructor
  auto tmpPtr = std::make_unique<Value>(std::move(tmp));
  auto out = lhs + tmp;
  out.dynamic_values.push_back(std::move(tmpPtr));
  return out;
}

Value operator-(Value &lhs, const float &rhs) {
  auto tmp = std::make_unique<Value>(-rhs, "literal -" + std::to_string(rhs));
  auto out = lhs + (*tmp);
  out.dynamic_values.push_back(std::move(tmp));
  return out;
}

Value operator*(const float &lhs, Value &rhs) {
  auto tmp = std::make_unique<Value>(lhs, "literal " + std::to_string(lhs));
  auto out = (*tmp) * rhs;
  out.dynamic_values.push_back(std::move(tmp));
  return out;
}

Value operator*(Value &lhs, const float &rhs) {
  auto tmp = std::make_unique<Value>(rhs, "literal " + std::to_string(rhs));
  auto out = lhs * (*tmp);
  return out;
}

Value operator/(const float &lhs, Value &rhs) {
  auto tmp = pow(rhs, -1.0f);
  auto tmpPtr = std::make_unique<Value>(std::move(tmp));
  auto out = lhs * tmp;
  out.dynamic_values.push_back(std::move(tmpPtr));
  return out;
}

Value operator/(Value &lhs, const float &rhs) {
  // Here there will be a multiplication of a literal and the
  // operator*(Value &lhs, const float &rhs)
  // will create the temporary
  auto out = lhs * std::pow(rhs, -1.0f);
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
  label_ = "tmp" + std::to_string(id_);
}

// FIXME: I only have a copy constructor due to
// auto tmpPtr = std::make_unique<Value>(tmp);
// How do I now implement the rule of 3?
Value::Value(const Value &value)
    : data_(value.data_), grad_(value.grad_), prev_(value.prev_) {
  // FIXME:
  std::cout << "I'm inside the copy ctor" << std::endl;
  ++instance_count;
  id_ = instance_count;
  label_ = "tmp" + std::to_string(id_);
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
  // We create the temporary object...
  auto tmp = pow(rhs, -1.0f);
  // ...we copy it to a dynamically allocated memory...
  // FIXME:
  std::cout << "Before the make_unique<Value>(tmp)" << std::endl;
  std::cout << "tmp.label_ = " << tmp.label_ << std::endl;
  std::cout << "children of tmp " << std::endl;
  for (const auto &child : tmp.prev_) {
    std::cout << "  label_=" << child->label_ << std::endl;
  }
  auto tmpPtr = std::make_unique<Value>(tmp);
  std::cout << "After the make_unique<Value>(tmp)" << std::endl;
  std::cout << "tmp.label_ = " << tmp.label_ << std::endl;
  std::cout << "tmpPtr->label_ = " << tmpPtr->label_ << std::endl;
  std::cout << "children of tmpPtr " << std::endl;
  for (const auto &child : tmpPtr->prev_) {
    std::cout << "  label_=" << child->label_ << std::endl;
  }
  std::cout << std::endl;
  // FIXME: You are here: Maybe this is where the funk happens?
  auto out = (*this) * tmp;
  // ...then we change ownership of that memory to out
  out.dynamic_values.push_back(std::move(tmpPtr));
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
  std::cout << "Enter toposort from " << value.label_ << std::endl;
  if (visited.find(&value) == visited.end()) {
    visited.insert(&value);
    // FIXME:
    std::cout << "  " << value.label_ << " not visited" << std::endl;
    for (const auto &child : value.prev_) {
      std::cout << "    child label: " << child->label_ << std::endl;
      std::cout << "    child data: " << child->data_ << std::endl;
      TopologicalSort(*child);
    }
    topology.push_back(&value);
  }
}
