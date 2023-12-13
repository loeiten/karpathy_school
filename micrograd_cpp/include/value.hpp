#ifndef MICROGRAD_CPP_INCLUDE_VALUE_HPP_
#define MICROGRAD_CPP_INCLUDE_VALUE_HPP_

#include <cstddef>     // for size_t
#include <cstdint>     // for uint64_t
#include <functional>  // for equal_to, hash, function
#include <memory>
#include <ostream>        // for ostream
#include <set>            // for set
#include <string>         // for string
#include <unordered_set>  // for unordered_set
#include <utility>        // for pair
#include <vector>         // for vector

// FIXME:
// These issues needs to be addressed:
// 1. One cannot move a function lambda, only the captured values
// 2. Named return value optimization is optional, so one should not rely on
// this (https://en.cppreference.com/w/cpp/language/copy_elision)
// 3. Chaining

// Forward declarations
class Value;
class Graph;

namespace std {
// We need a hash function in order to use unordered_set
template <>  // template<> is used to specialize a template for a specific type
struct hash<const Value *> {
  size_t operator()(const Value *value) const;
};
// We need equal_to in order to use .find() on the unordered_set
template <>
struct equal_to<const Value *> {
  bool operator()(const Value *lhs, const Value *rhs) const;
};
}  // namespace std

class Value {
 public:
  // We define as a friend as this is not a class member function, but need
  // access to the private member.
  // Overloads for the streaming output operators (<<) cannot be class members,
  // because the ostream& must be on the left in use and declaration.
  friend std::ostream &operator<<(std::ostream &os, const Value &value);
  friend Value& pow(Value &a, const double &n);
  friend Value& operator+(const double &lhs, Value &rhs);
  friend Value& operator+(Value &lhs, const double &rhs);
  friend Value& operator-(const double &lhs, Value &rhs);
  friend Value& operator-(Value &lhs, const double &rhs);
  friend Value& operator*(const double &lhs, Value &rhs);
  friend Value& operator*(Value &lhs, const double &rhs);
  friend Value& operator/(const double &lhs, Value &rhs);
  friend Value& operator/(Value &lhs, const double &rhs);
  friend Value& tanh(Value &value);
  friend Value& exp(Value &value);

  // Constructors
  Value(const double &data, const std::string &label);
  // data and op are copied to the new object even though they are passed as a
  // reference. We capture children as rvalue in order to save one copy call,
  // see implementation for details.
  Value(const double &data, std::set<Value *> &&children,
        const std::string &op);
  // Copy constructor
  // FIXME: Move impl to .cpp
  // FIXME: Are the rule of five really needed?
  // FIXME: Is the problem that I'm using std::function? Would it be better to
  // use a plain pointer?
  Value(const Value &value)
      : data_(value.data_),
        grad_(value.grad_),
        prev_(value.prev_),
        visited(value.visited),
        topology(value.topology),
        id_(value.id_),
        label_(value.label_),
        op_(value.op_),
        Backward_(value.Backward_) {
    // As dynamic_values are unique pointers we need to deep copy them
    for (auto &dynamic_value : value.dynamic_values) {
      dynamic_values.emplace_back(std::make_unique<Value>(*dynamic_value));
    }
  }

  Value(Value &&value)
      : data_(value.data_),  // Not movable
        grad_(value.grad_),  // Not movable
        prev_(std::move(value.prev_)),
        visited(std::move(value.visited)),
        topology(std::move(value.topology)),
        id_(value.id_),  // Not movable
        label_(std::move(value.label_)),
        op_(std::move(value.op_)),
        Backward_(std::move(value.Backward_)) {
    // NOTE: We need to move each element individually
    // NOTE: We cannot have const since we want to move
    for (auto &dynamic_value : value.dynamic_values) {
      // FIXME: Push or emplace?
      // Doesn't matter if it's emplace_back or push_back
      dynamic_values.emplace_back(std::move(dynamic_value));
    }
  }

  // FIXME: Verify this
  // We have dynamically allocated Values stored in dynamic_values
  // these values are automatically freed when vector goes out of scope

  // FIXME: Rule of 5
  Value &operator=(const Value &value) {
    data_ = value.data_;
    grad_ = value.grad_;
    prev_ = value.prev_;
    visited = value.visited;
    topology = value.topology;
    id_ = value.id_;
    label_ = value.label_;
    op_ = value.op_;
    Backward_ = value.Backward_;

    // As dynamic_values are unique pointers we need to deep copy them
    // Clear before making deep copies
    dynamic_values.clear();
    for (auto &dynamic_value : value.dynamic_values) {
      dynamic_values.emplace_back(std::make_unique<Value>(*dynamic_value));
    }
    return *this;
  }

  Value &operator=(Value &&value) {
    data_ = value.data_;  // Not movable
    grad_ = value.grad_;  // Not movable
    prev_ = std::move(value.prev_);
    visited = std::move(value.visited);
    topology = std::move(value.topology);

    // NOTE: We need to move each element individually
    for (auto &dynamic_value : value.dynamic_values) {
      // FIXME: Push or emplace?
      dynamic_values.emplace_back(std::move(dynamic_value));
    }

    id_ = value.id_;  // Not movable
    label_ = std::move(value.label_);
    op_ = std::move(value.op_);
    Backward_ = std::move(value.Backward_);
    return *this;
  }

  // FIXME: Write dtor

  // Notice that both the grad of this and rhs is being altered by this
  Value& operator+(Value &rhs);  
  Value& operator*(Value &rhs);  
  Value& operator/(Value &rhs);  
  Value& operator-();

  // Accessors and mutators (get and set functions) may be named like variables.
  // These returns a copy as we do not want anything other than the class to
  // modify the value of these
  const std::set<Value *> &get_children() const;
  const std::string &get_op() const;
  const double &get_data() const;
  const double &get_grad() const;
  int get_id() const;
  Graph &get_graph() const;
  std::shared_ptr<Value> get_shared_ptr() const;
  void set_label(const std::string &label);
  void set_grad(const double &grad);
  void set_op(const std::string &op);

  void AddProducer(std::shared_ptr<Value> producer);
  void UpdateGrad(const double &grad);

  void Backward();

 private:
  // FIXME: All of this should be part of the pImpl
  double data_;
  double grad_ = 0;
  // We do care about the order of the children for printing purposes
  std::set<Value *> prev_;
  std::unordered_set<const Value *> visited;
  std::vector<const Value *> topology;
  std::vector<std::unique_ptr<Value>> dynamic_values;
  int id_;
  std::string label_ = "";
  std::string op_ = "";
  static int instance_count;

  std::function<void()> Backward_ = nullptr;
  void TopologicalSort(const Value &value);

  class ValueImpl;  // Forward declaration
  std::unique_ptr<ValueImpl> impl;
};

// Define these functions here, so that other files can use it
// We need to use inline so that only one copy is used
inline std::size_t std::hash<const Value *>::operator()(
    const Value *value) const {
  return std::hash<int>()(value->get_id());
}

inline bool std::equal_to<const Value *>::operator()(const Value *lhs,
                                                     const Value *rhs) const {
  return lhs->get_id() == rhs->get_id();
}

namespace std {
// We need a hash to use the pair of Values in an unordered_set
template <>
struct hash<pair<const Value *, const Value *>> {
  size_t operator()(const pair<const Value *, const Value *> &p) const {
    // Compute a hash value for the pair using FNV-1a
    // Note that SipHash is more sophisticated and has replaced this method in
    // python
    uint64_t hash = 14695981039346656037ull;
    hash ^= static_cast<uint64_t>(p.first->get_id());
    hash *= 1099511628211ull;
    hash ^= static_cast<uint64_t>(p.second->get_id());
    hash *= 1099511628211ull;
    return hash;
  }
};

// Finally, since unordered_set checks for equality we need to check for this as
// well
template <>
struct equal_to<pair<Value *, Value *>> {
  bool operator()(const pair<Value *, Value *> &lhs,
                  const pair<Value *, Value *> &rhs) const {
    return (lhs.first->get_id() == rhs.first->get_id()) &&
           (lhs.second->get_id() == rhs.second->get_id());
  }
};
}  // namespace std

#endif  // MICROGRAD_CPP_INCLUDE_VALUE_HPP_
