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
#include <iostream>         // for vector

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
struct hash<const std::shared_ptr<Value>> {
  size_t operator()(const std::shared_ptr<Value> value) const;
};
// We need equal_to in order to use .find() on the unordered_set
template <>
struct equal_to<const std::shared_ptr<Value>> {
  bool operator()(const std::shared_ptr<Value> lhs,
                  const std::shared_ptr<Value> rhs) const;
};
}  // namespace std

class Value {
 public:
  // We define as a friend as this is not a class member function, but need
  // access to the private member.
  // Overloads for the streaming output operators (<<) cannot be class members,
  // because the ostream& must be on the left in use and declaration.
  friend std::ostream &operator<<(std::ostream &os, const Value &value);
  friend Value &pow(Value &a, const double &n);
  friend Value &operator+(const double &lhs, Value &rhs);
  friend Value &operator+(Value &lhs, const double &rhs);
  friend Value &operator-(const double &lhs, Value &rhs);
  friend Value &operator-(Value &lhs, const double &rhs);
  friend Value &operator*(const double &lhs, Value &rhs);
  friend Value &operator*(Value &lhs, const double &rhs);
  friend Value &operator/(const double &lhs, Value &rhs);
  friend Value &operator/(Value &lhs, const double &rhs);
  friend Value &tanh(Value &value);
  friend Value &exp(Value &value);
  friend Value &cos(Value &value);

  // Constructors
  Value(Graph &graph, const double &data);
  Value(Graph &graph, const double &data, const std::string &label);

  // Notice that both the grad of this and rhs is being altered by this
  Value &operator+(Value &rhs);
  Value &operator*(Value &rhs);
  Value &operator/(Value &rhs);
  Value &operator-();

  // Accessors and mutators (get and set functions) may be named like variables.
  // These returns a copy as we do not want anything other than the class to
  // modify the value of these
  const std::set<std::shared_ptr<Value>> &get_producers() const;
  const std::string &get_label() const;
  const std::string &get_op() const;
  const double &get_data() const;
  const double &get_grad() const;
  int get_id() const;
  Graph &get_graph() const;
  std::shared_ptr<Value> get_shared_ptr() const;
  void set_shared_ptr(const std::shared_ptr<Value> &value_shared_ptr);
  void set_label(const std::string &label);
  void set_grad(const double &grad);
  void set_op(const std::string &op);
  void set_backward(const std::function<void()> &func);

  void AddProducer(std::shared_ptr<Value> producer);
  void UpdateGrad(const double &grad);

  void Backward();

 private:
  std::shared_ptr<Value> value_shared_ptr_;
  // FIXME: Consider to use pImpl
  Graph &graph_;
  double data_;
  double grad_ = 0;
  // We do care about the order of the producers for printing purposes
  std::set<std::shared_ptr<Value>> producers;  // Aka prev_ aka children
  std::unordered_set<int> visited;
  std::vector<const std::shared_ptr<Value>> topology;
  int id_;
  std::string label_ = "";
  std::string op_ = "";
  static int instance_count;

  std::function<void()> Backward_ = nullptr;
  void TopologicalSort(const Value &value);
};

// Define these functions here, so that other files can use it
// We need to use inline so that only one copy is used
inline std::size_t std::hash<const std::shared_ptr<Value>>::operator()(
    const std::shared_ptr<Value> value) const {
  return std::hash<int>()(value->get_id());
}

inline bool std::equal_to<const std::shared_ptr<Value>>::operator()(
    const std::shared_ptr<Value> lhs, const std::shared_ptr<Value> rhs) const {
  return lhs->get_id() == rhs->get_id();
}

namespace std {
// We need a hash to use the pair of Values in an unordered_set
template <>
struct hash<pair<const std::shared_ptr<Value>, const std::shared_ptr<Value>>> {
  size_t operator()(const pair<const std::shared_ptr<Value>,
                               const std::shared_ptr<Value>> &p) const {
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
struct equal_to<pair<std::shared_ptr<Value>, std::shared_ptr<Value>>> {
  bool operator()(
      const pair<std::shared_ptr<Value>, std::shared_ptr<Value>> &lhs,
      const pair<std::shared_ptr<Value>, std::shared_ptr<Value>> &rhs) const {
    return (lhs.first->get_id() == rhs.first->get_id()) &&
           (lhs.second->get_id() == rhs.second->get_id());
  }
};
}  // namespace std

#endif  // MICROGRAD_CPP_INCLUDE_VALUE_HPP_
