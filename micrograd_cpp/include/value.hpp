#ifndef MICROGRAD_CPP_INCLUDE_VALUE_HPP_
#define MICROGRAD_CPP_INCLUDE_VALUE_HPP_

#include <cstddef>        // for size_t
#include <cstdint>        // for uint64_t
#include <functional>     // for equal_to, hash
#include <memory>         // for shared_ptr, hash
#include <ostream>        // for ostream
#include <string>         // for string
#include <unordered_set>  // for unordered_set
#include <utility>        // for pair
#include <vector>         // for vector

// Forward declarations
class Value;

namespace std {
// We need a hash function in order to use unordered_set
template <>  // template<> is used to specialize a template for a specific type
struct hash<shared_ptr<Value>> {
  size_t operator()(const shared_ptr<Value> &value) const;
};
// We need equal_to in order to use .find() on the unordered_set
template <>
struct equal_to<shared_ptr<Value>> {
  bool operator()(const shared_ptr<Value> &lhs,
                  const shared_ptr<Value> &rhs) const;
};
}  // namespace std

class Value {
 public:
  // We define as a friend as this is not a class member function, but need
  // access to the private member.
  // Overloads for the streaming output operators (<<) cannot be class members,
  // because the ostream& must be on the left in use and declaration.
  friend std::ostream &operator<<(std::ostream &os, const Value &value);

  // Constructors
  Value(const double &data, const std::string &label);
  // data and op are copied to the new object even though they are passed as a
  // reference. We capture children as rvalue in order to save one copy call,
  // see implementation for details.
  Value(const double &data,
        std::unordered_set<std::shared_ptr<Value>> &&children,
        const std::string &op);

  // Notice that both the grad of this and rhs is being altered by this
  Value operator+(Value &rhs);
  Value operator*(Value &rhs);

  // Accessors and mutators (get and set functions) may be named like variables.
  // These returns a copy as we do not want anything other than the class to
  // modify the value of these
  const std::unordered_set<std::shared_ptr<Value>> &get_children() const;
  const std::string &get_op() const;
  int get_id() const;
  void set_label(const std::string &label);
  void set_grad(const double &grad);
  Value tanh();

  void Backward();

  std::function<void()> Backward_;

 private:
  double data_;
  double grad_;
  std::unordered_set<std::shared_ptr<Value>> prev_;
  std::unordered_set<const Value *> visited;
  std::vector<const Value *> topo;
  int id_;
  std::string label_ = "";
  std::string op_ = "";
  static int instance_count;

  void BuildTopo(const Value &value);
};

// Define these functions here, so that other files can use it
// We need to use inline so that only one copy is used
inline size_t std::hash<std::shared_ptr<Value>>::operator()(
    const std::shared_ptr<Value> &value) const {
  return std::hash<int>()(value->get_id());
}

inline bool std::equal_to<std::shared_ptr<Value>>::operator()(
    const std::shared_ptr<Value> &lhs,
    const std::shared_ptr<Value> &rhs) const {
  return lhs->get_id() == rhs->get_id();
}

namespace std {
// We need a hash to use the pair of Values in an unordered_set
template <>
struct hash<pair<shared_ptr<Value>, shared_ptr<Value>>> {
  size_t operator()(const pair<shared_ptr<Value>, shared_ptr<Value>> &p) const {
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
struct equal_to<pair<shared_ptr<Value>, shared_ptr<Value>>> {
  bool operator()(const pair<shared_ptr<Value>, shared_ptr<Value>> &lhs,
                  const pair<shared_ptr<Value>, shared_ptr<Value>> &rhs) const {
    return (lhs.first->get_id() == rhs.first->get_id()) &&
           (lhs.second->get_id() == rhs.second->get_id());
  }
};
}  // namespace std

#endif  // MICROGRAD_CPP_INCLUDE_VALUE_HPP_
