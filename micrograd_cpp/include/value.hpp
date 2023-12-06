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

// Forward declarations
class Value;

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
  friend Value pow(Value &a, const float &n);
  friend Value operator+(const float &lhs, Value &rhs);
  friend Value operator+(Value &lhs, const float &rhs);
  friend Value operator-(const float &lhs, Value &rhs);
  friend Value operator-(Value &lhs, const float &rhs);
  friend Value operator*(const float &lhs, Value &rhs);
  friend Value operator*(Value &lhs, const float &rhs);
  friend Value operator/(const float &lhs, Value &rhs);
  friend Value operator/(Value &lhs, const float &rhs);

  // Constructors
  Value(const double &data, const std::string &label);
  // data and op are copied to the new object even though they are passed as a
  // reference. We capture children as rvalue in order to save one copy call,
  // see implementation for details.
  Value(const double &data, std::set<Value *> &&children,
        const std::string &op);
  // Copy constructor
  // Value(const Value &value);

  // FIXME: Verify this
  // We have dynamically allocated Values stored in dynamic_values
  // these values are automatically freed when vector goes out of scope

  // Notice that both the grad of this and rhs is being altered by this
  Value operator+(Value &rhs);  // NOLINT
  Value operator*(Value &rhs);  // NOLINT
  Value operator/(Value &rhs);  // NOLINT
  Value operator-();

  // Accessors and mutators (get and set functions) may be named like variables.
  // These returns a copy as we do not want anything other than the class to
  // modify the value of these
  const std::set<Value *> &get_children() const;
  const std::string &get_op() const;
  int get_id() const;
  void set_label(const std::string &label);
  void set_grad(const double &grad);
  Value tanh();
  Value exp();

  void Backward();

 private:
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
