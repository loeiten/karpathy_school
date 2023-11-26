#ifndef MICROGRAD_CPP_INCLUDE_VALUE_HPP_
#define MICROGRAD_CPP_INCLUDE_VALUE_HPP_

#include <cstddef>        // for size_t
#include <cstdint>        // for uint64_t
#include <functional>     // for hash, equal_to
#include <memory>         // for shared_ptr, hash
#include <ostream>        // for ostream
#include <string>         // for string
#include <unordered_set>  // for unordered_set
#include <utility>        // for pair

class Value {
 public:
  // We define as a friend as this is not a class member function, but need
  // access to the private member.
  // Overloads for the streaming output operators (<<) cannot be class members,
  // because the ostream& must be on the left in use and declaration.
  friend std::ostream &operator<<(std::ostream &os, const Value &value);
  explicit Value(const float &data);
  // data and op are copied to the new object even though they are passed as a
  // reference. We capture children as rvalue in order to save one copy call,
  // see implementation for details.
  Value(const float &data,
        std::unordered_set<std::shared_ptr<Value>> &&children,
        const std::string &op);

  Value operator+(const Value &rhs) const;
  Value operator*(const Value &rhs) const;

  // Accessors and mutators (get and set functions) may be named like variables.
  // These returns a copy as we do not want anything other than the class to
  // modify the value of these
  std::unordered_set<std::shared_ptr<Value>> get_children() const;
  std::string get_op() const;
  int get_id() const;

 private:
  float data_;
  std::unordered_set<std::shared_ptr<Value>> prev_;
  std::string op_ = "";
  int id_;
  static int instance_count;
};

// Declare these functions here, so that other files can use it
namespace std {
// We need a has function in order to use unordered_set
template <>  // template<> is used to specialize a template for a specific type
struct hash<Value> {
  size_t operator()(const Value &value) const {
    return std::hash<int>()(value.get_id());
  }
};

// We need equal_to in order to use .find() on the unordered_set
template <>
struct equal_to<Value> {
  bool operator()(const Value &lhs, const Value &rhs) const {
    return lhs.get_id() == rhs.get_id();
  }
};

// We need a hash to use the pair of Values in an unordered_set
template <>
struct hash<std::pair<Value, Value>> {
  size_t operator()(const std::pair<Value, Value> &p) const {
    // Compute a hash value for the pair using FNV-1a
    // Note that SipHash is more sophisticated and has replaced this method in
    // python
    uint64_t hash = 14695981039346656037ull;
    hash = (hash ^ static_cast<uint64_t>(p.first.get_id())) * 1099511628211ull;
    hash = (hash ^ static_cast<uint64_t>(p.second.get_id())) * 1099511628211ull;
    return hash;
  }
};

// Finally, since unordered_set checks for equality we need to check for this as
// well
template <>
struct equal_to<std::pair<Value, Value>> {
  bool operator()(const std::pair<Value, Value> &lhs,
                  const std::pair<Value, Value> &rhs) const {
    return (lhs.first.get_id() == rhs.first.get_id()) &&
           (lhs.second.get_id() == rhs.second.get_id());
  }
};
}  // namespace std

#endif  // MICROGRAD_CPP_INCLUDE_VALUE_HPP_
