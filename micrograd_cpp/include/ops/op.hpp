#ifndef MICROGRAD_CPP_INCLUDE_OPS_OP_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_OP_HPP_

#include <memory>  // for shared_ptr

class Value;
class Graph;

class Op {
 public:
  // We don't need to delete these, I just want to be explicit
  Op() = delete;
  Op(const Op&) = delete;
  Op(const Op&&) = delete;
  virtual Value& Forward() = 0;
  virtual void Backward() = 0;

 protected:
  Graph& graph;
  explicit Op(std::shared_ptr<Value> val);
  // If the base class does not have a virtual destructor, the destructor of the
  // derived class will not be called, which can lead to memory leaks or
  // undefined behavior
  virtual ~Op() = default;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_OP_HPP_
