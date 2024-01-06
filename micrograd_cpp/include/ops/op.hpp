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
  Op(std::shared_ptr<Value> val);
  // FIXME: This is to avoid destructor called on non-final 'Tanh' that has
  // virtual functions but non-virtual destructor, but I'm not sure I understand
  // it
  virtual ~Op() = default;

  /* FIXME:
  // This is not needed, I just want to be explicit
  ~Op();
  */
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_OP_HPP_
