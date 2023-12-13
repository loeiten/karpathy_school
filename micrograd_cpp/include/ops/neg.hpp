#ifndef MICROGRAD_CPP_INCLUDE_OPS_NEG_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_NEG_HPP_

#include <memory>
#include "op.hpp"

class Value;

class Neg: private Op{
  public:
    Neg(std::shared_ptr<Value> val);
    Value &Forward()  final ;
    void Backward() final ;
  private:
    std::shared_ptr<Value> val_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_NEG_HPP_
