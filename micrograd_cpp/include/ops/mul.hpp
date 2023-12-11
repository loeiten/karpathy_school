#ifndef MICROGRAD_CPP_INCLUDE_OPS_MUL_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_MUL_HPP_

#include <memory>
#include "op.hpp"

class Value;

class Mul: private Op{
    Mul(std::shared_ptr<Value> rhs, std::shared_ptr<Value> lhs);
    Mul(float &rhs, std::shared_ptr<Value> lhs);
    Mul(std::shared_ptr<Value> rhs, float lhs);
    void Forward()  final ;
    void Backward() final ;
  private:
    std::shared_ptr<Value> rhs_;
    std::shared_ptr<Value> lhs_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_MUL_HPP_
