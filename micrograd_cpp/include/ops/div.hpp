#ifndef MICROGRAD_CPP_INCLUDE_OPS_DIV_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_DIV_HPP_

#include <memory>
#include "op.hpp"

class Value;

class Div: private Op{
    Div(std::shared_ptr<Value> rhs, std::shared_ptr<Value> lhs);
    Div(float &rhs, std::shared_ptr<Value> lhs);
    Div(std::shared_ptr<Value> rhs, float lhs);
    void Forward()  final ;
    void Backward() final ;
  private:
    std::shared_ptr<Value> rhs_;
    std::shared_ptr<Value> lhs_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_DIV_HPP_
