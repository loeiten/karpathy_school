#ifndef MICROGRAD_CPP_INCLUDE_OPS_ADD_HPP_
#define MICROGRAD_CPP_INCLUDE_OPS_ADD_HPP_

#include <memory>
#include "op.hpp"

class Value;

class Add: private Op{
    Add(std::shared_ptr<Value> rhs, std::shared_ptr<Value> lhs);
    Add(const float &rhs, std::shared_ptr<Value> lhs);
    Add(std::shared_ptr<Value> rhs, const float &lhs);
    Value& Forward() final ;
    void Backward() final ;
  private:
    std::shared_ptr<Value> rhs_;
    std::shared_ptr<Value> lhs_;
};

#endif  // MICROGRAD_CPP_INCLUDE_OPS_ADD_HPP_
