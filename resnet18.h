#ifndef RESNET18_H
#define RESNET18_H
#include <iostream>
#include <vector>

void get_submodule_resnet18(torch::jit::script::Module module,std::vector<torch::jit::Module> *child, std::vector<int> *basicblock);
at::Tensor forward_resnet18(std::vector<torch::jit::Module> &child, std::vector<torch::jit::IValue> inputs);

#endif
