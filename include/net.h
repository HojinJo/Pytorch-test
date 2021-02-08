#ifndef NET_H
#define NET_H
#include<vector>
#include <torch/torch.h>

//extern "C"{

typedef struct _net
{
	std::vector<torch::jit::Module> child;
	std::vector<torch::jit::IValue> inputs;
	std::vector<int> basicblock;
	at::Tensor output;
	int index_n;
}Net;
typedef struct _netlayer
{
	Net *net;
	std::vector<int> add_identity; //resnet
	int j; //resnet
	int index; //layer index
}netlayer;

//}
#endif
