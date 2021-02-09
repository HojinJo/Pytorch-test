#ifndef NET_H
#define NET_H
#include<vector>
#include <torch/torch.h>

//extern "C"{

typedef struct _net
{
	std::vector<torch::jit::Module> child;
	std::vector<torch::jit::IValue> inputs;
	std::vector<int> basicblock; //resnet
	int j; //resnet
	at::Tensor output;
	at::Tensor identity;
	int index_n;
}Net;
typedef struct _netlayer
{
	Net *net;
	std::vector<int> add_identity; //resnet
	//at::Tensor identity;
	int index; //layer index
}netlayer;

//}
#endif
