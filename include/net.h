#ifndef NET_H
#define NET_H

#include <vector>
#include <torch/torch.h>
#include <functional>

typedef struct _layer
{
	//std::vector<torch::jit::IValue> input;
	at::Tensor output;
}Layer;

typedef struct _net
{
	Layer *layer;
	std::vector<torch::jit::IValue> input;
	std::vector<torch::jit::Module> child;
	std::vector<std::pair<int, int>> block; // resnet,densenet
	int j; //resnet
	at::Tensor identity;
	int index; //layer index
	int index_n; //network
}Net;

typedef struct _netlayer
{
	Net *net;
}netlayer;

#endif
