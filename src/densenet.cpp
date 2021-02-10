#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "densenet.h"

using namespace std;
namespace F = torch::nn::functional;

void get_submodule_densenet(torch::jit::script::Module module, std::vector<torch::jit::Module> &child, vector<pair<int, int>> &denseblock){ 
        if(module.children().size() == 0){
                child.push_back(module);
                return;
        }
        for(auto children : module.named_children()){
                if(children.name.find("denseblock") != std::string::npos){
                    int num_layer = 0;
                    for(auto layer : children.value.named_children()){
                        if(layer.name.find("denselayer") != std::string::npos)
                            num_layer++;
                    }
                    denseblock.push_back(make_pair(child.size(), num_layer));
                }
                get_submodule_densenet(children.value, child, denseblock);
        }
}

at::Tensor vector_cat(vector<torch::jit::IValue> inputs){
    at::Tensor out = inputs[0].toTensor();
    for(int i=1;i<inputs.size();i++){
        out = torch::cat({out, inputs[i].toTensor()}, 1);
    }
    return out;
}
at::Tensor denselayer_forward(vector<torch::jit::Module> module_list, vector<torch::jit::IValue> inputs, int idx){
    at::Tensor concated_features = vector_cat(inputs);

    for(int i=0;i<6;i++){
        inputs.clear();
        inputs.push_back(concated_features);
        concated_features = module_list[idx+i].forward(inputs).toTensor();
    }
    return concated_features;
}

at::Tensor denseblock_forward(vector<torch::jit::Module> module_list, vector<torch::jit::IValue> inputs, int idx, int num_layer)
{
    at::Tensor out;
    for(int i=0;i<num_layer;i++){
        out = denselayer_forward(module_list, inputs, idx);
        inputs.push_back(out);
        idx += 6;
    }

    return vector_cat(inputs);
}

void *predict_densenet(Net *input){
    std::vector<torch::jit::Module> child = input->child;
	std::vector<torch::jit::IValue> inputs = input->inputs;
	std::cout<<child.size()<<"\n";
    for(int i=0;i<child.size();i++){
        std::cout<< "dense layer " << i <<"\n";
        pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1; //right?
        netlayer nl;
		nl.net = input;
        nl.net->index = i;

		th_arg th;
		th.arg = &nl;
        std::cout << "Before thpool add work DENSE " << i << "\n";
        thpool_add_work(thpool,(void(*)(void *))forward_densenet,&th);
        std::cout << "After thpool add work DENSE " << i << "\n";
        while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
        i = nl.net->index;
		input->inputs.clear();
		input->inputs.push_back(input->output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
    }
    std::cout << "\n*****Dense result*****" << "\n";
	std::cout << (input->output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
}

void forward_densenet(th_arg *th){ //vector<torch::jit::Module> &child, vector<torch::jit::IValue> inputs, vector<pair<int, int>> denseblock){
    pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
    std::vector<torch::jit::Module> child = nl->net->child;
	std::vector<torch::jit::IValue> inputs = nl->net->inputs;
    std::vector<pair<int,int>> denseblock = nl->net->block;
    at::Tensor out = nl->net->output;

    int k =nl->net->index;
    int j = 0;
	if(k != 0)
		j = nl->net->j;
    
    if(k == 485){
        out = F::relu(inputs[0].toTensor(), F::ReLUFuncOptions().inplace(true));
        out = F::adaptive_avg_pool2d(out, F::AdaptiveAvgPool2dFuncOptions(1));
        out = out.view({out.size(0), -1});
	    inputs.clear();
	    inputs.push_back(out);
        out = child[k].forward(inputs).toTensor();
    }
    else if(j < denseblock.size() && k == denseblock[j].first){
        out = denseblock_forward(child, inputs, k, denseblock[j].second);
        k += denseblock[j].second*6 - 1;
        j += 1;
    }
    else{
        out = child[k].forward(inputs).toTensor();
    }
    nl->net->output = out;
    nl->net->j = j;
    nl->net->index = k;
	cond_i[nl->net->index_n]=0;
    std::cout<< "dense layer " << k <<"end\n";
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
}