#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "densenet.h"

using namespace std;
namespace F = torch::nn::functional;

struct Concat : torch::jit::Module{
   /* at::Tensor forward(Layer *layer){
        at::Tensor out = layer->output[layer->index];
        out = torch::cat({out[layer->index-1], out[layer->index-7]}, 1);
        return out;
    }*/
};
void get_submodule_densenet(torch::jit::script::Module module, std::vector<torch::jit::Module> &child, vector<pair<int, int>> &denseblock){
	Concat concat;
    if(module.children().size() == 0){
        child.push_back(module);
        return;
    }
    for(auto children : module.named_children()){
        if(children.name.find("denseblock") != std::string::npos){
            int num_layer = 0;
            int size = child.size();
            for(auto layer : children.value.named_children()){
                if(layer.name.find("denselayer") != std::string::npos){
                    num_layer++;
                }
                get_submodule_densenet(layer.value,child,denseblock);
                child.push_back(concat);
            }
            denseblock.push_back(make_pair(size, num_layer));
            continue;
        }
        get_submodule_densenet(children.value, child, denseblock);
    }
}
/*
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
}*/

void *predict_densenet(Net *input){
    std::cout<< "dense" <<"\n";
    std::vector<torch::jit::Module> child = input->child;
	std::vector<torch::jit::IValue> inputs = input->input;
	std::cout<<child.size()<<"\n";
    int i;
    for(i=0;i<child.size();i++){
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
        cout<<nl.net->index<<"\n";
		input->input.clear();
		input->input.push_back(input->layer[i].output);
        //cout<<"sssssssssssssssssssssssss\n";
		pthread_mutex_unlock(&mutex_t[input->index_n]);
    }
    std::cout << "\n*****Dense result*****" << "\n";
	std::cout << (input->layer[i].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
}

void forward_densenet(th_arg *th){ //vector<torch::jit::Module> &child, vector<torch::jit::IValue> inputs, vector<pair<int, int>> denseblock){
    pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
    std::vector<torch::jit::Module> child = nl->net->child;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
    std::vector<pair<int,int>> denseblock = nl->net->block;
    int k =nl->net->index;
    at::Tensor out;
    int j;
    if(k==0)
        j=0;
    else
        j = nl->net->j;

    std::cout<<"j = "<<j<<"\n";
    std::cout<<denseblock[j].first<<"\n";
    
    if(k == child.size()-1){
        out = F::relu(inputs[0].toTensor(), F::ReLUFuncOptions().inplace(true));
        out = F::adaptive_avg_pool2d(out, F::AdaptiveAvgPool2dFuncOptions(1));
        out = out.view({out.size(0), -1});
	    inputs.clear();
	    inputs.push_back(out);
        out = child[k].forward(inputs).toTensor();
    }
    else if(typeid(child[k])==typeid(Concat)){
        std::cout<< "typeid\n";
        out = torch::cat({nl->net->layer[k-7].output, nl->net->layer[k-1].output}, 1);
    }
    else if(j < denseblock.size() && k == denseblock[j].first){
        std::cout<<"jjjjjjjjjjjjjjjjjjjjjjjjjj\n";
        out = torch::cat({nl->net->layer[k-1].output},1);
        inputs.clear();
        inputs.push_back(out);
        
        out = child[k].forward(inputs).toTensor();
        j+=1;
        std::cout<<"vvvvvvv\n";
    }
    else{
        std::cout<<"eeeeeeeeeeeeeeeeeeeee\n";
        out = child[k].forward(inputs).toTensor();
    }
    std::cout<< &(nl->net->layer[k].output)<<"\n";
    std::cout<< out<<"\n";
    at::Tensor ten = out;
    std::cout<<typeid(out).name() << typeid(nl->net->layer[k].output).name()<<std::endl;
    nl->net->layer[k].output = out;
    std::cout<< &(nl->net->layer[k].output)<<"\n";
    nl->net->index = k;
    std::cout<<"pppppppp\n";
    nl->net->j =j;
	cond_i[nl->net->index_n]=0;
    std::cout<< "dense layer " << k <<"end\n";
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
}
