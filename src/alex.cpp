#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "alex.h"

namespace F = torch::nn::functional;
//using namespace std;

void get_submodule_alexnet(torch::jit::script::Module module, std::vector<torch::jit::Module> &child){
	if(module.children().size() == 0){
		child.push_back(module);
		return;
	}
	for(auto children : module.named_children()){
		get_submodule_alexnet(children.value, child);
	}
}

void *predict_alexnet(Net *input){
	std::vector<torch::jit::Module>  child = input->child;
	std::vector<torch::jit::IValue> inputs = input->input;
	//std::cout<<child.size()<<'\n';
	int i;
	for(i=0;i<child.size();i++){
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1;
		
		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		//std::cout<<"index = "<<nl.index<<'\n';
		std::cout << "Before thpool add work Alex "<< i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_alexnet,&th);
		std::cout << "After thpool add work Alex "<< i << "\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		input->input.clear();
		input->input.push_back(input->layer[i].output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout << "\n*****Alex result*****" << "\n";
	std::cout << (input->layer[i].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}

void forward_alexnet(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::cout << nl->net->child.size() << " " << nl->net->input.size() << " "<< nl->net->index_n<< "\n";
	std::vector<torch::jit::Module> child = nl->net->child;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	int k = nl->net->index;
	at::Tensor out;
	std::cout<<"k = "<<k<<"\n";
	if(k==14){
		//out = out.view({out.size(0), -1});
		out = out.view({inputs[0].toTensor().size(0), -1});
		inputs.clear();
		inputs.push_back(out);
		out = child[k].forward(inputs).toTensor();
	}
	else{
		out = child[k].forward(inputs).toTensor();
	}
	std::cout<<"before out\n";
	nl->net->layer[k].output = out;
	std::cout<<"after out\n";
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}


#if 0 
void *predict_alexnet(Net *input){
	std::vector<torch::jit::Module> child = input->child;
	std::vector<torch::jit::IValue> inputs = input->layer->input;
	//std::cout<<child.size()<<'\n';
	for(int i=0;i<child.size();i++){
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1;
		
		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		//std::cout<<"index = "<<nl.index<<'\n';
		//std::cout << "Before thpool add work Alex "<< i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_alexnet,&th);
		//std::cout << "After thpool add work Alex "<< i << "\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		input->layer->input.clear();
		input->layer->input.push_back(input->layer->output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout << "\n*****Alex result*****" << "\n";
	std::cout << (input->layer->output[0]).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}

void forward_alexnet(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	//std::cout << nl->net->child.size() << " " << nl->net->inputs.size() << " "<< nl->net->index_n<< "\n";
	std::vector<torch::jit::Module> child = nl->net->child;
	std::vector<torch::jit::IValue> inputs = nl->net->layer->input;
	at::Tensor out = nl->net->layer->output;
	int k = nl->net->index;

	if(k==14){
		out = out.view({out.size(0), -1});
		//out = out.view({inputs[0].toTensor().size(0), -1});
		inputs.clear();
		inputs.push_back(out);
		out = child[k](inputs).toTensor();
		//out = child[k].forward(inputs).toTensor();
	}
	else{
		out = child[k](inputs).toTensor();
		//out = child[k].forward(inputs).toTensor();
	}
	nl->net->layer->output= out;
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}
#endif