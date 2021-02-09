#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "resnet18.h"

using namespace std;
namespace F = torch::nn::functional;

void get_submodule_resnet18(torch::jit::script::Module module,std::vector<torch::jit::Module> *child, vector<int> *basicblock){
	
    if(module.children().size() == 0){
	    (*child).push_back(module);
            return;
    }
    for(auto children : module.named_children()){
	   //std::cout<<typeid(children.value).name()<<"\n";
	   // std::cout<<children.value.type()->name().value().name()<<"\n";
    	if(children.name == "0" || children.name == "1")
			(*basicblock).push_back((*child).size());

		if(children.name != "downsample") //conv1x1
			get_submodule_resnet18(children.value, child, basicblock);
		else
			(*child).push_back(children.value);

    }
}

void *predict_resnet18(Net *input){
    std::vector<torch::jit::Module> child = input->child;
	std::vector<torch::jit::IValue> inputs = input->inputs;
	//std::vector<int> basicblock = input->basicblock;
	std::vector<int> add_identity;
	std::cout<<"child = "<<child.size()<<'\n';
	std::cout<<"basic = "<<input->basicblock.size()<<'\n';

	//pthread_mutex_lock(&mutex_t[input->index_n]);
	//cond_i[input->index_n] = 1; //right?
 	
	 for(int i=0;i<(input->basicblock).size();i++)
 	{
	 	if(input->basicblock[i] == 14 || input->basicblock[i] == 25 || input->basicblock[i] == 36)
			add_identity.push_back(input->basicblock[i]+5);
	 	else
			add_identity.push_back(input->basicblock[i]+4);
		std::cout<<input->basicblock[i]<<" ";
	}
	std::cout<<"\n";
	
	for(int i=0;i<add_identity.size();i++){
		std::cout<<add_identity[i]<<" ";
	}
	std::cout<<"\n";

	for(int i = 0;i<child.size();i++) {
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1; //right?
		
		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.add_identity = add_identity;
		nl.index = i;

		th_arg th;
		th.arg = &nl;
		//std::cout << "Before thpool add work RES " << i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_resnet18,&th);
		//std::cout << "After thpool add work RES " << i <<"\n";
		//cout<<"id = " <<(nl.net->identity)<<"\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		input->inputs.clear();
		input->inputs.push_back(input->output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout << "\n*****Res result*****" << "\n";
	std::cout << (input->output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
}

void forward_resnet18(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::Module> child = nl->net->child;
	std::vector<torch::jit::IValue> inputs = nl->net->inputs;
	std::vector<int> basicblock = nl->net->basicblock;
	std::vector<int> add_identity = nl->add_identity;
	at::Tensor identity = nl->net->identity;
	//vector<torch::jit::IValue> input; // ==inputs?
	vector<torch::jit::IValue> inputs_cpy;
	at::Tensor out = nl->net->output;
	int k =nl->index;
	static int j;

	if(k != 0)
		j = nl->net->j;

    //std::cout<<"res layer index = "<<k<<"\n";
    //output.clear();
    if(j < basicblock.size() && k == basicblock[j]){
	   //std::cout<<"basicblock\n";
	   identity = out;
    }

    if(k==48)
    {	
		out = nl->net->output.view({nl->net->output.size(0), -1});
		//out = out.view({out.size(0), -1});
       	inputs.clear();
       	//out =  out.to(at::kCPU);
       	inputs.push_back(out);
       	//child[k].to(at::kCPU);
       	out = child[k].forward(inputs).toTensor();
    }
    else if(k == 19 || k == 30 || k == 41){
		inputs_cpy.clear();
		inputs_cpy.push_back(identity);
        identity = child[k].forward(inputs_cpy).toTensor();
    }
    else{
    	out = child[k].forward(inputs).toTensor();
    }

    if((j) < add_identity.size() && k == add_identity[j]){
		out += identity;
		inputs.clear();
		inputs.push_back(nl->net->output);
		//if(add_identity[j]-basicblock[j]==5)
		//	out = child[i-3].forward(input).toTensor();
		//else
		//	out = child[i-2].forward(input).toTensor();
		nl->net->output = child[2].forward(inputs).toTensor();
		j++;
    }
    nl->net->output = out;
	nl->net->identity = identity;
	nl->net->j = j;
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
    //cout<<out.sizes()<<"\n\n";
}
