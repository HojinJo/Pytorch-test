#include <torch/script.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>


using namespace std;

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
	   if(children.name != "downsample")
		 get_submodule(children.value, child, basicblock);
	   else
		 (*child).push_back(children.value);

    }
}

at::Tensor forward_resnet18(vector<torch::jit::Module> &child, vector<torch::jit::IValue> inputs){

 vector<int> add_identity;
 for(int i=0;i<basicblock.size();i++)
 {
	 if(basicblock[i] == 14 || basicblock[i] == 25 || basicblock[i] == 36)
		 add_identity.push_back(basicblock[i]+5);
	 else
		 add_identity.push_back(basicblock[i]+4);
		
	 cout<<basicblock[i]<<" ";
 }
 cout<<"\n";
 for(int i=0;i<add_identity.size();i++){
	 cout<<add_identity[i]<<" ";
 }
 cout<<"\n";

 vector<torch::jit::IValue> input;
 vector<torch::jit::IValue> input_cpy;
 at::Tensor identity;
 at::Tensor out;
 
 int j = 0;
 // layer forward 
 for(int i = 0;i<child.size();i++) {
    cout<<"i = "<<i<<"\n";
    //print_script_module(child[i], 0);
    //output.clear();
    if(j < basicblock.size() && i == basicblock[j]){
	   //cout<<"basicblock\n";
	   identity = out; 
    }
    if(i==48)
    {
       out = out.view({out.size(0), -1});
       input.clear();
       out =  out.to(at::kCPU);
       input.push_back(out);
       child[i].to(at::kCPU);
       out = child[i].forward(input).toTensor();
    }
    else if(i == 19 || i == 30 || i == 41){
	input_cpy.clear();
	input_cpy.push_back(identity);
        identity = child[i].forward(input_cpy).toTensor();
    }
    else{
    	out = child[i].forward(input).toTensor();
    }

    if(j<add_identity.size() && i == add_identity[j]){
    	out += identity;
	input.clear();
	input.push_back(out);
	//if(add_identity[j]-basicblock[j]==5)
	//	out = child[i-3].forward(input).toTensor();
	//else
	//	out = child[i-2].forward(input).toTensor();
	out = child[2].forward(input).toTensor();
	j++;
    }
    input.clear();
    input.push_back(out);
    cout<<out.sizes()<<"\n\n";
    //cout<<out.slice(1,0,1)<<"\n\n\n";
    //std::cout << out.sizes() << '\n';  
 }
cout<<"\n\n\n";
 std::cout << "*****resnet result*****"<<"\n";
 std::cout << out.slice(/*dim=*/1, /*start=*/0, /*end=*/15) << '\n';
 std::cout << out.sizes()<<"\n";  
}
