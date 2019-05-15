#include <iostream>
#include "gaussian_mixture_model.hpp"

#include <fstream>
#include <string>
#include <sstream>

//3 cluster must be -2 -2 -2 & 0 0 0 & 2 2 2





void test_1D_gmm(){
    size_t NUMBER_POINTS = 1000;

    Eigen::MatrixXf data(NUMBER_POINTS,1);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::normal_distribution<float> distribution_1(0.,1.0);
    std::normal_distribution<float> distribution_2(2.,1.0);
    std::normal_distribution<float> distribution_3(-2.,1.0);

    std::uniform_int_distribution<> k(0,2);

    for(size_t i =0;i<NUMBER_POINTS; ++i){
        int val = k(rng);
        if(val == 0)
            data.row(i) << distribution_1(rng);
        else if(val == 1)
            data.row(i) << distribution_2(rng);
        else if(val == 2)
            data.row(i) << distribution_3(rng);
    }
    gmm::GaussianMixtureModel gmm_ (gmm::CovarianceTypes::FULL, 3, 10000);
    gmm_.fit(data,100);
}

void test_gmm_from_load_data(std::string path, int dim){

    std::ifstream file(path);
    if(!file.is_open()) {
        std::cerr << "Error openning data file." <<std::endl;
        return;
    }
    std::string   line;

    std::vector<std::string> strs;

    int cpt = 0;
    while (std::getline(file, line))
    {
        strs.push_back(line);
    }
    Eigen::MatrixXf data(strs.size(),dim);

    for(int i = 0; i < strs.size(); ++i)
    {

        char *end;
        char *pointer = const_cast<char *>(strs[i].data());
        for(int d = 0; d < dim; ++d){

            data.row(i)(d) = std::strtof(pointer, &end);
            pointer = end;
        }

    }
    assert(dim == data.cols());
    std::cout << data.row(0) <<std::endl;
    gmm::GaussianMixtureModel gmm_ (gmm::CovarianceTypes::FULL, 3, 10000);
    gmm_.fit(data,100);

}

int main() {
    //int dim = 3, string path = data.txt
    //test_gmm_from_load_data(dim,dim);
    test_1D_gmm();
    return 0;
}