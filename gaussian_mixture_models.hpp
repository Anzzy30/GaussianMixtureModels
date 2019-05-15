//
// Created by anthony on 09/05/19.
//

#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <limits>
#include "timer.hpp"
namespace gmm{

    std::random_device dev;
    std::mt19937 rng(dev());

    typedef Eigen::MatrixXf MatrixX;
    typedef Eigen::VectorXf VectorX;
    typedef std::vector<MatrixX ,Eigen::aligned_allocator<MatrixX> > vectorMaxtrixX;
    typedef std::vector<VectorX ,Eigen::aligned_allocator<VectorX> > vectorVectorX;

    //https://stats.stackexchange.com/questions/326671/different-covariance-types-for-gaussian-mixture-models
    enum CovarianceTypes{
        FULL,
        TIED,
        DIAGONAL,
        TIED_DIAGONAL,
        SPHERICAL
    };


    static void probability_density(const MatrixX& input_cluster,const VectorX& means,const MatrixX& cov, MatrixX &responses){
        assert(cov.size() <= 16);//Must implement more efficient inverse and determinant computing (such as PartialPivLU and FullPivLU)
        float det = cov.determinant();
        if (det < 1e-8) {
            responses = MatrixX::Zero(input_cluster.rows(), 1);
        }
        MatrixX inverse_cov = cov.inverse();
        MatrixX dist = input_cluster.rowwise() - means.transpose();
        MatrixX maha_dist = (dist * (inverse_cov * dist.transpose())).diagonal();
//        MatrixX maha_dist = (inverse_cov * dist.transpose()).cwiseProduct(dist);
        float inv_norm = 1/std::sqrt(std::pow(2.f * M_2_SQRTPI,means.size()) * det);

        responses = inv_norm * (-0.5 * maha_dist).array().exp();

    }
    static void computeCovariance(const vectorMaxtrixX& input_clusters,const MatrixX& means, vectorMaxtrixX& out_covs){
        assert(!input_clusters.empty() && means.cols() > 0 && means.rows() > 0);
        out_covs.resize(means.size(),MatrixX::Identity(means.cols(),means.cols()));
        MatrixX Epsilon = MatrixX::Identity(means.cols(),means.cols()) * 1e-8;
        for(size_t i = 0; i < means.rows(); ++i){
            MatrixX centered = input_clusters[i].rowwise() - means.row(i);
            out_covs[i] = (centered.adjoint() * centered) / float(input_clusters[i].rows()) + Epsilon;
        }
    }


    class KMeans{
    public:
        KMeans(const MatrixX &data,const uint32_t k,const size_t max_iter):_data(data),_K(k),_max_iter(max_iter){
        }


        void start_clustering(){
            random_init();
            estimate_centers();
        }


        const vectorMaxtrixX &getClusterData() const {
            return cluster_data;
        }
        const MatrixX &getCenters() const {
            return _centers;
        }
    protected:
        void random_init() {
            std::cout << "Random Initialisation" <<std::endl;
            _centers = MatrixX(_K, _data.cols());
            std::uniform_int_distribution<std::mt19937::result_type> distribution(0, _data.rows()-1);
            // Completely randomly choose initial cluster centers
            for (uint32_t i = 0; i < _K; i++) {
                //Ensure not same center
                if(i > 0) {
                    float dist;
                    do{
                        _centers.row(i) = _data.row((distribution(rng)));
                        dist = (_centers.block(0, 0, i, _data.cols()).rowwise() - _centers.row(i)).rowwise().squaredNorm().minCoeff();


                    }while(dist < 0.01);


                }else{
                    _centers.row(i) = _data.row((distribution(rng)));
                }


            }

            _labels.resize(_data.rows());
            for (uint32_t i = 0; i < _labels.size(); i++) _labels[i] = i;

        }


        bool compute(){
            bool hasChanged = false;
            MatrixX sum_points = MatrixX::Zero(_centers.rows(),_centers.cols());
            VectorX counts = VectorX::Zero(_centers.rows(),1);
            for (uint32_t idx = 0; idx < _data.rows(); ++idx) {
                uint32_t best_cluster = 0;
                (_centers.rowwise() - _data.row(idx)).rowwise().squaredNorm().minCoeff(&best_cluster);
                if(_labels[idx] != best_cluster) {
                    _labels[idx] = best_cluster;
                    hasChanged = true;
                }

                sum_points.row(best_cluster) = sum_points.row(best_cluster) +_data.row(idx);
                counts.row(best_cluster).array() = counts.row(best_cluster).array() + 1.f;

            }
            _centers = sum_points.array()/ counts.array().replicate(1,sum_points.cols());
            return hasChanged;
        }

        void estimate_centers(){

            size_t iter = 0;
            std::cout << "Start estiamation" <<std::endl;
            while(compute() && iter++ < _max_iter);
            std::cout << "Estimation end at iteration: " << iter << "/"<< _max_iter <<std::endl;
            std::cout << "Clusters centers are: " << _centers <<std::endl;
            fill_cluster();
        }


        void fill_cluster(){
            cluster_data.resize(_K,MatrixX::Zero(0,0));

            for (uint32_t idx = 0; idx < _data.rows(); ++idx) {
                cluster_data[_labels[idx]].conservativeResize(cluster_data[_labels[idx]].rows()+1,_data.cols());
                cluster_data[_labels[idx]].row(cluster_data[_labels[idx]].rows()-1)  << _data.row(idx);

            }

        }



    private:

        MatrixX _data;
        MatrixX _centers;

        uint32_t _K;
        size_t _max_iter;
        std::vector<uint32_t > _labels;

        vectorMaxtrixX cluster_data;
    };

    class GaussianMixtureModel{


    public:


        GaussianMixtureModel(CovarianceTypes covariance_type, uint32_t n_components, size_t max_iter){
            _n_components = n_components;
            _covariance_type = covariance_type;
            _max_iter = max_iter;
        }

        bool kmeans_init(const MatrixX & data, size_t kmean_max_iter){
            assert(data.cols() > 0 && data.rows() > 0);
            KMeans kmeans(data, _n_components,kmean_max_iter);
            kmeans.start_clustering();

            _weights = MatrixX::Constant(_n_components,1,1.f/_n_components);

            _means = MatrixX(kmeans.getCenters().rows(),kmeans.getCenters().cols());
            _means << kmeans.getCenters();
            computeCovariance(kmeans.getClusterData(),_means,_covs);


            return true;
        }

        void fit(const MatrixX & data, size_t kmean_max_iter){
            kmeans_init(data, kmean_max_iter);

            MatrixX responses = MatrixX::Zero(data.rows(),_n_components);
            MatrixX weighted_sum;
            VectorX nc;
            MatrixX diff;
            MatrixX Epsilon = MatrixX::Identity(_n_components,_n_components) * 1e-8;
            float ll = 0;
            float prev_ll = std::numeric_limits<float>::max();
            for(size_t iter =0;iter < _max_iter; ++iter) {
                //E-STEP
                for (size_t i = 0; i < _n_components; ++i) {
                    MatrixX buffer;
                    probability_density(data, _means.row(i), _covs[i], buffer);
                    responses.col(i) << buffer * _weights(i);
                }
                responses = (responses.array()) / responses.rowwise().sum().array().replicate(1,responses.cols());
                ll = responses.array().log().sum();

                if(std::abs(ll - prev_ll) < 1e-4){
                    std::cout << "Converged at iterationn: " << iter <<std::endl;
                    break;
                }
                prev_ll = ll;
                nc = responses.colwise().sum();
                weighted_sum = responses.transpose() * data ;
                _means = weighted_sum.array() / nc.array().replicate(1,data.cols());
                _weights = nc / data.rows();
                for (size_t i = 0; i < _n_components; ++i) {
                    diff = data.rowwise() - _means.row(i);
                    _covs[i] = Epsilon;
                    _covs[i] = diff.transpose() * (diff.array() * responses.col(i).array().replicate(1,data.cols())).matrix()   / nc(i);
                }

            }

            std::cout << "Mean:\n" << _means <<std::endl;
            std::cout << "Weights:\n" << _weights.transpose() <<std::endl;
            std::cout << "Covariance:\n";
            for (size_t i = 0; i < _n_components; ++i) {
                std::cout << _covs[i] <<std::endl;
            }
        }


    private:
        CovarianceTypes _covariance_type;
        uint32_t _n_components;
        size_t _max_iter;

        float _epsilon = 1e-8;


        MatrixX _weights;
        vectorMaxtrixX _covs;
        MatrixX _means;

    };

}