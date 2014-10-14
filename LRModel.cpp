/*
 * LRModel.cpp
 *
 *  Created on: Sep 21, 2014
 *      Author: wtipton
 */

#include "LRModel.h"

#include <cmath>
#include <iostream>
#include <zlib.h>

LRModel::LRModel(int n, int t, const std::vector<std::vector<double>>& data, double alpha, double lambda, double e)
: n_(n), t_(t), div_len_(1.0/t), num_lattice_points_(pow(t+1, n)) {
	for (int i = 0; i < num_lattice_points_; i++)
		b_.push_back(1.0);

	initAdjLatticePoints();

	trainModel(data, alpha, lambda, e);
}

LRModel::LRModel(const std::string& filename) {
	gzFile inFile = gzopen(filename.c_str(), "rb");
	if (inFile == Z_NULL)
		std::cout << "ERROR: LRModel could not open file " << filename << std::endl;

	if (gzread (inFile, (char*)(&n_), sizeof (int)) < (int) sizeof(int))
		std::cout << "ERROR: gzread failed in LRModel reading n_" << std::endl;
	if (gzread (inFile, (char*)(&t_), sizeof (int)) < (int) sizeof(int))
		std::cout << "ERROR: gzread failed in LRModel reading t_" << std::endl;
	if (gzread (inFile, (char*)(&div_len_), sizeof (double)) < (int) sizeof(double))
		std::cout << "ERROR: gzread failed in LRModel reading div_len_" << std::endl;
	if (gzread (inFile, (char*)(&num_lattice_points_), sizeof (int)) < (int) sizeof(int))
		std::cout << "ERROR: gzread failed in LRModel reading num_lattice_points_" << std::endl;

	double b;
	for (int i = 0; i < num_lattice_points_; i++) {
		if (gzread (inFile, (char*)(&b), sizeof (double)) < (int) sizeof(double))
			std::cout << "ERROR: gzread failed in LRModel reading b_" << std::endl;
		b_.push_back(b);
	}

	gzclose(inFile);

	initAdjLatticePoints();
}

void LRModel::writeToDisk(const std::string& filename) const {
	gzFile outFile = gzopen(filename.c_str(), "wb");
	if (outFile == Z_NULL)
		std::cout << "ERROR: gzwrite failed to open " << filename << std::endl;

	// Save: n, t, div_len_, num_lattice_points_, and b. Will need to re-calculate adj_lattice_pts_ upon reading.

	if (! gzwrite (outFile, (char*)(&n_), sizeof (int)))
		std::cout << "ERROR: gzwrite failed writing n_ in LRModel:writeToDisk" << std::endl;
	if (! gzwrite (outFile, (char*)(&t_), sizeof (int)))
		std::cout << "ERROR: gzwrite failed writing t_ in LRModel:writeToDisk" << std::endl;
	if (! gzwrite (outFile, (char*)(&div_len_), sizeof (double)))
		std::cout << "ERROR: gzwrite failed writing div_len_ in LRModel:writeToDisk" << std::endl;
	if (! gzwrite (outFile, (char*)(&num_lattice_points_), sizeof (int)))
		std::cout << "ERROR: gzwrite failed writing num_lattice_points_ in LRModel:writeToDisk" << std::endl;

	for (int i = 0; i < num_lattice_points_; i++)
		if (! gzwrite (outFile, (char*)(&(b_[i])), sizeof(double)))
			std::cout << "ERROR: gzwrite failed in LRModel:writeToDisk" << std::endl;

	gzclose(outFile);
}

LRModel::~LRModel() {
	// nothing to do here
}


void LRModel::initAdjLatticePoints() {
	for (int i_point = 0; i_point < num_lattice_points_; i_point++) {
		std::vector<int> adj_pts; // will hold indices of points adjacent to point i_point

		// Get the coords of i_point.
		int x[n_];
		indxToCoords(i_point, x);

		// Initialize delta to all -1's.  (x+delta) is a candidate adjacent point, where each
		// entry of delta is -1, 0, or 1.  We just need to check that delta isn't all 0's nor is it
		// out of bounds.  And, we'll iterate through all possible delta vectors, essentially, counting
		// in base 3 where the three possible digits are -1, 0, and 1.
		int delta[n_];
		for (int i = 0; i < n_; i++)
			delta[i] = -1;

		// the following loop does this, essentially:
		/*
		while (1) {
			if (delta is not all 0s && x+delta is in bounds) {
				add x+delta to adj_pts;
			}
			if (delta is all 1s)
				break;
			increment delta;
		}
		*/

		while (1) {
			// check if delta is all zeros
			bool delta_is_all_zeros = true;
			for (int i = 0; i < n_; i++)
				if (delta[i] != 0)
					delta_is_all_zeros = false;

			// calculate x+delta
			int x_plus_delta[n_];
			for (int i = 0; i < n_; i++)
				x_plus_delta[i] = x[i] + delta[i];

			// check if x+delta is in bounds
			bool x_plus_delta_is_in_bounds = true;
			for (int i = 0; i < n_; i++)
				if (x_plus_delta[i] < 0 || x_plus_delta[i] >= t_+1)
					x_plus_delta_is_in_bounds = false;

			// add x+delta to list of adjacent points if appropriate
			if (!delta_is_all_zeros && x_plus_delta_is_in_bounds) {
				adj_pts.push_back(coordsToIndx(x_plus_delta));
			}

			// if delta is all ones, we're done
			bool delta_is_all_ones = true;
			for (int i = 0; i < n_; i++)
				if (delta[i] != 1)
					delta_is_all_ones = false;
			if (delta_is_all_ones)
				break;

			// increment delta
			int i = 0;
			while (i < n_ && delta[i] == 1) {
				delta[i] = -1;
				i++;
			}
			delta[i]++;
		}

		adj_lattice_pts_.push_back(adj_pts);
	}

/*	// print adjacent lattice points
	for (int n = 0; n < num_lattice_points_; n++) {
		std::cout << "adj to " << n << ":";
		for (int k=0; k<adj_lattice_pts_[n].size(); k++)
			std::cout << " " << adj_lattice_pts_[n][k];
		std::cout << std::endl;
	}*/
}


void LRModel::trainModel(const std::vector<std::vector<double>>& data, double alpha, double lambda, double e) {
	std::vector<std::map<int,double>> weights; // weights of training examples
	std::vector<std::vector<int>> adjLatticeToData; // essentially a map from lattice points to adjacent training examples
	std::vector<std::vector<int>> adjDataToLattice; // essentially a map from training examples to adjacent lattice points
	// initialize weights and adjacency structures
	makeTrainingWeightsAndAdj(data, weights, adjDataToLattice, adjLatticeToData);

	// batch gradient descent
	while (1) {
		double norm_of_grad = 0.0;
		for (int k = 0; k < num_lattice_points_; k++) {
			double dJdbk = 0.0;
			// calculate the derivative of J wrt b_k
			for (int i : adjLatticeToData[k]) { // here, i is an index into the data vector of an example that is adjacent to lattice point k
				double prediction_i = 0.0;
				for (int j : adjDataToLattice[i]) // for lattice points j adjacent to i
					prediction_i += b_[j] * weights[i][j];
				dJdbk += (prediction_i - data[i][0]) * weights[i][k] / data.size();
			}
			for (int n : adj_lattice_pts_[k]) { // regularization: for nodes n adjacent to k
				dJdbk -= lambda * 2 * (b_[n] - b_[k]);
			}
			b_[k] -= alpha * dJdbk;
			norm_of_grad += dJdbk*dJdbk;
		}
		norm_of_grad = sqrt(norm_of_grad);
		std::cout << alpha << " " << norm_of_grad << " " << getRMSE(data) << std::endl;
		if (norm_of_grad <= e)
			break;
	}

/*	// Print learned values at each lattice point.
	for (int k = 0; k < num_lattice_points_; k++)
		std::cout << b_[k] << std::endl;
*/
}

void LRModel::makeTrainingWeightsAndAdj(const std::vector<std::vector<double>>& data, std::vector<std::map<int,double>>& weights,
		                                std::vector<std::vector<int>>& adjDataToLattice, std::vector<std::vector<int>>& adjLatticeToData) {
	weights.clear();
	adjDataToLattice.clear();
	adjLatticeToData.clear();
	for (size_t i = 0; i < data.size(); i++) {
		weights.push_back(std::map<int,double>());
	}
	for (int i = 0; i < num_lattice_points_; i++) {
		adjLatticeToData.push_back(std::vector<int>());
	}

	for (size_t i = 0; i < data.size(); i++) {
		std::map<int,double> a = getWeights(&(data[i][1]));
		std::vector<int> adj_pts;
		for (std::pair<int,double> p : a) {
			int lat_pt_indx = p.first;
			double lat_pt_weight = p.second;

			adjLatticeToData[lat_pt_indx].push_back(i);
			weights[i][lat_pt_indx] = lat_pt_weight;
			adj_pts.push_back(lat_pt_indx);
		}
		adjDataToLattice.push_back(adj_pts);
	}
}

std::map<int,double> LRModel::getWeights(const double x[]) const {
	std::map<int,double> result;

	// Find the  each dimension, there's ind the upper and lower bounding tick marks for x in each dimension
	int low[n_], high[n_];
	for (int i = 0; i < n_; i++) {
		low[i] = floor(x[i]*t_);
		high[i] = ceil(x[i]*t_);
	}

	// Avoid the degenerate case where a point is right on a lattice point, because that'd lead to a div-by-0 later.
	for (int i = 0; i < n_; i++) {
		if (low[i] == high[i]) {
			if (low[i] == 0)
				high[i]++;
			else
				low[i]--;
		}
	}

	// convert from lattice to cartesian
	double d_low[n_], d_high[n_];
	for (int i = 0; i < n_; i++) {
		d_low[i] = div_len_ * low[i];
		d_high[i] = div_len_ * high[i];
	}

	// Calculate weights for adjacent points.
	// Denom is the total area of one grid box.
	double denom = 1.0;
	for (int i = 0; i < n_; i++)
		denom *= (d_high[i] - d_low[i]);

	// A little bit of magic here to loop over all lattice points adjacent to x.  So there's one such lattice point
	// for each number from 0 up to (2^n_)-1.  Think of it in binary, and the ith bit controls whether we use the
	// high or low tick mark in the ith dimension.
	for (int i = 0; i < (1<<n_); i++) {
		int c[n_];
		int ci = i;
		double area = 1.0;
		for (int j = 0; j < n_; j++) {
			if (ci % 2) {
				c[j] = low[j];
				area *= d_high[j] - x[j];
			} else {
				c[j] = high[j];
				area *= x[j] - d_low[j];
			}
			ci /= 2;
		}
		result[coordsToIndx(c)] = area / denom;
	}

	return result;
}

double LRModel::predict(const double x[]) const {
	std::map<int,double> a = getWeights(x);

	double result = 0.0;
	for (auto p : a) {
		result += p.second * b_[p.first];
	}

	return result;
}

double LRModel::getRMSE(const std::vector<std::vector<double>>& data) const {
	double mse = 0.0;
	for (auto datapt : data) {
		double resid = datapt[0] - predict(&datapt[1]);
		mse += resid * resid;
	}
	mse /= data.size();
	return sqrt(mse);
}

int LRModel::coordsToIndx(const int c[]) const {
	int result = c[0];
	for (int i = 1; i < n_; i++) {
		result *= t_+1;
		result += c[i];
	}
	return result;
}

void LRModel::indxToCoords(int indx, int c[]) const {
	for (int i = n_-1; i >=0; i--) {
		c[i] = indx % (t_+1);
		indx /= (t_+1);
	}
}
