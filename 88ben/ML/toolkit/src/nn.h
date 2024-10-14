// cd Documents/BYU/2017Winter/CS478/CPP/toolkit/src
// make opt &&  ../bin/MLSystemManager -L nn -A ../../datasets/vowel.arff -E random 0.75

#ifndef NN_H
#define NN_H

#include "learner.h"
#include "rand.h"
#include "error.h"
#include <iostream>

using namespace std;

class NNLearner : public SupervisedLearner {


private:

	Rand& m_rand;

	int vs;				// size of validation set
	int numL;			// number of layers
	int rows;			// number of rows
	int * numN;			// number of nodes per layer
	int * inputs;		// input values

	double *** wgts;	// weights
	double *** bestW;	// best weights
	double *** chgW;	// change in weights
	double ** bias;		// biases
	double ** vals;		// output values (input for first layer)
	double ** errs;		// errors
	double * tars;		// targets

	double momentum;	// used in calculating weight change
	double c;			// learning rate


public:

	NNLearner(Rand& r) : SupervisedLearner(), m_rand(r) { }

	virtual ~NNLearner() {

		for (int web = 0; web < numL - 1; web++) {
			for (int n = 0; n < numN[web]; n++) {
				delete [] wgts [web][n];
				delete [] chgW [web][n];
				delete [] bestW[web][n];
			}

			delete [] wgts [web];
			delete [] chgW [web];
			delete [] bestW[web];
		}

		for (int layer = 0; layer < numL; layer++) {
			delete [] vals[layer];
			delete [] bias[layer];
			delete [] errs[layer];
		}

		delete [] wgts;			delete [] vals;			delete [] numN;
		delete [] chgW;			delete [] bias;			delete [] tars;
		delete [] bestW;		delete [] errs;			delete [] inputs;
	}


	virtual void train(Matrix& features, Matrix& labels) {

		if(features.rows() != labels.rows())
			ThrowError("Features and labels do not have same number of rows");

		createNet(features, labels);

		int numE = 0;		// number of epochs
		int maxE = 1000;	// max number of epochs allowed
		int badE = 0;		// number of sequential bad epochs
		int stopE = 100;	// stopping criteria for number of bad epochs
		int sameE = 0;		// number of epochs that VS gets SAME accuracy
		int mostE = 0;		// most sequential good epochs in a row

		while (maxE-- && badE != stopE) {

			numE++;
			features.shuffleRows(m_rand, &labels);

			for (int r = 0; r < rows - vs; r++) {
				resetNet(features, labels, r);
				forward();
				backprop();
			}

			// VALIDATION SET

			int goodE = 0;
			for (int r = rows - vs; r < rows; r++) {
				resetNet(features, labels, r);
				forward();
				if (tars[findMax()]) goodE++;
			}

			if (goodE == mostE) {
				updatebestW();
				sameE++;
			} else if (goodE > mostE) {
				updatebestW();
				mostE = goodE;
				badE = 0;
				sameE = 0;
			} else {
				sameE = 0;
				badE += 1 + sameE;
			}
		}
	}


	void updatebestW()
	{
		for (int web = 0; web < numL - 1; web++)
			for (int parent = 0; parent < numN[web]; parent++)
				for (int child = 0; child < numN[web+1]; child++)
					bestW[web][parent][child] = wgts[web][parent][child];
	}


	void forward() {

		for (int layer = 1; layer < numL; layer++)
			for (int n = 0; n < numN[layer]; n++)
				activate(layer, n);
	}


	void backprop() {

		for (int layer = numL-1; layer > 0; layer--) {
			for (int n = 0; n < numN[layer]; n++) {
				calcError(layer, n);
				weights(layer-1, n);
			}
		}
	}


	void resetNet(Matrix& features, Matrix& labels, int r) {

		for (int n = 0; n < numN[0]; n++)
			vals[0][n] = features[r][inputs[n]];

		for (int n = 0; n < numN[numL-1]; n++)
			tars[n] = 0;

		tars[int(labels[r][0])] = 1;
	}


	void resetNet(const std::vector<double>& features) {

		for (int n = 0; n < numN[0]; n++)
			vals[0][n] = features[inputs[n]];
	}


	virtual void predict(const std::vector<double>&features, std::vector<double>&labels) {
		resetNet(features);
		forward();
		labels[0] = findMax();
	}


	void calcError(int layer, int n) {

		if (layer != numL - 1) {
			errs[layer][n] = 0;
			for (int child = 0; child < numN[layer+1]; child++)
				errs[layer][n] += errs[layer+1][child] * wgts[layer][n][child];
		} else {
			errs[layer][n] = (tars[n] - vals[layer][n]);
		}

		errs[layer][n] *= vals[layer][n] * (1 - vals[layer][n]);
		bias[layer][n] += c * errs[layer][n] * bias[layer][n];
	}


	void weights(int web, int child) {

		for (int parent = 0; parent < numN[web]; parent++) {
			chgW[web][parent][child] = (c * errs[web+1][child] * vals[web][parent]) + chgW[web][parent][child] * momentum;
			wgts[web][parent][child] += chgW[web][parent][child];
		}
	}


	void activate(int layer, int n) {

		double net = 0.0;
		for (int parent = 0; parent < numN[layer-1]; parent++)
			net += vals[layer-1][parent] * wgts[layer-1][parent][n];
		net += bias[layer][n];
		vals[layer][n] = 1.0 / (1.0 + exp(-net));
	}


	void createNet(Matrix& features, Matrix& labels) {

		rows = labels.rows();
		c = 0.1;
		numL = 3;				// min of 3 (input, hidden, output)
		momentum = 0.5;			// 0 for iris, 0.5 for vowels
		vs = rows * 0.5;		// 0 for iris, 0.9 for vowels

		numN = new int[numL];
		for (int layer = 0; layer < numL; layer++)
			numN[layer] = layer < numL-1 ? features.cols() : labels.valueCount(0);

		tars = new double[numN[numL-1]];
		filterInputs(numN[0]);

		// OUTPUTS AND ERRORS

		vals = new double*[numL];
		errs = new double*[numL];
		for (int layer = 0; layer < numL; layer++) {
			vals[layer] = new double[numN[layer]];
			errs[layer] = new double[numN[layer]];
		}

		// WEIGHTS

		bestW = new double**[numL-1];
		wgts  = new double**[numL-1];
		chgW  = new double**[numL-1];

		for (int web = 0; web < numL - 1; web++) {

			bestW[web] = new double*[numN[web]];
			wgts[web]  = new double*[numN[web]];
			chgW[web]  = new double*[numN[web]];
			for (int parent = 0; parent < numN[web]; parent++) {

				bestW[web][parent] = new double[numN[web+1]];
				wgts[web][parent]  = new double[numN[web+1]];
				chgW[web][parent]  = new double[numN[web+1]];
				for (int child = 0; child < numN[web+1]; child++) {

					bestW[web][parent][child] = 0;
					chgW[web][parent][child] = 0; 
					wgts[web][parent][child] = randNorm(3);
				}
			}
		}

		// BIASES

		bias = new double*[numL];
		for (int i = 1; i < numL; i++) {
			bias[i] = new double[numN[i]];
			for (int j = 0; j < numN[i]; j++)
				bias[i][j] = (i!=0);
		}
	}


	double randNorm(double span) {

		double sum = 0;
		int max = 999999;
		int iter = 20;

		for (int i = 0; i < iter; i++) sum += rand() % max;

		sum /= iter;
		sum -= max / 2;			// mean of 0
		sum /= (max / span);

		return sum;
	}


	void filterInputs(int numI) {

		inputs = new int[numI];

		for (int i = 0; i < numI; i++)
			inputs[i] = i;
	}


	int findMax() {

		int maxI = -1;
		double maxV = 0;

		for (int n = 0; n < numN[numL-1]; n++) {

			if (maxV < vals[numL - 1][n]) {
				maxV = vals[numL - 1][n];
				maxI = n;
			}
		}

		return maxI;
	}
};

#endif // NN_H