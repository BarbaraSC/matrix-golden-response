// make opt &&  ../bin/MLSystemManager -L neuralnet -A ../../datasets/vowel.arff -E random 0.75

// Questions
// why when hidden layers are increased does my accuracy significantly get less
// why when momentum is activated is my accuracy significantly less
// understand the graphs I am to make
// why does my stopping criteria effect the output so much when it frequently does the same amount of iterations? (and it seems to take a snapshot of near ~3rd iteration)

#ifndef NEURALNET_H
#define NEURALNET_H

#include "learner.h"
#include "rand.h"
#include "error.h"
// #include <iostream>
// #include <iomanip>
// #include <math.h>
// #include <stdlib.h>
// #include <cmath>

using namespace std;

class NeuralNetLearner : public SupervisedLearner
{
private:
	Rand& m_rand;

	int vs;				// number of filtered inputs
	int numL;			// number of layers
	int rows;
	int * numN;			// number of nodes per layer

	double *** wgts;	// weights
	double *** bestW;	// best weights
	double *** chgW;	// change in weights
	double ** bias;		// biases
	double ** vals;		// output values (input for first layer)
	double ** errs;		// errors

	double c;			// learning rate
	double momentum;	// used in calculating weight change
	double * tars;		// targets

	int * inputs;		// input values

public:
	NeuralNetLearner(Rand& r)
	: SupervisedLearner(), m_rand(r)
	{
	}

	virtual ~NeuralNetLearner()
	{
		for (int web = 0; web < numL - 1; web++) {
			for (int n = 0; n < numN[web]; n++) {
				delete [] chgW [web][n];
				delete [] bestW[web][n];
				delete [] wgts [web][n];
			}

			delete [] bestW[web];
			delete [] chgW [web];
			delete [] wgts [web];
		}

		for (int layer = 0; layer < numL; layer++) {
			delete [] bias[layer];
			delete [] errs[layer];
			delete [] vals[layer];
		}

		delete [] bestW;
		delete [] chgW;
		delete [] wgts;
		delete [] bias;
		delete [] errs;
		delete [] vals;
		delete [] numN;
		delete [] tars;
		delete [] inputs;
	}

	void initData(int oNodes, int ntars, Matrix& features, Matrix& labels) 
	{
		// numI = features.cols(); // how many inputs do I want to capture?
		rows = labels.rows();

		momentum = 0.5;
		// randomlyChooseInputs(features, labels);
		chooseInputs(features.cols());

		vs = rows * 0.5;
		c = 0.1;
		numL = 3;

		numN = new int[numL];
		for (int i = 0; i < numL; i++)
			numN[i] = i < numL-1 ? features.cols() : labels.valueCount(0);

		tars = new double[numN[numL-1]];
	}

	void initbias()
	{
		// setup bias table
		bias = new double*[numL];

		// all bias starts at 1
		for (int i = 0; i < numL; i++)
		{
			bias[i] = new double[numN[i]];
			for (int j = 0; j < numN[i]; j++)
			{
				bias[i][j] = i!=0;
			}
		}
	}

	void initvals()
	{
		// output doesn't matter, just initialize it all
		// the only output that matters is for the input
		vals = new double*[numL];
		errs = new double*[numL];
		for (int i = 0; i < numL; i++) 
		{
			vals[i] = new double[numN[i]];
			errs[i] = new double[numN[i]];
		}
	}

	void initwgts() 
	{
		// setup weight table
		bestW = new double**[numL - 1];
		wgts = new double**[numL - 1];
		chgW = new double**[numL - 1];

		for (int i = 0; i < numL - 1; i++)
		{
			bestW[i] = new double*[numN[i]];
			wgts[i] = new double*[numN[i]];
			chgW[i] = new double*[numN[i]];
			for (int j = 0; j < numN[i]; j++)
			{
				bestW[i][j] = new double[numN[i+1]];
				wgts[i][j] = new double[numN[i+1]];
				chgW[i][j] = new double[numN[i+1]];
				for (int k = 0; k < numN[i+1]; k++)
				{
					bestW[i][j][k] = 0;
					chgW[i][j][k] = 0; 
					wgts[i][j][k] = getEvenlyDistributedNumbersWithMeanAndRange(0, 5);
				}
			}
		}
	}

	void updatewgts()
	{
		for (int i = 0; i < numL - 1; i++)
			for (int j = 0; j < numN[i]; j++)
				for (int k = 0; k < numN[i+1]; k++)
					wgts[i][j][k] += chgW[i][j][k];
	}

	void updatebestW()
	{
		for (int i = 0; i < numL - 1; i++)
			for (int j = 0; j < numN[i]; j++)
				for (int k = 0; k < numN[i+1]; k++)
					bestW[i][j][k] = wgts[i][j][k];
	}

	void setupNodes(){
		initwgts();
		initbias();
		initvals();
	}

	// double * getChildWeight(int c, int r, int i)
	// {
	// 	if (c == numL - 1)
	// 	{
	// 		cout << "error, output doesn't have a child\n";
	// 		return NULL;
	// 	}
	// 	return &wgts[c][r][i];
	// }

	// double * getParentWeight(int c, int r, int i)
	// {
	// 	if (c == 0)
	// 	{
	// 		cout << "error, input doesn't have a parent weight\n";
	// 		return NULL;
	// 	}
	// 	return &wgts[c-1][i][r];
	// }

	// double * getBestParentWeight(int c, int r, int i)
	// {
	// 	if (c == 0)
	// 	{
	// 		cout << "error, input doesn't have a parent weight\n";
	// 		return NULL;
	// 	}
	// 	return &bestW[c-1][i][r];
	// }

	// double * getBias(int c, int r)
	// {
	// 	if (c == 0)
	// 	{
	// 		cout << "error, input doesn't have a bias weight\n";
	// 		return NULL;
	// 	}
	// 	return &bias[c][r];
	// }

	// double * getParentOutput(int c, int i)
	// {
	// 	if (c == 0)
	// 	{
	// 		cout << "error, input shouldn't have a parent\n";
	// 		return NULL;
	// 	}
	// 	return &vals[c-1][i];
	// }

	void chooseInputs(int numI)
	{
		inputs = new int[numI];

		for (int i = 0; i < numI; i++)
			inputs[i] = i;
	}

	void calcAndUpdateOutput(int column, int row)
	{
		if (column == 0)
		{
			cout << "error, should not calculate output on input node\n";
			return;
		}
		double net = 0.0;
		for (int i = 0; i < numN[column-1]; i++)
		{
			// net += *(getParentOutput(column, i)) * *(getParentWeight(column, row, i));
			net += vals[column-1][i] * wgts[column-1][i][row];
		}
		net += bias[column][row];

		vals[column][row] = 1.0/(1.0 + exp(-net));
	}

	void calcAndUpdateBestOutput(int column, int row)
	{
		if (column == 0)
		{
			cout << "error, should not calculate output on input node\n";
			return;
		}
		double net = 0.0;
		for (int i = 0; i < numN[column-1]; i++)
		{
			net += vals[column-1][i] * bestW[column-1][i][row];
		}
		net += bias[column][row];

		vals[column][row] = 1.0/(1.0 + exp(-net));
	}

	void runEpochs(Matrix& features, Matrix& labels)
	{
		// static int startingEpochCount = 1000;
		static int badCountStoppingCriteria = 100;
		int epochCount = 1000;
		int alreadyHere = 0;

		// double smallestError;
		// double currentError;
		int numEpochs = 0;

		int worseXTimesInARow = 0;
		int bestGoodCount = 0;

		while (epochCount-- && worseXTimesInARow != badCountStoppingCriteria)
		{
			numEpochs++;
			// currentError = 0;

			// scramble input order
			features.shuffleRows(m_rand, &labels);

			// loop through all inputs (not nodes)

			for (int i = 0; i < rows - vs; i++) 
			{
				resetTargetAndInput(features, labels, i);

				// forward propogate
				for (int j = 1; j < numL; j++)
					for (int k = 0; k < numN[j]; k++)
						calcAndUpdateOutput(j, k);
				
				// backward propogate
				// output error
				int j = numL - 1;
				for (int k = 0; k < numN[j]; k++)
				{

					errs[j][k] = (tars[k] - vals[j][k]) * vals[j][k] * (1 - vals[j][k]); // (target - output) * output * (1 - output);
					for (int l = 0; l < numN[j - 1]; l++)
					{
						chgW[j-1][l][k] = (c * errs[j][k] * vals[j-1][l]) + chgW[j-1][l][k] * momentum;
						wgts[j-1][l][k] += chgW[j-1][l][k];
					}
					bias[j][k] += c * errs[j][k] * bias[j][k];
				}

				// hidden error
				for (j = numL - 2; j > 0; j--)
				{
					for (int k = 0; k < numN[j]; k++)
					{
						double errorWeightSum = 0;
						for (int l = 0; l < numN[j + 1]; l++)
						{
							errorWeightSum += errs[j + 1][l] * wgts[j][k][l];
						}

						errs[j][k] = vals[j][k] * (1 - vals[j][k]) * errorWeightSum;

						for (int l = 0; l < numN[j - 1]; l++)
						{
							double cw = (c * errs[j][k] * vals[j-1][l]);
							double mCW = chgW[j-1][l][k] * momentum;
							chgW[j-1][l][k] = cw + mCW;
							wgts[j-1][l][k] += chgW[j-1][l][k];
						}

						bias[j][k] += c * errs[j][k] * bias[j][k];
					}
				}
				// after done looping (reached input) update all weights (rather than as I go)
				// updatewgts();
			}

			// loop through validation set
			int goodCount = 0;
			for (int i = rows - vs; i < rows; i++)
			{
				resetTargetAndInput(features, labels, i);

				// forward propogate
				for (int j = 1; j < numL; j++)
				{
					for (int k = 0; k < numN[j]; k++)
					{
						calcAndUpdateOutput(j, k);
					}
				}

				double greatestOutput = 0;
				int best;
				for (int j = 0; j < numN[numL-1]; j++)
				{
					// cout << vals[numL - 1][i] << "\t";
					if (greatestOutput < vals[numL - 1][j])
					{
						greatestOutput = vals[numL - 1][j];
						best = j;
					}
				}
				// cout << best << " " << tars[best] << " " << greatestOutput << endl;
				if (tars[best])
					goodCount++;
			}
			// cout << "goodCount: " << goodCount << endl;

			// given good VS count...

			if (goodCount == bestGoodCount)
			{
				updatebestW();
				alreadyHere++;
			}

			if (goodCount > bestGoodCount)
			{
				updatebestW();
				bestGoodCount = goodCount;
				worseXTimesInARow = 0;
				alreadyHere = 0;
			} else {
				alreadyHere = false;
				worseXTimesInARow += 1 + alreadyHere;
			}

			// cout << "current error: " << currentError << endl;
			// if (epochCount == startingEpochCount - 1 || smallestError > currentError)
			// {
			// 	smallestError = currentError;
			// 	// cout << "epoch # " << numEpochs << " ";
			// 	updatebestW();
			// 	worseXTimesInARow = 0;
			// } else {
			// 	worseXTimesInARow++;
			// }
		}
		// display();

		cout << "\nNumber of epochs: " << numEpochs << "\n";
	}

	void resetTargetAndInput(Matrix& features, Matrix& labels, int row)
	{
		// reset input node's outputs
		for (int i = 0; i < numN[0]; i++)
		{
			vals[0][i] = features[row][inputs[i]];	
		}

		// reset output nodes tars
		for (int i = 0; i < numN[numL-1]; i++)
		{
			tars[i] = 0;
		}
		tars[int(labels[row][0])] = 1;
	}

	// Train the model to predict the labels
	virtual void train(Matrix& features, Matrix& labels)
	{
		if(features.rows() != labels.rows())
			ThrowError("Expected the features and labels to have the same number of rows");

		initData(labels.valueCount(0), labels.valueCount(0), features, labels);
		setupNodes();
		runEpochs(features, labels);
	}

	double getEvenlyDistributedNumbersWithMeanAndRange(double mean, double higherAndLowerBound) 
	{
		int numRanToSum = 20;
		int maxNum = 1000000;
		double sum = 0;
		for (int i = 0; i < numRanToSum; i++)
			sum += rand() % maxNum;
		sum /= numRanToSum;

		// make sum around zero
		sum -= maxNum / 2;

		sum /= (maxNum / higherAndLowerBound);

		// make sum around mean
		sum += mean;
		return sum;
	}

	// Evaluate the features and predict the labels
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels)
	{
		// reset input node's outputs
		for (int i = 0; i < numN[0]; i++)
			vals[0][i] = features[inputs[i]];

		// forward
		for (int j = 1; j < numL; j++)
			for (int k = 0; k < numN[j]; k++)
				calcAndUpdateOutput(j, k);

		double greatestOutput = 0;
		for (int i = 0; i < numN[numL-1]; i++)
		{
			if (greatestOutput < vals[numL - 1][i])
			{
				greatestOutput = vals[numL - 1][i];
				labels[0] = i;
			}
		}
	}
};


#endif // NEURALNET_H
