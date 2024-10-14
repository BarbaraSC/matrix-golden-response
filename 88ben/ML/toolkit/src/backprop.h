// ----------------------------------------------------------------
// Benjamin Walker
// clear;clear && make opt &&  ../bin/MLSystemManager -L backPropogation -A ../../BP/vowel.arff -E random 0.75
// ----------------------------------------------------------------



// Questions
// why when hidden layers are increased does my accuracy significantly get less
// why when momentum is activated is my accuracy significantly less
// understand the graphs I am to make
// why does my stopping criteria effect the output so much when it frequently does the same amount of iterations? (and it seems to take a snapshot of near ~3rd iteration)

#ifndef BACKPROPOGATION_H
#define BACKPROPOGATION_H

// #define DEBUG

#include "learner.h"
#include "rand.h"
#include "error.h"
#include <iostream>
#include <iomanip>
#include <math.h> // exp
#include <stdlib.h>
#include <cmath>

using namespace std;

class BackPropogationLearner : public SupervisedLearner
{
private:
	Rand& m_rand;
	double momentum;

	int numInputNodes;
	int numLayers;
	int * numNodesPerLayer;
	int numOutputNodes;

	double *** weightTable;
	double *** bestWeightTable;
	double *** changeWeightTable;
	double ** biasTable;
	double ** outputTable;
	double ** errorTable;

	int numValidateItems;

	double learningRate;
	int numTargets;
	double * targets;

	// int numInputNodes; // same as numInputNodes
	int * inputIndices;

public:
	BackPropogationLearner(Rand& r)
	: SupervisedLearner(), m_rand(r)
	{
	}

	virtual ~BackPropogationLearner()
	{
		for (int i = 0; i < numLayers - 1; i++)
		{
			for (int j = 0; j < numNodesPerLayer[i]; i++){
				// why?
				if (j != 0)
				{
					delete [] changeWeightTable[i][j];
					delete [] bestWeightTable[i][j];
				}
				delete [] weightTable[i][j];
			}
			delete [] bestWeightTable[i];
			delete [] changeWeightTable[i];
			delete [] weightTable[i];
		}
		delete [] bestWeightTable;
		delete [] changeWeightTable;
		delete [] weightTable;

		for (int i = 0; i < numLayers; i++)
			delete [] biasTable[i];
		delete [] biasTable;

		for (int i = 0; i < numLayers; i++)
			delete [] errorTable[i];
		delete [] errorTable;

		for (int i = 0; i < numLayers; i++)
			delete [] outputTable[i];
		delete [] outputTable;

		delete [] numNodesPerLayer;
		delete [] targets;
		delete [] inputIndices;
	}

	void initData(int oNodes, int nTargets, Matrix& features, Matrix& labels) 
	{
		numOutputNodes = oNodes;
		numInputNodes = 20; // how many inputs do I want to capture? 

		momentum = 0.5;
		randomlyChooseInputs(features, labels);
		// chooseInputs(features, labels);

		numTargets = nTargets;

		double percentOfTestingSetIsValidate = 0.70;
		numValidateItems = labels.rows() * percentOfTestingSetIsValidate;

		learningRate = 0.1;

		targets = new double[numOutputNodes];

		// must exist at least 1 hidden layer
		int numHiddenLayers = 1;
		numLayers = numHiddenLayers + 2;

		int numberNodesPerHiddenLayer = numInputNodes;
		numNodesPerLayer = new int[numLayers];
		numNodesPerLayer[0] = numInputNodes;
		for (int i = 1; i < numLayers - 1; i++)
		{
			numNodesPerLayer[i] = numberNodesPerHiddenLayer; 
		}
		numNodesPerLayer[numLayers - 1] = numOutputNodes;

	}

	void initBiasTable()
	{
		// setup bias table
		biasTable = new double*[numLayers];

		// input should have no bias, so set to zero just in case
		biasTable[0] = new double[numNodesPerLayer[0]];
		for (int j = 0; j < numNodesPerLayer[0]; j++)
		{
			biasTable[0][j] = 0;
		}

		// all bias starts at 1
		for (int i = 1; i < numLayers; i++)
		{
			biasTable[i] = new double[numNodesPerLayer[i]];
			for (int j = 0; j < numNodesPerLayer[i]; j++)
			{
				biasTable[i][j] = 1;
			}
		}
	}

	void initOutputTable()
	{
		// output doesn't matter, just initialize it all
		// the only output that matters is for the input
		outputTable = new double*[numLayers];
		errorTable = new double*[numLayers];
		for (int i = 0; i < numLayers; i++) 
		{
			outputTable[i] = new double[numNodesPerLayer[i]];
			errorTable[i] = new double[numNodesPerLayer[i]];
		}
	}

	void initWeightTable() 
	{
		// setup weight table
		bestWeightTable = new double**[numLayers - 1];
		weightTable = new double**[numLayers - 1];
		changeWeightTable = new double**[numLayers - 1];

		for (int i = 0; i < numLayers - 1; i++)
		{
			bestWeightTable[i] = new double*[numNodesPerLayer[i]];
			weightTable[i] = new double*[numNodesPerLayer[i]];
			changeWeightTable[i] = new double*[numNodesPerLayer[i]];
			for (int j = 0; j < numNodesPerLayer[i]; j++)
			{
				bestWeightTable[i][j] = new double[numNodesPerLayer[i+1]];
				weightTable[i][j] = new double[numNodesPerLayer[i+1]];
				changeWeightTable[i][j] = new double[numNodesPerLayer[i+1]];
				for (int k = 0; k < numNodesPerLayer[i+1]; k++)
				{
					bestWeightTable[i][j][k] = 0;
					changeWeightTable[i][j][k] = 0; 
					weightTable[i][j][k] = getEvenlyDistributedNumbersWithMeanAndRange(0, 2);
				}
			}
		}
	}

	void updateWeightTable()
	{
		for (int i = 0; i < numLayers - 1; i++)
		{
			for (int j = 0; j < numNodesPerLayer[i]; j++)
			{
				for (int k = 0; k < numNodesPerLayer[i+1]; k++)
				{
					weightTable[i][j][k] += changeWeightTable[i][j][k];
				}
			}
		}
	}

	void updateBestWeightTable()
	{
		for (int i = 0; i < numLayers - 1; i++)
			for (int j = 0; j < numNodesPerLayer[i]; j++)
				for (int k = 0; k < numNodesPerLayer[i+1]; k++) 
				{
					bestWeightTable[i][j][k] = weightTable[i][j][k];
				}
	}

	void setupNodes(){
		initWeightTable();
		initBiasTable();
		initOutputTable();
	}

	double * getChildWeight(int c, int r, int i)
	{
		if (c == numLayers - 1)
		{
			cout << "error, output doesn't have a child\n";
			return NULL;
		}
		return &weightTable[c][r][i];
	}

	double * getParentWeight(int c, int r, int i)
	{
		if (c == 0)
		{
			cout << "error, input doesn't have a parent weight\n";
			return NULL;
		}
		return &weightTable[c-1][i][r];
	}

	double * getBestParentWeight(int c, int r, int i)
	{
		if (c == 0)
		{
			cout << "error, input doesn't have a parent weight\n";
			return NULL;
		}
		return &bestWeightTable[c-1][i][r];
	}

	double * getBias(int c, int r)
	{
		if (c == 0)
		{
			cout << "error, input doesn't have a bias weight\n";
			return NULL;
		}
		return &biasTable[c][r];
	}

	double * getParentOutput(int c, int i)
	{
		if (c == 0)
		{
			cout << "error, input shouldn't have a parent\n";
			return NULL;
		}
		return &outputTable[c-1][i];
	}

	void display() 
	{
		// display the input/output, layer, node number for the layer, incoming weight, and output
		for (int i = 0; i < numLayers; i++)
		{
			for (int j = 0; j < numNodesPerLayer[i]; j++)
			{
				if (i == 0)
					cout << "Input  ";
				else if (i == numLayers - 1)
					cout << "Output ";
				else
					cout << "Hidden ";
				cout << setw(15) << "layer " << i
					 << setw(15) << "node #: " << j
					 << setw(15) << "output " << outputTable[i][j];
				
				if (i != 0)
					cout << setw(15) << "bias " << *(getBias(i,j));

				if (i == numLayers - 1)
					cout << setw(15) << "target: " << targets[j];
					 
				cout << "\n";
				if (i != 0) {
					for (int k = 0; k < numNodesPerLayer[i-1]; k++)
					{
						cout << "\t\tparentList\t" << i << j << k << " " << *(getParentWeight(i,j,k)) << "\tparentOutput: " << *(getParentOutput(i,k)) << "\n"; 
					}
					cout << endl;
				}
				if (i != numLayers - 1) {
					for (int k = 0; k < numNodesPerLayer[i+1]; k++)
					{
						cout << "\t\tchildList\t" << i << j << k << " " << *(getChildWeight(i,j,k))  << endl; 
					}
				}
			}
		}
	}

	void chooseInputs(Matrix& features, Matrix& labels)
	{
		numInputNodes = 4;
		inputIndices = new int[numInputNodes];

		inputIndices[0] = 0;
		inputIndices[1] = 1;
		inputIndices[2] = 3;
		inputIndices[3] = 7;
	}

	void randomlyChooseInputs(Matrix& features, Matrix& labels)
	{
		int values = features.cols();

		if (numInputNodes > values || numInputNodes <= 0)
		{
			cout << "can't pick number of items out of range than there actually are, changed to " << values << "\n";
			numInputNodes = values;	
		}

		inputIndices = new int[numInputNodes];
		for (int i = 0; i < numInputNodes; i++)
		{
			int temp = m_rand.next();
			if (temp < 0)
				temp *= -1;
			int newRand = temp % values;
			bool contains = false;

			// if (newRand == 1)
			// {
			// 	contains = true;
			// }

			for (int j = 0; j < i; j++)
				if (inputIndices[j] == newRand)
					contains = true;
			if (!contains)
				inputIndices[i] = newRand;
			else
				i--;
		}

		cout << "\nHere are the indicies chosen:\n";
		for (int i = 0; i < numInputNodes; i++)
		{
			cout << inputIndices[i] << ", ";
		}
		cout << endl;

		// for (int j = 0; j < labels.rows(); j++) 
		// {
		// 	cout << "Inputs: ";
		// 	for (int k = 0; k < numInputNodes; k++) 
		// 	{
		// 		cout << features[j][inputIndices[k]] << ", ";
		// 	}
		// 	cout << "target: " << labels[j][0] << endl;
		// }
	}

	void calcAndUpdateOutput(int column, int row)
	{
		if (column == 0)
		{
			cout << "error, should not calculate output on input node\n";
			return;
		}
		double net = 0.0;
		for (int i = 0; i < numNodesPerLayer[column-1]; i++)
		{
			net += *(getParentOutput(column, i)) * *(getParentWeight(column, row, i));
		}
		net += *(getBias(column, row));

		outputTable[column][row] = 1.0/(1.0 + exp(-net));
	}

	void calcAndUpdateBestOutput(int column, int row)
	{
		if (column == 0)
		{
			cout << "error, should not calculate output on input node\n";
			return;
		}
		double net = 0.0;
		for (int i = 0; i < numNodesPerLayer[column-1]; i++)
		{
			net += *(getParentOutput(column, i)) * *(getBestParentWeight(column, row, i));
		}
		net += *(getBias(column, row));

		outputTable[column][row] = 1.0/(1.0 + exp(-net));
	}

	void runEpochs(Matrix& features, Matrix& labels)
	{
		static int startingEpochCount = 1000;
		static int badCountStoppingCriteria = 100;
		int epochCount = startingEpochCount;
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

			for (unsigned int i = 0; i < labels.rows() - numValidateItems; i++) 
			{
				resetTargetAndInput(features, labels, i);

				// forward propogate
				for (int j = 1; j < numLayers; j++)
				{
					for (int k = 0; k < numNodesPerLayer[j]; k++)
					{
						calcAndUpdateOutput(j, k);
					}
				}
				
				// backward propogate
				// output error
				int j = numLayers - 1;
				for (int k = 0; k < numNodesPerLayer[j]; k++)
				{

					errorTable[j][k] = (targets[k] - outputTable[j][k]) * outputTable[j][k] * (1 - outputTable[j][k]); // (target - output) * output * (1 - output);
					// currentError += (errorTable[j][k] > 0) ? errorTable[j][k] : -errorTable[j][k];
					for (int l = 0; l < numNodesPerLayer[j - 1]; l++)
					{
						double cw = (learningRate * errorTable[j][k] * *(getParentOutput(j, l)));
						double mCW = changeWeightTable[j-1][l][k] * momentum;
						changeWeightTable[j-1][l][k] = cw + mCW;
						// cout << changeWeightTable[j-1][l][k] << endl;
					}
					// update bias
					biasTable[j][k] += learningRate * errorTable[j][k] * biasTable[j][k];
				}

				// hidden error
				for (j = numLayers - 2; j > 0; j--)
				{
					for (int k = 0; k < numNodesPerLayer[j]; k++)
					{
						double errorWeightSum = 0;
						for (int l = 0; l < numNodesPerLayer[j + 1]; l++)
						{
							errorWeightSum += errorTable[j + 1][l] * *(getChildWeight(j,k,l));
						}

						errorTable[j][k] = outputTable[j][k] * (1 - outputTable[j][k]) * errorWeightSum;
						// currentError += (errorTable[j][k] > 0) ? errorTable[j][k] : -errorTable[j][k];

						for (int l = 0; l < numNodesPerLayer[j - 1]; l++)
						{
							double cw = (learningRate * errorTable[j][k] * *(getParentOutput(j, l)));
							double mCW = changeWeightTable[j-1][l][k] * momentum;
							changeWeightTable[j-1][l][k] = cw + mCW;
							// cout << changeWeightTable[j-1][l][k] << endl;
						}

						biasTable[j][k] += learningRate * errorTable[j][k] * biasTable[j][k];
					}
				}
				// cout << "Epoch: " << numEpochs << endl;

				// after done looping (reached input) update all weights (rather than as I go)
				updateWeightTable();
			}

			// loop through validation set
			int goodCount = 0;
			for (unsigned int i = labels.rows() - numValidateItems; i < labels.rows(); i++)
			{
				resetTargetAndInput(features, labels, i);

				// forward propogate
				for (int j = 1; j < numLayers; j++)
				{
					for (int k = 0; k < numNodesPerLayer[j]; k++)
					{
						calcAndUpdateOutput(j, k);
					}
				}

				double greatestOutput = 0;
				int best;
				for (int j = 0; j < numOutputNodes; j++)
				{
					// cout << outputTable[numLayers - 1][i] << "\t";
					if (greatestOutput < outputTable[numLayers - 1][j])
					{
						greatestOutput = outputTable[numLayers - 1][j];
						best = j;
					}
				}
				// cout << best << " " << targets[best] << " " << greatestOutput << endl;
				if (targets[best])
					goodCount++;
			}
			// cout << "goodCount: " << goodCount << endl;

			// given good VS count...

			if (goodCount == bestGoodCount)
			{
				cout << "1" << endl;
				updateBestWeightTable();
				alreadyHere++;
			}

			if (goodCount > bestGoodCount)
			{
				cout << "2" << endl;
				updateBestWeightTable();
				bestGoodCount = goodCount;
				worseXTimesInARow = 0;
				alreadyHere = 0;
			} else {
				cout << "3" << endl;
				alreadyHere = false;
				worseXTimesInARow += 1 + alreadyHere;
			}

			// cout << "current error: " << currentError << endl;
			// if (epochCount == startingEpochCount - 1 || smallestError > currentError)
			// {
			// 	smallestError = currentError;
			// 	// cout << "epoch # " << numEpochs << " ";
			// 	updateBestWeightTable();
			// 	worseXTimesInARow = 0;
			// } else {
			// 	worseXTimesInARow++;
			// }
		}
		// display();

		cout << "\n\n\nNumber of epochs ran: " << numEpochs << endl << endl;
	}

	void resetTargetAndInput(Matrix& features, Matrix& labels, int row)
	{
		// reset input node's outputs
		for (int i = 0; i < numInputNodes; i++)
		{
			outputTable[0][i] = features[row][inputIndices[i]];	
		}

		// reset output nodes targets
		for (int i = 0; i < numTargets; i++)
		{
			targets[i] = 0;
		}
		targets[int(labels[row][0])] = 1;
	}

	// Train the model to predict the labels
	virtual void train(Matrix& features, Matrix& labels)
	{
		// Check assumptions
		if(features.rows() != labels.rows())
			ThrowError("Expected the features and labels to have the same number of rows");

		initData(labels.valueCount(0), labels.valueCount(0), features, labels);
	
		cout << "\nnumer of output nodes: " << numOutputNodes << endl;
		cout << "numer of input nodes:  " << numInputNodes << endl; 

		
		cout << "\n\nstarted setupNodes\n";
		setupNodes();

		cout << "\n\nstarted epoch run\n\n";
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
		for (int i = 0; i < numInputNodes; i++)
		{
			outputTable[0][i] = features[inputIndices[i]];	
		}

		// forward propogate
		for (int j = 1; j < numLayers; j++)
		{
			for (int k = 0; k < numNodesPerLayer[j]; k++)
			{
				calcAndUpdateOutput(j, k);
			}
		}

		double greatestOutput = 0;
		// cout << "\n";
		for (int i = 0; i < numOutputNodes; i++)
		{
			// cout << outputTable[numLayers - 1][i] << "\t";
			if (greatestOutput < outputTable[numLayers - 1][i])
			{
				greatestOutput = outputTable[numLayers - 1][i];
				labels[0] = i;
				// cout << " " << i << "\n";
			}
		}
		// cout << "\n" << labels[0] << endl;

	}
};


#endif // BACKPROPOGATION_H