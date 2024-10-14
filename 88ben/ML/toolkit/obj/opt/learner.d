../obj/opt/learner.o: learner.cpp learner.h matrix.h rand.h error.h
	g++ -Wall -O3 -c learner.cpp -o ../obj/opt/learner.o
