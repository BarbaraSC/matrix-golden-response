../obj/opt/filter.o: filter.cpp filter.h matrix.h learner.h rand.h error.h
	g++ -Wall -O3 -c filter.cpp -o ../obj/opt/filter.o
