../obj/dbg/main.o: main.cpp learner.h matrix.h rand.h baseline.h error.h backprop.h \
 nn.h filter.h
	g++ -Wall -g -D_DEBUG -c main.cpp -o ../obj/dbg/main.o
