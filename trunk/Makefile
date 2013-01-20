all:
	g++ -ggdb `pkg-config --cflags opencv` -o `basename reconocer.cpp .cpp` reconocer.cpp `pkg-config --libs opencv`
	g++ -ggdb `pkg-config --cflags opencv` -o `basename entrenar.cpp .cpp` entrenar.cpp `pkg-config --libs opencv`
