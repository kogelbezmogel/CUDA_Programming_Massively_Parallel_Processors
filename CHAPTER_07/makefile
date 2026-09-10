CC = nvcc
OBJECTS = main_convolution2D.o convolution2D_const_memory.o convolution2D_tiled_v1.o convolution2D_tiled_v2.o convolution2D_simple.o
BIN = main_convolution2D.out

all:$(BIN)

$(BIN):$(OBJECTS)
	$(CC) -std=c++11 -o $(BIN) $(OBJECTS)

%.o:%.cu
	$(CC) -std=c++11 -c -o $@ $^

clean:
	rm -rf $(BIN) $(OBJECTS)