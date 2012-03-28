CC=g++
CPPFLAGS=-O3

all:
	$(CC) $(CPPFLAGS) force.cc

clean:
	rm -f a.out

-include makefile.opt

