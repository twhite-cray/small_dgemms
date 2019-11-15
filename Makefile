ccomp?=cc
cflags=-fopenmp -Ofast
cexec=many-small-dgemms.x
csources=many-small-dgemms.c
cobjects=${patsubst %.c,%.c.obj, $(csources)}
clibs=-L $(LIBRARY_PATH) -lmagma -lcublas -lcudart -lopenblas -lstdc++ -lcusparse

main: $(cexec)

$(cexec): $(cobjects)
	$(ccomp) $(cflags) -g -o $(cexec) $(cobjects) $(clibs)

%.c.obj: %.c
	$(ccomp) $(cflags) -g -c -o $@ $<

clean:
	rm -f *.o *.c.obj $(cexec)

