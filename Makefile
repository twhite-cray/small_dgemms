ccomp?=cc
cflags=-fopenmp -Ofast
cexec=many-small-dgemms.x
csources=many-small-dgemms.c
cobjects=${patsubst %.c,%.c.obj, $(csources)}
clibs=-L $(LIBRARY_PATH) -lmagma -lcublas -lcudart -lopenblas

main: $(cexec)

$(cexec): $(cobjects)
	$(ccomp) $(cflags) -g $(clibs) -o $(cexec) $(cobjects)

%.c.obj: %.c
	$(ccomp) $(cflags) -g -c -o $@ $<

clean:
	rm -f *.o *.c.obj $(cexec)

