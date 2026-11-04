MODULE_big = flipcast_sizes
OBJS = flipcast_sizes.o
PGFILEDESC = "flipcast_sizes - planner uses catalog relation sizes"

EXTENSION = flipcast_sizes
DATA = flipcast_sizes--0.3.sql

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
