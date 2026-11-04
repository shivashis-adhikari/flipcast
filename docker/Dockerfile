# PostgreSQL 18 with the flipcast_sizes module, for the shadow server.
# pg_stat_statements is preloaded so the same image works as a demo source.

FROM postgres:18 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential postgresql-server-dev-18 \
 && rm -rf /var/lib/apt/lists/*
COPY ext/flipcast_sizes /src/flipcast_sizes
# with_llvm=no: skip JIT bitcode, it would pull in clang for nothing
RUN make -C /src/flipcast_sizes with_llvm=no \
 && make -C /src/flipcast_sizes with_llvm=no install DESTDIR=/out

FROM postgres:18
COPY --from=build /out/ /
CMD ["postgres", "-c", "shared_preload_libraries=pg_stat_statements", \
     "-c", "pg_stat_statements.track=all"]
