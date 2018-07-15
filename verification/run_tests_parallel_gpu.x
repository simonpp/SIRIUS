#!/bin/bash

if [ -z "$SIRIUS_BINARIES" ];
then
    export SIRIUS_BINARIES=../../apps/dft_loop
fi

exe=${SIRIUS_BINARIES}/sirius.scf
# check if path is correct
type -f ${exe} || exit 1

for f in ./*; do
    if [ -d "$f" ]; then
        echo "running '${f}'"
        (
            cd ${f}
            mpirun -np 4 ${exe} \
                   --test_against=output_ref.json \
                   --std_evp_solver_name=scalapack \
                   --gen_evp_solver_name=scalapack \
                   --processing_unit=gpu --mpi_grid="2 2"
            err=$?

            if [ ${err} == 0 ]; then
                echo "OK"
            else
                echo "'${f}' failed"
                exit ${err}
            fi
        )
    fi
done

echo "All tests were passed correctly!"
