#!/bin/bash

rm path*

cd batch_variables_com
rm *

cd ..

cd batch_variables
rm *

cd ../../
# go to Dataset
cd Dataset/Linux/Client/

cd Client_structured_com
rm log_value_vector.csv
rm Path*
cd ..
cd Client_structured
rm Path*
rm log_value*

exit 0
