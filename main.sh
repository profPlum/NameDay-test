#!/bin/bash

#conda env create -f NameDay.yml
#conda activate NameDay

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
python write_vocab_data.py
#Rscript -e "install.packages(c('tidyverse', 'tm','tokenizers','moments','qdapDictionaries'), dependencies=T)"
Rscript ./R_scripts/make_survey.R
Rscript R_scripts/survey_data_analysis.R
Rscript R_scripts/dataset_demo.R
Rscript R_scripts/inspect_correlation_test.R
cd -

#conda deactivate
