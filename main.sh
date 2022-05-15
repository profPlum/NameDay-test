#!/bin/bash

#conda env create -f NameDay.yml
#conda activate NameDay

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
#python write_vocab_data.py
#Rscript -e "install.packages(c('tidyverse', 'tm','tokenizers','moments','qdapDictionaries'), dependencies=T)"
Rscript ./R_scripts/make_survey.R
mv Rplots.pdf make_survey_plots.pdf
Rscript R_scripts/survey_data_analysis.R
mv Rplots.pdf survey_analysis_plots.pdf
Rscript R_scripts/dataset_demo.R
mv Rplots.pdf dataset_demo_plots.pdf
Rscript R_scripts/inspect_correlation_test.R
mv Rplots.pdf correlation_test_plots.pdf
cd -

#conda deactivate
