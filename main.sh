#!/bin/bash

# cd to this script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

# condo has interfered with this before
conda deactivate
conda deactivate

pip install -r requirements.txt
python write_vocab_data.py
#Rscript -e "install.packages(c('tidyverse', 'tm','tokenizers','qdapDictionaries'),
            repos='https://cran.rstudio.com/')"
echo conducting correlation test:
Rscript R_scripts/inspect_correlation_test.R
mv Rplots.pdf correlation_test_plots.pdf
echo Inspecting Hard Max Interpretability & making sample survey:
Rscript ./R_scripts/make_survey.R
mv Rplots.pdf Hard_Max_Interpretability_inspection_plots.pdf
echo analyzing real survey data:
Rscript R_scripts/survey_data_analysis.R
mv Rplots.pdf survey_analysis_plots.pdf
echo running dataset simplification & visualization demo:
Rscript R_scripts/dataset_demo.R
mv Rplots.pdf dataset_simplification_demo_plots.pdf
mv 'logs/true --> soft names plot.png' .
cd -

echo See plot pdf files for figures.
