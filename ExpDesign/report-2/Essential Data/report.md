# Experiment Design in Computer Science
# Second Report

## Report Summary:

In this report, you will perform an experiment comparing the performance of different configurations of the Differential Evolution (DE) algorithm. 

You will then analyse the data collected from the experiment using the statistical techniques discussed in class, and explain the results of the experiment and analysis, using calculations and figures as necessary.

## General Description of the Report

Differential Evolution (DE) is a Meta-Heuristic Optimization algorithm. The performance of DE is dependent on its parameters, including the `Selection Policy`, the Differential Weight (F), and the Crossover Probability (Cr).

Your first objective is to perform an experiment to compare the effect of three different Selection Policies on the performance of DE: 
- Policy 1: "Select Best", 
- Policy 2: "Select Random", 
- Policy 3: "Select Random-to-Best". 

Your second (extra) objective is to compare the policies again, after performing a fine-tuning of the parameters F and Cr for each policy.

To evaluate the different DE configurations, you are provided with a script that runs each policy on a set of benchmark problems (from the CEC 2014 optimization benchmark set). You can select the number of benchmark functions to be used, and the number of repetitions to be done on each benchmark function.

The output variable of the experiment is optimum (minimum) vale found by running one algorithm configuration on one benchmark instance.

To successfully complete the experiment, you must provide a comprehensive analysis of the results, including estimating the variance of the experiment, choosing the appropriate statistical test and sample size, checking the assumptions of the test, and providing extra figures and analysis as necessary.

Expected parameters for the statistical analysis: 95% confidence, 80% power, relative difference of 5% on the effect size.

The report must be a PDF file that describes the experiment setup, results and conclusion, as well as any programs that were created and modified for the experiment. 

There is no maximum or minimum number of pages, but the students should prefer brevity.

## Grading Of the Report

- 0-75 points: First Objective (comparison of 3 selection policies)
  - The student collected the necessary data to achieve the first objective of the experiment.
  - The student presented this data in the report using appropriate graphs and tables.
  - The appropriate statistical procedure to analyze this data was chosen and applied.
  - The assumptions of the statistical procedure were validated.
  - Report includes appropriate conclusions that follow correctly from the experiment data, results of the statistical analysis, and the statistical setup (power, confidence, and effect).

- 0-15 points: Extra analysis beyond the minimum necessary for the first objective.
  - Sample size calculation and power analysis
  - Analysis of outliers
  - Runtime analysis of the optimization algorithms

- 0-10 points: Second Objective (Finetuning of F and Cr)
  - Extra necessary experiments were organized for fine tuning F and Cr for each of the three policy variants, following the General Description of the report.

## Technical Guidelines

The students are provided with a python script (`base_script.py`) that execute the three variants of DE in a set of benchmark problems. This python script requires the installation of the `pygmo` library, which is available through pip. Running the script produces two CSV files:   

The first file (resultbest_<date>.csv) provides the value of the output variable for each algorithm/problem run, and should be the basic file used for the analysis in this report. 

The second file (resultall_<date>.csv) provides the same information as the first one, except that the output variable is provided for every 100 time steps whithin each algorithm run. This data can be used for runtime analysis.

You might need to run this script several times with different parameter values to achieve the objectives of this report. Making this judgement is part of the report's requirements.

It is not necessary for the students to change this script except for the parts indicated inside the file. However, it is acceptable for the students to change other parts of the script. These extra changes are the responsibility of the student, and if non-trivial should be explained in the report.
