# First_Project
My Personal Project, foreground detection?

Suggested git push method:
  - Modify and test in another directory and simply change "testHW_diff_1.py" file (or add a new file by changing the file name) to modified one.

Additional File(Need to be installed):
  - 'average' folder
  - 'NCP' folder
  - 'STD' folder
  - 'MTG' folder
  

Files generated by execution:
  - avg_result%06d.png
  - ncp_result%06d.png
  - mtg_result%06d.png
  - std_result%06d.png
  - average.png
  - Reversed Standard Deviation.png
  - Standard Deviation.png
  - CT_float64.png
  - CT_int8.png
  
 

Baseline(testHW_diff.py)
  - Recall: 44.236%
  - Precision: 90.006%

Latest:
  - 'Minus Average' Approach:
    - Recall: 71.787%
    - Precision: 92.712%
  - 'Minus Average And Standard Deviation' Approach:
    - Recall: 99.993%
    - Precision: 5.948%
  - 'Next - Current - Previous' Approach:
    - Recall: 32.690%
    - Precision: 95.661%
  - 'Basic Baseline (with modified threshold)' Approach:
    - Recall: 62.054%
    - Precision: 61.798%
  - 'Mind The Gap' Approach:
    - Recall: 41.568%
    - Precision: 97.547%
