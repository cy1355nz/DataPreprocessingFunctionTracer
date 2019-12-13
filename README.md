# Team 6: Fairness-Aware Instrumentation of ML Pipelines
Fairness is becoming one of the most popular topics in machine learning community. While most of related researches focus on fairness in algorithms and modeling, our paper digs into fairness issues arise during data preparation stage. To enhance fairness-awareness in preprocessing, the paper proposes a Data Preprocessing Function Tracer (DPF Tracer) to help users track changes in user-defined features for protected groups in an automated way. DPF Tracer produces two forms of outputs 1) a DAG visualization of data pipelines and 2) logs on related feature changes after each operation. We also demonstrate a few use cases and results after using DPF Tracer on four fairness-related datasets. In the end, we discuss about the benefits and drawbacks of our tool and list some future works.  

## Git Repo
- team6_instrumentation.pdf is the final summary paper. 
- fairness_instru.ipynb includes key functions and outputs from six cases we designed. DAG is not included in the notebook. When tracer function is called, a DAG will be generated and saved in current directory.
- utils.py contains all the support functions.
- Graphs has the collection of codes, DAG and output logs.  
- slides.pdf is the presentation slides.  

## Update Logs  
2019-11-17 Init Complete & sklearn graph updated    
2019-11-17 Dataset Init  
2019-11-20 DAG package test and Func Wrapper Pandas Trace   
2019-11-21 Version 0.1.0 ready. Count used for evaluation   
2019-11-23 Version 0.1.1 ready. Add multiple metrices   
2019-11-26 Version 0.1.2 ready. Numerical & Categorical; Add changes output; Handle # classes dict comparison   
2019-11-27 Version 0.2.0 ready. Able to handle pipeline 3 & 4; Add supporting functions   
2019-12-01 Version 0.2.1 ready. DAG plots ready for testing; connect Pandas with Sklearn operations  
2019-12-02 Version 0.2.2 ready. Test for multiple pipelines with varying complexity   
2019-12-03 Version 0.2.3 ready. Bug fixed and graph generated.   



## Team Member

Biao Huang  

Chenqin Yang  

Rui Jiang  

Zhengyuan Ding  









   

