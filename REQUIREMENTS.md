# Hardware Environment Requirement: 
The experiment is performed on NVIDIA Tesla A100 with 128GB RAM on the Ubuntu system. 

# Software Environment Requirement: 
We fully implement an analyzer to get SFG for a Java method on top of Spoon, and the analyzer supports modern Java versions up to Java 16.
The explicit versioning information of python environment can be received in environment-cpu.yml and environment-gpu.yml.
## Requirements
* Conda
  * install conda: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
  * Create a new conda environment:
      * if you are running with GPU: 
        ```
        conda env create -f environment-gpu.yml
        conda activate semanticcodebert
        ```
        Dependencies include support for CUDA_11.4. If you are using a different CUDA version update the dependencies accordingly.
      * if you are running with CPU:   
        ```
        conda env create -f environment-cpu.yml
        conda activate semanticcodebert
        ```
* Dataset
  * Download `dataset.zip` from [https://drive.google.com/file/d/1ReVzBC-1WSciPgKH0Shz6pPEkueTeA0I/view?usp=sharing](https://drive.google.com/file/d/1ReVzBC-1WSciPgKH0Shz6pPEkueTeA0I/view?usp=sharing)
  * Put `dataset.zip` in the main directory and unzip

* SemanticCodeBERT
  * Download `SemanticCodeBERT.zip` from [https://drive.google.com/drive/folders/1xsQothDM9Wfg7piPG5CwCKhlivq5QXfG?usp=sharing](https://drive.google.com/drive/folders/1xsQothDM9Wfg7piPG5CwCKhlivq5QXfG?usp=sharing)
  * Put `SemanticCodeBERT.zip` in the main directory and unzip
  
* BERTOverflow
  * Download `BERTOverflow.zip` from [https://drive.google.com/drive/folders/1xsQothDM9Wfg7piPG5CwCKhlivq5QXfG?usp=sharing](https://drive.google.com/drive/folders/1xsQothDM9Wfg7piPG5CwCKhlivq5QXfG?usp=sharing)
  * Put `BERTOverflow.zip` in the main directory and unzip