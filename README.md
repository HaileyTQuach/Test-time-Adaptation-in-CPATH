<h3>[ Duration ]</h3>
- January 15, 2024 to April 24th, 2024 (WINTER 2024, Concordia University)


<br>
<br>

<h3>[ Team ]</h3>

| NAME |
| --- |
| Paria Mehbrod | 
| Phuong Thao Quach |

<br>

<h3>Overview of Test-Time Adaptation (TTA) for Digital Histopathology</h3>

<h4>Problem Statement</h4>
<p>
  In computational pathology, distribution shifts in imaging protocols and staining techniques pose significant challenges for analyzing digital histopathology images, impacting the accuracy of disease diagnosis and treatment planning. This project employs Test-Time Adaptation (TTA) techniques to enhance the generalization of deep learning models across different datasets, focusing on prostate cancer diagnosis in whole slide images (WSIs).
</p>

<h4>Proposed Methodologies</h4>
<p>
  The methodology involves two main phases:
</p>
<ol>
  <li>
    <strong>Model Training:</strong> Utilizing the ResNet50 architecture, models are trained on a primary dataset to classify prostate cancer tissues, incorporating modifications for domain-specific challenges.
  </li>
  <li>
    <strong>Test-Time Adaptation:</strong> Several TTA techniques are applied to adapt the pre-trained models to a secondary dataset from a different distribution, aiming to reduce prediction uncertainty and improve model reliability.
  </li>
</ol>

<h4>Challenges and Adaptation Strategies</h4>
<p>
  The primary challenge is the high variability in medical images, which requires specialized TTA strategies. Techniques like TENT, SAR, LAME, and DELTA are utilized to address these variations during model adaptation.
</p>

<h4>Results and Evaluation</h4>
<p>
  The performance of TTA techniques is continuously evaluated against baseline metrics to quantify improvements. This iterative process helps refine the techniques to ensure optimal adaptation to new data distributions.
</p>


<h3>[ Description on how to obtain the Dataset from an available download link ]</h3>
<p>
  Download links for the datasets required for this project are provided below. The first link leads to the dataset used in phase 1. The second leads to dataset used in phase 2. We also refer to the techniques used in the paper "Quality control stress test for deep learning-based diagnostic model in digital pathology" in the Artifact folder to apply 9 different kinds of artifacts on dataset 1 for experiments in phase 2. 
</p>
<ul>
  <li>
    <a href="https://zenodo.org/records/4789576/files/02_training_native.tar?download=1">Dataset 1</a>
  </li>
  <li>
    <a href="https://zenodo.org/records/4789576/files/01_case_western_native.tar?download=1">Dataset 2</a>
  </li>
</ul>
<br>

<h3>[ Requirements to run your Python code (libraries, etc) ]</h3>
  <p>
  To successfully run the Python code in this repository, several libraries and dependencies need to be installed. The code primarily relies on popular Python libraries such as NumPy, Matplotlib, Pandas, Seaborn, and Scikit-Learn for data manipulation, statistical analysis, and machine learning tasks.
</p>
<p>
  For deep learning models, the code uses PyTorch, along with its submodules such as <code>torchvision</code> and <code>torch.nn</code>. Ensure that you have the latest version of PyTorch installed, which can handle neural networks and various related functionalities.
</p>
<p>
  Additionally, the project uses the <code>Orion</code> library, an asynchronous hyperparameter optimization framework. This can be installed directly from its GitHub repository using the command <code>!pip install git+https://github.com/epistimio/orion.git@develop</code> and its related <code>profet</code> package with <code>!pip install orion[profet]</code>.
</p>
<p>Here is a comprehensive list of all the required libraries:</p>
<ul>
  <li>NumPy</li>
  <li>Pandas</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
  <li>Scikit-Learn</li>
  <li>PyTorch (along with <code>torch.nn</code>, <code>torch.optim</code>, <code>torch.utils.data</code>, etc.)</li>
  <li>Torchvision (including datasets, models, transforms)</li>
  <li>Orion (including the <code>profet</code> package)</li>
  <li>Argparse (for parsing command-line options)</li>
  <li>TSNE (from Scikit-Learn for dimensionality reduction techniques)</li>
  <li>KNeighborsClassifier, GridSearchCV (from Scikit-Learn for machine learning models)</li>
  <li>RandomForestClassifier (from Scikit-Learn for machine learning models)</li>
  <li>Classification metrics from Scikit-Learn (confusion_matrix, classification_report, etc.)</li>
</ul>
<p>
  For visualization and data analysis, Matplotlib and Seaborn are extensively used. Ensure all these libraries are installed in your environment to avoid any runtime errors.
</p>
<p>
  To install these libraries, you can use pip (Python's package installer). For most libraries, the installation can be as simple as running <code>pip install library-name</code>. For specific versions or sources, refer to the respective library documentation.
</p>

<br>
<h3>Instructions on How to Train/Validate Models and Experiment with TTA techniques</h3>

<p>For cluster environments like Compute Canada, utilize the provided shell scripts to train and validate the models. Ensure you clone the project repository and have all the required files before proceeding.</p>
    
<h4>Task 1: Training the Model</h4>
<p> To begin training the model using your cluster, follow these steps:</p>
<ul>
  <li>Navigate to the cloned project directory.</li>
  <li>Ensure that the <code>job_train.sh</code> script has execution permissions, setting them with <code>chmod +x job_train.sh</code> if needed.</li>
  <li>Submit the training job to the cluster using the command <code>sbatch job_train.sh</code>.</li>
  <li>Monitor the job's progress through your cluster's job management tools.</li>
</ul>

<h4>Task 2: Applying TTA techniques on trained model</h4>
<p>After training, you can validate the model using the following steps:</p>
<ul>
  <li>Make sure <code>job.sh</code> is executable, modifying permissions similarly if required.</li>
  <li>Launch the validation process by submitting <code>sbatch job.sh</code> to the cluster's scheduler.</li>
  <li>Check the output and error files generated by the scheduler for logs and results.</li>
</ul>

<p><strong>Additional Scripts:</strong></p>
<ul>
  <li><code>job_corrupt.sh</code>: Use this script if you need to simulate data corruption as part of your validation process.</li>
  <li><code>download.sh</code>: This script helps with downloading the necessary datasets for training and validation if they are not present in the local environment.</li>
</ul>

<p><strong>Note:</strong> Ensure that all the datasets and models are in the correct directories as expected by the scripts. Refer to the scripts' internal documentation for detailed information on their expected environments and parameters.</p>

<p>For detailed results analysis, use the <code>analyze_results.ipynb</code> notebook in a Jupyter environment to visualize and interpret your model's performance.</p>

