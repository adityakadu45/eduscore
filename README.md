# EduScore

**Student Academic Performance Prediction Using Machine Learning**

Aditya Dattatray Kadu · Exam Seat SYMCA15 · Master of Computer Applications, Semester III  
Guide · Mr. Atish Shriniwar  
Indira College of Engineering and Management, Pune

EduScore compares five supervised classifiers and demonstrates a six-input prediction workflow for Low, Medium and High academic performance. The research objectives are to compare algorithms under a common protocol, examine class-specific errors, and present academic indicators and model estimates in an interactive dashboard.

## Dataset and input ranges

The original 300-row, nine-column CSV is included byte-for-byte as `dataset/student_performance.csv`. It includes 58 Low, 191 Medium and 51 High records. There are zero missing values, exact duplicates and repeated student IDs.

| Input | Observed range accepted by the prediction form |
|---|---:|
| Previous GPA | 4–9.9 |
| Attendance percentage | 45.1–99.9 |
| Weekly study hours | 2.1–35 |
| Assignment score | 35.2–99.9 |
| Internal marks | 12.3–40 |
| Backlogs | 0–4, whole numbers |

These bounds describe the supplied records and the IEEE paper’s observed ranges. The target is `performance_category`. The pipeline excludes `student_id` and `final_percentage` to avoid using an identifier or the final outcome as a predictor. The paper’s illustrative percentage cutoffs are Low below 50, Medium from 50 to below 75, and High from 75. They conflict with 28 supplied Low labels. The experiment retains all supplied labels.

## Architecture and methodology

CSV → validation and duplicate audit → six predictors and supplied classes → stratified split → training-only cross-validation → held-out evaluation → saved pipelines → Streamlit.

- Python 3.12, Streamlit, Pandas, NumPy, scikit-learn, Plotly and Joblib power the workflow. Matplotlib and Seaborn are included in the environment for analysis. HTML and CSS style the dashboard.
- The five models are Logistic Regression, Decision Tree, Random Forest, SVM and KNN.
- The holdout split uses 80:20 stratification and seed 42, yielding 240 training and 60 test records. Exact duplicate rows would be removed before splitting. Conflicting student IDs stop training.
- Every pipeline fits median imputation inside its training partition. Logistic Regression, SVM and KNN also fit standardization. Models accept string class labels directly. SVM enables class probability estimates.
- Model selection uses mean macro F1 across five shuffled stratified training folds. Mean CV accuracy and ascending model name break ties. All models are evaluated on the same held-out partition after selection.
- Macro precision, recall and F1 give all classes equal weight. Weighted metrics use actual class support. Confusion matrices use Low, Medium, High order. Permutation importance uses 20 shuffles and reports the reduction in holdout macro F1 for the selected model.

## Experimental results

| Algorithm | Training CV macro F1 | Test accuracy | Test macro precision | Test macro recall | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 70.10% | 66.67% | 64.67% | 52.51% | 55.12% |
| SVM | 70.05% | 66.67% | 71.06% | 50.61% | 52.95% |
| KNN | 64.21% | 66.67% | 64.67% | 52.51% | 55.12% |
| Random Forest | 61.27% | 70.00% | 77.50% | 52.92% | 54.56% |
| Decision Tree | 59.17% | 60.00% | 52.76% | 50.12% | 47.96% |

**Logistic Regression** is selected for prediction. Random Forest led held-out accuracy. The training-majority baseline achieved 63.33% on the test set. Low-class recall was 33.33% for Logistic Regression and 16.67% for Random Forest.

## Application

- **Home Dashboard** shows dataset and model indicators, performance distribution and the workflow.
- **Dataset Explorer** offers search, category filters, downloads, descriptive statistics and data-quality checks.
- **Exploratory Data Analysis** charts numeric relationships, distributions and correlations.
- **Model Training and Evaluation** compares scores, shows CV folds and class reports, and supports retraining.
- **Student Performance Prediction** validates inputs, predicts a category and displays estimated probabilities and general study guidance.
- **About Research Project** summarizes the research, preliminary findings, limitations and future scope.

## Windows setup and launch

Use Python 3.12 and extract the ZIP before running its launchers. From the project directory, run `setup.bat` once, then double-click `run_project.bat`. The default address is `http://localhost:8501`.

Manual PowerShell commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py --validate-only
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
```

If Windows provides Python through `python`, use `python -m venv .venv` in the first command. If neither command finds it, use the absolute path to a Python 3.12 installation. When port 8501 is busy, append `--server.port 8502` to the Streamlit command and open the resulting URL. For missing dependencies, rerun setup from the extracted project directory. For missing or incompatible saved models, rerun training using the pinned requirements. Load Joblib files only from a trusted project copy.

## Deploy on Streamlit Community Cloud

The deployment uses Python 3.12, the root-level `app.py` entry point and the pinned `requirements.txt`. The app loads its data, assets and saved models relative to the project directory. The checked-in dataset, trained pipelines and evaluation results are available on first launch. The app needs no API keys or Streamlit secrets.

The dataset contains row-level academic records with student IDs. Because its provenance is unverified, confirm it is approved for public redistribution before publishing. Dataset Explorer allows visitors to view and download the supplied records.

1. Extract the submission ZIP into a fresh folder. Open a terminal in the folder that contains `app.py`. The extracted project includes the source, data, fitted models, results, tests and hidden configuration folders.
2. In [GitHub's new repository form](https://github.com/new), create an empty public repository named `eduscore`. Leave the README, license and ignore-file initialization options off. The extracted package already includes a README and ignore rules.
3. Run the commands below. Replace `YOUR-USERNAME` with the repository owner's GitHub username. Follow the Git Credential Manager sign-in prompt if it appears.

```powershell
git init -b main
git add .
git diff --cached --name-only
git commit -m "Prepare EduScore for Streamlit deployment"
git remote add origin https://github.com/YOUR-USERNAME/eduscore.git
git push -u origin main
```

4. On GitHub, confirm the repository root contains `app.py`, `requirements.txt`, `.streamlit/config.toml`, `dataset/`, `models/` and `results/`. The **Actions** tab shows the Linux and Python 3.12 test run. Review its logs if a check fails.
5. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and authorize repository access. Choose **Create app**, then **Yup, I have an app**.
6. Select repository `YOUR-USERNAME/eduscore`, branch `main` and main file path `app.py`. Choose an available subdomain, such as `eduscore-symca15`. In **Advanced settings**, select **Python 3.12**, leave Secrets empty, save and deploy.
7. Once the app opens, visit its actual `streamlit.app` URL in a signed-out or private browser window. Confirm that Home, Dataset Explorer and Model Evaluation load, then submit a profile on the Prediction page. Send faculty the verified `streamlit.app` URL shown in your browser.

If deployment fails, inspect the app logs and check the repository, branch, entry point and Python version. On Linux, file and folder names are case-sensitive. The [Community Cloud deploy guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [sharing guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app) show where to find those settings.

For a local Linux smoke test from the extracted folder:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m streamlit run app.py --server.headless true
```

## Project files

| Path | Contents |
|---|---|
| `app.py`, `assets/`, `.streamlit/` | Streamlit dashboard, styling and theme |
| `train_model.py`, `utils/` | Data validation, training, predictions and chart utilities |
| `dataset/` | Original CSV, copied without edits |
| `models/` | Fitted candidate pipelines, selected bundle and metadata |
| `results/` | Comparison CSV, classification reports, fold scores and importance |
| `tests/` | Dataset, training, prediction and Streamlit interaction tests |
| `setup.bat`, `run_project.bat` | Windows setup and launch |

The four supplied research documents remain in the workspace. The ZIP is a runnable Python project with Windows launch scripts.

## Limitations and reference differences

The project report gives preliminary Random Forest accuracy of 68.33%. The IEEE paper presents a separate fixed holdout benchmark with 70.00% Random Forest accuracy and 16.67% Low recall. Both references describe Streamlit as a proposed extension. This implementation adds Streamlit and training-only CV, so the selected prediction pipeline can differ from the held-out accuracy leader.

The dataset is small, imbalanced and of unverified provenance. If synthetic, its results demonstrate technical feasibility only. Real-world predictive validity would require verified labels and testing on independent representative cohorts. Correlation and feature importance describe associations or model behavior. Class probabilities are estimates. Academic suggestions are general guidance for research and decision support.

