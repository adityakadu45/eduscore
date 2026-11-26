from html import escape
import logging

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from train_model import train
from utils.data_loader import CLASSES, DATASET, FEATURES, ROOT, TARGET, audit, load_data
from utils.prediction import predict
from utils.visualizations import COLORS, categories, plot

TITLE = "Student Academic Performance Prediction Using Machine Learning"
ARTIFACT = ROOT / "models/best_model.joblib"
PAGES = ["Home Dashboard", "Dataset Explorer", "Exploratory Data Analysis",
         "Model Training and Evaluation", "Student Performance Prediction", "About Research Project"]

st.set_page_config(page_title="EduScore", page_icon="🎓", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_dataset(version):
    return load_data()


@st.cache_resource(show_spinner=False)
def read_bundle(version):
    return joblib.load(ARTIFACT)


def chart(figure):
    st.plotly_chart(figure, config={"displaylogo": False})


def home(frame, bundle):
    st.markdown(f"""<div class="edu-hero">
    <div class="edu-brand">EduScore</div><h1>{TITLE}</h1>
    <p>MCA research project · Aditya Dattatray Kadu · SYMCA15</p></div>""",
                unsafe_allow_html=True)
    for column, label, value in zip(
        st.columns(4), ["Student records", "Input features", "Algorithms", "Performance categories"],
        [len(frame), len(FEATURES), len(bundle["results"]["models"]), frame[TARGET].nunique()],
    ):
        column.metric(label, value)
    name = bundle["metadata"]["best_model"]
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Performance distribution")
        chart(categories(frame))
    with right:
        st.subheader("Selected model")
        st.metric(name, f"{bundle['results']['models'][name]['accuracy']:.2%} test accuracy")
        st.caption("Selection uses mean macro F1 across five training folds.")
        st.subheader("Research objectives")
        st.write("Compare five classifiers, evaluate performance across all three categories, "
                 "and demonstrate how historical academic indicators can support a prediction workflow.")
    st.subheader("System workflow")
    st.write("Validate data → stratify the split → cross-validate and select → evaluate on holdout → predict")


def explorer(frame):
    st.title(PAGES[1])
    st.caption(f"{len(frame)} records · {len(frame.columns)} columns")
    left, right = st.columns(2)
    query = left.text_input("Search all columns").strip()
    labels = right.multiselect("Performance categories", CLASSES, default=CLASSES)
    filtered = frame[frame[TARGET].isin(labels)]
    if query:
        matches = filtered.astype("string").apply(
            lambda col: col.str.contains(query, case=False, regex=False, na=False))
        filtered = filtered[matches.any(axis=1)]
    st.write(f"{len(filtered)} matching records")
    st.dataframe(filtered, hide_index=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode(),
                       "eduscore_filtered.csv", "text/csv")
    st.download_button("Download original CSV", DATASET.read_bytes(), DATASET.name, "text/csv")
    quality = audit(frame)
    st.subheader("Schema and data quality")
    st.dataframe(pd.DataFrame({"Data type": frame.dtypes.astype(str), "Missing": quality["missing"]}))
    st.write(f"Exact duplicate rows: {quality['duplicate_rows']}. "
             f"Repeated student IDs: {quality['duplicate_ids']}. "
             f"Disagreements with the paper’s illustrative cutoffs: {quality['threshold_disagreements']}.")
    st.subheader("Descriptive statistics")
    st.dataframe(frame.describe(include="all").T)
    chart(categories(frame))
    st.write("Inputs are " + ", ".join(FEATURES.values()) + ". The target is performance_category. "
             "Student ID identifies records. Final percentage is the observed outcome. Both are excluded from predictors.")


def eda(frame):
    st.title(PAGES[2])
    st.caption("Descriptive analysis of the supplied records. Associations do not establish causation.")
    distribution, outcomes, features = st.tabs(["Category distribution", "Academic outcomes", "Feature distributions"])
    with distribution:
        chart(categories(frame))
        st.subheader("Numeric correlations")
        numeric = frame[[*FEATURES, "final_percentage"]].rename(columns=FEATURES)
        chart(plot(px.imshow(numeric.corr(), text_auto=".2f", zmin=-1, zmax=1,
                             color_continuous_scale="RdBu_r", aspect="auto"), 460))
    with outcomes:
        context = dict(color=TARGET, category_orders={TARGET: CLASSES}, color_discrete_map=COLORS,
                       labels={**FEATURES, TARGET: "Performance category",
                               "final_percentage": "Final percentage"})
        for variable in ("attendance_percentage", "study_hours_per_week", "previous_gpa", "backlogs"):
            st.subheader(f"{FEATURES[variable]} and final percentage")
            chart(plot(px.scatter(frame, x=variable, y="final_percentage", opacity=.65, **context)))
        for variable in ("assignment_score", "internal_marks"):
            st.subheader(f"{FEATURES[variable]} by category")
            chart(plot(px.box(frame, x=TARGET, y=variable, points="outliers", **context)))
    with features:
        feature = st.selectbox("Input feature", list(FEATURES), format_func=FEATURES.get)
        chart(plot(px.histogram(frame, x=feature, color=TARGET, barmode="overlay", opacity=.68,
                                color_discrete_map=COLORS, labels=FEATURES)))


def evaluation(bundle):
    st.title(PAGES[3])
    if st.button("Retrain Models", type="primary"):
        try:
            with st.spinner("Validating the dataset and training five classifiers…"):
                train()
            st.session_state.training_complete = True
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(f"Training failed. {error}")
    if st.session_state.pop("training_complete", False):
        st.success("New model and evaluation files saved.")
    metadata, results = bundle["metadata"], bundle["results"]
    comparison, winner = bundle["comparison"], metadata["best_model"]
    st.write("Algorithms: " + ", ".join(results["models"]) + ".")
    for column, label, value in zip(st.columns(3), ["Training records", "Test records", "Majority baseline"],
                                    [metadata["train_size"], metadata["test_size"],
                                     f"{metadata['baseline_accuracy']:.2%}"]):
        column.metric(label, value)
    st.success(f"Selected for prediction · {winner}")
    st.write("The stratified 80:20 split uses seed 42. Each training fold fits its own median imputer. "
             "Logistic Regression, SVM and KNN also standardize features within the pipeline. "
             "Selection uses five-fold training CV macro F1, with CV accuracy as the first tie-breaker.")
    st.caption("Macro metrics give the three categories equal weight. Weighted metrics use class support.")
    st.dataframe(comparison.style.format({column: "{:.2%}" for column in comparison if column != "model"}),
                 hide_index=True)
    metrics = {"accuracy": "Accuracy", "macro_precision": "Macro precision",
               "macro_recall": "Macro recall", "macro_f1": "Macro F1"}
    for tab, (key, label) in zip(st.tabs(list(metrics.values())), metrics.items()):
        with tab:
            fig = px.bar(comparison, x="model", y=key, labels={"model": "Algorithm", key: label},
                         color_discrete_sequence=["#2856c5"])
            chart(plot(fig.update_yaxes(range=[0, 1], tickformat=".0%")))
    model = st.selectbox("Review an individual model", list(results["models"]))
    scores = pd.DataFrame(results["cross_validation"][model], index=pd.RangeIndex(1, 6, name="Fold"))
    st.subheader("Training cross-validation")
    st.dataframe(scores.style.format("{:.2%}"))
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        matrix = results["models"][model]["confusion_matrix"]
        chart(plot(px.imshow(matrix, x=CLASSES, y=CLASSES, text_auto=True,
                             labels={"x": "Predicted", "y": "Actual"}, color_continuous_scale="Blues")))
    with right:
        st.subheader("Classification report")
        report = pd.DataFrame({key: value for key, value in bundle["reports"][model].items()
                               if isinstance(value, dict)}).T
        st.dataframe(report.style.format(precision=3))
    st.subheader(f"Permutation importance · {winner}")
    importance = pd.DataFrame(results["importance"]).T.rename(index=FEATURES).sort_values("mean")
    chart(plot(px.bar(importance, x="mean", y=importance.index, error_x="std", orientation="h",
                      labels={"y": "Input", "mean": "Mean holdout macro F1 decrease"})))
    st.caption("Error bars show standard deviation across 20 shuffles. Importance describes model sensitivity.")


def prediction(frame, bundle):
    st.title(PAGES[4])
    st.caption("Enter a student profile within the observed dataset ranges.")
    ranges = bundle["metadata"]["feature_ranges"]
    medians = frame[list(FEATURES)].median()
    if st.button("Reset Inputs"):
        for key in FEATURES:
            st.session_state.pop(f"input_{key}", None)
        st.session_state.pop("prediction", None)
    with st.form("student_profile"):
        columns = st.columns(2)
        values = {}
        for index, (key, label) in enumerate(FEATURES.items()):
            integer = key == "backlogs"
            cast = int if integer else float
            low, high = map(cast, ranges[key])
            values[key] = columns[index % 2].number_input(
                label, min_value=low, max_value=high, value=cast(medians[key]),
                step=1 if integer else 0.1, key=f"input_{key}",
                help=f"Observed range: {low:g}–{high:g}.",
            )
        submitted = st.form_submit_button("Predict Performance", type="primary")
    if submitted:
        st.session_state.pop("prediction", None)
        try:
            result = predict(bundle["pipeline"], ranges, values)
            st.session_state.prediction = (bundle["metadata"]["trained_at"], values, result)
        except ValueError as error:
            st.error(str(error))
    saved = st.session_state.get("prediction")
    if saved and saved[0] == bundle["metadata"]["trained_at"]:
        _, submitted_values, (category, probabilities) = saved
        st.markdown(f'<div class="edu-result" style="--result:{COLORS[category]}">'
                    f'Predicted performance<strong>{escape(category)}</strong></div>', unsafe_allow_html=True)
        if probabilities:
            chart(plot(px.bar(x=CLASSES, y=[probabilities[label] for label in CLASSES],
                              color=CLASSES, color_discrete_map=COLORS,
                              labels={"x": "Category", "y": "Estimated probability"})
                       .update_layout(showlegend=False).update_yaxes(range=[0, 1], tickformat=".0%")))
            st.caption("The bars are model estimates. Individual outcomes can differ.")
        st.subheader("General academic guidance")
        suggestions = {
            "previous_gpa": "Review prerequisite topics with a tutor or instructor.",
            "attendance_percentage": "Identify barriers to attendance and plan how to catch up on missed classes.",
            "study_hours_per_week": "Set a weekly study schedule with short review sessions.",
            "assignment_score": "Review feedback and ask for clarification before the next assignment.",
            "internal_marks": "Practice assessment questions and review recurring mistakes with your instructor.",
        }
        tips = [tip for key, tip in suggestions.items() if submitted_values[key] < medians[key]]
        if submitted_values["backlogs"] > 0:
            tips.append("Agree on a backlog recovery plan with an academic adviser.")
        for tip in tips or ["Continue your current study routine and review feedback after each assessment."]:
            st.write(f"• {tip}")
        st.caption("Guidance uses descriptive reference points and has no intervention validation.")
    st.info("For academic research and decision support. Review predictions alongside other academic evidence.")


def about(bundle, quality):
    st.title(PAGES[5])
    st.subheader(TITLE)
    st.write("Aditya Dattatray Kadu · SYMCA15 · MCA Semester III")
    st.write("Guide · Mr. Atish Shriniwar")
    st.write("Indira College of Engineering and Management, Pune")
    sections = {
        "Problem and objectives": "Summative results arrive late. This study compares five classifiers "
        "using prior academic performance, engagement and assessment indicators, and evaluates class-specific errors.",
        "Methodology and implemented system": "The proposal, proposed system, report and IEEE paper "
        "describe a three-class supervised benchmark. EduScore implements the proposed Streamlit interface "
        "and adds training-only cross-validation. Algorithms: " + ", ".join(bundle["results"]["models"]) + ".",
        "Research findings": "The report lists preliminary Random Forest accuracy of 68.33%. The IEEE paper "
        "reports 70.00% Random Forest accuracy and 16.67% Low-class recall in its held-out benchmark. "
        "EduScore selects its prediction model using training CV. See Model Training and Evaluation for run results.",
        "Limitations": f"The dataset is small and imbalanced, with unverified provenance. {quality['threshold_disagreements']} "
        "supplied labels disagree with the paper’s illustrative thresholds. If the records are synthetic, "
        "the results demonstrate technical feasibility only. Real-world predictive validity requires external validation.",
        "Future scope": "Verify data provenance and label rules, collect representative cohorts, evaluate minority "
        "class recall and calibration, and test generalization across institutions.",
        "Conclusion": "EduScore demonstrates a reproducible comparison and prediction workflow. "
        "Class-specific results and data provenance limit practical interpretation.",
    }
    for heading, content in sections.items():
        st.subheader(heading)
        st.write(content)


st.sidebar.title("🎓 EduScore")
page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.caption("MCA research project · SYMCA15")
try:
    frame = read_dataset(DATASET.stat().st_mtime_ns)
    bundle = read_bundle(ARTIFACT.stat().st_mtime_ns)
except (OSError, ValueError):
    logging.getLogger(__name__).exception("EduScore could not load its deployment artifacts")
    st.error("EduScore could not load the dataset or model. Check that "
             "dataset/student_performance.csv and models/best_model.joblib are present and readable.")
    st.stop()

if page == PAGES[0]:
    home(frame, bundle)
elif page == PAGES[1]:
    explorer(frame)
elif page == PAGES[2]:
    eda(frame)
elif page == PAGES[3]:
    evaluation(bundle)
elif page == PAGES[4]:
    prediction(frame, bundle)
else:
    about(bundle, audit(frame))

st.markdown('<div class="edu-footer">EduScore · MCA Semester III · SYMCA15 · '
            'Indira College of Engineering and Management, Pune</div>', unsafe_allow_html=True)

