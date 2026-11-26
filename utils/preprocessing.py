from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def make_models():
    estimators = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), True),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED), False),
        "Random Forest": (RandomForestClassifier(random_state=SEED), False),
        "SVM": (SVC(probability=True, random_state=SEED), True),
        "KNN": (KNeighborsClassifier(), True),
    }
    return {
        name: Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler() if scale else "passthrough"),
            ("classifier", estimator),
        ])
        for name, (estimator, scale) in estimators.items()
    }

