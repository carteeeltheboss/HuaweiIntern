import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_experiment("testingMLFlow")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimators = 200
depth = 10

print(f"Training model with {estimators} estimators and max depth {depth}...")
with mlflow.start_run():
    
    mlflow.log_param("n_estimators", estimators)
    mlflow.log_param("max_depth", depth)
    
    clf = RandomForestClassifier(n_estimators=estimators, max_depth=depth)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "my_random_forest_model")

print(f"Done! Model achieved {accuracy:.4f} accuracy.")