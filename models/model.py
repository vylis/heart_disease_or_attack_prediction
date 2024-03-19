import numpy as np
from joblib import dump
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
)
from sklearn.ensemble import (
    VotingClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch

from utils.fetch_data_source import fetch_data_source
from utils.cleaning import clean_data
from utils.preprocessing import process_data


def build_model():
    try:
        # load the dataset
        dataset = "data/heart_disease_dataset.csv"
        df = fetch_data_source(dataset)

        # clean the data
        df = clean_data(df)

        # create output and logs directories
        os.makedirs("./models/output", exist_ok=True)
        os.makedirs("./models/output/logs", exist_ok=True)

        # split features and target
        X = df.drop(columns=["HeartDiseaseorAttack"])
        y = df["HeartDiseaseorAttack"]

        # split df into train, test and val sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # process the data
        X_train = process_data(X_train, scaling_method="min_max")
        X_test = process_data(X_test, scaling_method="min_max")
        X_val = process_data(X_val, scaling_method="min_max")

        # models
        def build_nn(hp=None):
            nn = Sequential()
            nn.add(InputLayer(shape=(X_train.shape[1],)))
            units = (
                32
                if hp is None
                else hp.Int("units", min_value=32, max_value=512, step=32)
            )
            nn.add(Dense(units=units, activation="relu"))
            nn.add(Dense(units=units, activation="relu"))
            nn.add(Dense(1, activation="sigmoid"))
            nn.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            return nn

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, verbose=0, restore_best_weights=True
        )

        nn = build_nn()
        lr = LogisticRegression()
        dt = DecisionTreeClassifier()
        rf = RandomForestClassifier()
        gb = GradientBoostingClassifier()
        knn = KNeighborsClassifier()

        models = [
            nn,
            lr,
            dt,
            rf,
            gb,
            knn,
        ]

        model_names = [
            "Neural Network",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "K-Nearest Neighbors",
        ]

        # cross validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for i, model in enumerate(models):
            if model == nn:
                fold_no = 1
                nn_scores = []
                nn_preds = []
                y_tests = []

                for train, _ in kfold.split(X_train, y_train):
                    nn.fit(
                        x=X_train.iloc[train],
                        y=y_train.iloc[train],
                        epochs=30,
                        batch_size=32,
                        verbose=0,
                        callbacks=[early_stopping],
                        validation_data=(X_val, y_val),
                    )

                    # metrics
                    scores = nn.evaluate(X_val, y_val, verbose=0)

                    nn_scores.append(scores[1])
                    nn_pred = nn.predict(X_val)
                    nn_preds.extend(nn_pred)
                    y_tests.extend(y_val)

                    fold_no = fold_no + 1

                # measure performance with k-fold cross-validation
                print(f"Neural Network Cross-Validation Accuracy: {np.mean(nn_scores)}")
                print(f"Neural Network Standard Deviation: {np.std(nn_scores)}")

                # feature selection
                # could have used permutation importance importance but it's computationally expensive
                # because it re-fit the model for each feature and each fold
                # so we will use select k best and f_classif
                selector = SelectKBest(score_func=f_classif, k=5)
                selector.fit(X_train, y_train)
                importances = selector.scores_

                # get most important features
                indices = np.argsort(importances)[::-1]

                print("Feature Ranking:")
                for f in range(X_train.shape[1]):
                    print(
                        f"{f + 1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]})"
                    )

                # confusion matrix
                cm = confusion_matrix(y_tests, np.round(nn_preds))
                print(f"Neural Network Confusion Matrix: \n{cm}")

                # roc curve
                lb = LabelBinarizer()
                lb.fit(y_tests)
                y_test_lb = lb.transform(y_tests)
                y_pred_lb = lb.transform(np.round(nn_preds))
                fpr, tpr, _ = roc_curve(y_test_lb, y_pred_lb)
                roc_auc = auc(fpr, tpr)
                print(f"Neural Network ROC AUC: {roc_auc}\n")
            else:
                model.fit(X_train, y_train)

                # measure performance with k-fold cross-validation
                scores = cross_val_score(
                    model, X_val, y_val, cv=kfold, scoring="accuracy"
                )
                print(f"{model_names[i]} Cross-Validation Accuracy: {np.mean(scores)}")
                print(f"{model_names[i]} Standard Deviation: {np.std(scores)}")

                # feature selection
                if isinstance(
                    model,
                    (
                        DecisionTreeClassifier,
                        RandomForestClassifier,
                        GradientBoostingClassifier,
                    ),
                ):
                    # feature_importances for tree-based models
                    importances = model.feature_importances_
                elif isinstance(model, LogisticRegression):
                    # coef_ for linear models
                    importances = model.coef_.ravel()
                else:
                    # select k best for other models
                    selector = SelectKBest(score_func=f_classif, k=5)
                    selector.fit(X_train, y_train)
                    importances = selector.scores_

                # get most important features
                indices = np.argsort(importances)[::-1]

                print(f"{model_names[i]} Feature Ranking:")
                for f in range(X_train.shape[1]):
                    print(
                        f"{f + 1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]})"
                    )

                print("\n")

                # confusion matrix
                cm = confusion_matrix(y_test, model.predict(X_test))
                print(f"{model_names[i]} Confusion Matrix: \n{cm}")

                # roc curve
                lb = LabelBinarizer()
                lb.fit(y_test)
                y_test_lb = lb.transform(y_test)
                y_pred_lb = lb.transform(model.predict(X_test))
                fpr, tpr, _ = roc_curve(y_test_lb, y_pred_lb)
                roc_auc = auc(fpr, tpr)
                print(f"{model_names[i]} ROC AUC: {roc_auc}\n")

        # search best parameters
        best_models = {}
        best_params = {}

        # keras tuner random search
        tuner = RandomSearch(
            build_nn,
            objective="val_accuracy",
            max_trials=5,
            executions_per_trial=3,
            directory="keras",
            project_name="best_params",
        )

        tuner.search_space_summary()

        tuner.search(
            X_train,
            y_train,
            epochs=30,
            verbose=0,
            callbacks=[early_stopping],
            validation_data=(X_val, y_val),
        )

        nn_best_params = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Neural Network Best Parameters: {nn_best_params.get('units')}")
        nn_best_score = tuner.oracle.get_best_trials(1)[0]
        print(f"Neural Network Accuracy: {nn_best_score.score}")

        best_models[nn] = nn_best_score.score
        best_params[nn] = nn_best_params

        # grid search cv
        lr_params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        dt_params = {"max_depth": [None, 5, 10, 15, 20]}
        rf_params = {"n_estimators": [10, 50, 100, 200]}
        gb_params = {
            "n_estimators": [10, 50, 100, 200],
            "learning_rate": [0.01, 0.1, 1],
        }
        knn_params = {"n_neighbors": [3, 5, 7, 9, 11]}

        models_gs = [
            (lr, lr_params),
            (dt, dt_params),
            (rf, rf_params),
            (gb, gb_params),
            (knn, knn_params),
        ]

        for model, params in models_gs:
            gs = GridSearchCV(model, params, cv=kfold, scoring="accuracy", verbose=0)
            gs.fit(X_train, y_train)
            print(f"{model} Best Parameters: {gs.best_params_}")
            print(f"{model} Accuracy: {gs.best_score_}")

            # save best parameters and best model
            best_models[model] = gs.best_score_
            best_params[model] = gs.best_params_

        # combine models with voting classifier
        best_estimators = []

        models_voting = [lr, dt, rf, gb, knn]

        for model in models_voting:
            params = best_params[model]
            best_estimators.append((str(model), model.__class__(**params)))

        print(f"Best Estimators: {best_estimators}")
        voting_clf = VotingClassifier(
            estimators=best_estimators,
            voting="soft",
            verbose=False,
        )
        voting_clf.fit(X_train, y_train)
        print(f"Voting Classifier Accuracy: {voting_clf.score(X_test, y_test)}")

        best_models[voting_clf] = voting_clf.score(X_test, y_test)
        best_params[voting_clf] = best_estimators

        # define best model
        best_model_key = list(best_models.keys())[0]
        print(f"Best Model: {best_model_key}")

        # train best model on all the data and save it
        best_model_key = list(best_models.keys())[0]

        if best_model_key == nn:
            best_model = build_nn(hp=best_params[best_model_key])
            best_model.fit(
                X_train,
                y_train,
                epochs=30,
                batch_size=32,
                verbose=0,
                callbacks=[early_stopping],
                validation_data=(X_val, y_val),
            )
            best_model.save("./models/output/model.keras")
            print("Model saved")
        elif best_model_key == voting_clf:
            best_model = best_model_key.__class__(best_params[best_model_key])
            best_model.fit(X_train, y_train)
            dump(best_model, "./models/output/model.joblib")
            print("Model saved")
        else:
            best_model = best_model_key.__class__(**best_params[best_model_key])
            best_model.fit(X_train, y_train)
            dump(best_model, "./models/output/model.joblib")
            print("Model saved")

        y_pred_final = np.round(best_model.predict(X_test))

        # plot and save confusion matrix
        cm = confusion_matrix(y_test, y_pred_final)
        print(f"Confusion Matrix: \n{cm}")

        plt.figure(figsize=(10, 7))
        plt.title("Confusion Matrix")
        labels = ["TN", "FP", "FN", "TP"]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues")
        plt.savefig(
            f"./models/output/logs/{best_model_key.__class__.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_confusion_matrix.png"
        )
        print("Confusion matrix saved")

        # plot and save roc curve
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test_lb = lb.transform(y_test)
        y_pred_lb = lb.transform(y_pred_final)
        fpr, tpr, _ = roc_curve(y_test_lb, y_pred_lb)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc}")

        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color="darkorange", label="ROC curve" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(
            f"./models/output/logs/{best_model_key.__class__.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_roc_curve.png"
        )
        print("ROC curve saved")

        # create and save classification report
        evaluation_metrics = classification_report(y_test, y_pred_final)
        print(f"Classification Report: \n{evaluation_metrics}")

        with open(
            f"./models/output/logs/{best_model_key.__class__.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_classification_report.txt",
            "w",
        ) as f:
            f.write(evaluation_metrics)
        print("Classification report saved")

    except Exception as e:
        print(f"Error building model: {e}")
        raise e
