class ModelExplainer():
    def __init__(self, model, dataset, args):
        self.model = model
        self.feature_to_loc = dataset.feature_to_loc
        self.args = args
        self.model_name = type(model).__name__

    def explain_model(self):
        if self.model_name == "LogisticRegression":
            found_predicted_feature = False
            for i, (feature, loc) in enumerate(self.feature_to_loc.items()):
                found_predicted_feature = found_predicted_feature or self.args.feature_to_predict == feature
                print(feature, ": ", self.model.coef_[0][loc-found_predicted_feature])
        # elif self.model_name == "RandomForestClassifier":
        else:
            raise NotImplementedError
