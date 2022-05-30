class ModelExplainer():
    def __init__(self, model, dataset, args):
        self.model = model
        self.feature_to_loc = dataset.feature_to_loc
        self.args = args
        self.model_name = type(model).__name__

    def explain_model(self):
        if self.model_name == "LogisticRegression":
            found_predicted_feature = False
            found_protected_feature = False
            for i, (feature, loc) in enumerate(self.feature_to_loc.items()):
                if self.args.feature_to_predict == feature:
                    found_predicted_feature = True
                    continue
                elif self.args.protected_feature == feature:
                    found_protected_feature = True
                    continue
                add_protected_char = self.args.remove_protected_char and found_protected_feature
                print(feature, ": ", self.model.coef_[0][loc-found_predicted_feature - add_protected_char])
        # elif self.model_name == "RandomForestClassifier":
        else:
            raise NotImplementedError
