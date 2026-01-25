import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsClassifier
import explainability_techniques.PDP as pdp
from util import cartesian_product
from util import vecPredictProba

'''
def vecPredictProba(models, X):
    if type(X) is list:
        X = np.array(X)
    if not isinstance(models, list):
        models = [models]
    probas = np.empty((X.shape[0], len(models)))
    for i, model in enumerate(models):
        probas[:, i] = model.predict_proba(X)[:, 1]
    return probas

'''

class CustomPlanner:
    def __init__(self, X, n_neighbors, n_startingSolutions,
                 reqClassifiers, targetConfidence,
                 controllableFeaturesNames,
                 controllableFeatureIndices,
                 controllableFeatureDomains,
                 optimizationDirections,
                 optimizationScoreFunction,
                 delta=0.5, plotsPath=None):

        preprocessingStartTime = time.time()

        self.n_neighbors = n_neighbors
        self.n_startingSolutions = n_startingSolutions
        self.reqClassifiers = reqClassifiers
        self.targetConfidence = targetConfidence
        self.controllableFeatureIndices = np.array(controllableFeatureIndices)
        self.externalFeatureIndices = np.delete(np.arange(X.shape[1]), controllableFeatureIndices)
        self.controllableFeatureDomains = controllableFeatureDomains
        self.optimizationDirections = optimizationDirections
        self.optimizationScoreFunction = optimizationScoreFunction
        self.delta = delta
    

        knn = KNeighborsClassifier()
        knn.fit(X.values, np.zeros((X.shape[0],)))
        self.knn = knn

        self.pdps = {}
        for i, feature in enumerate(controllableFeaturesNames):
            self.pdps[i] = []
            for j, reqClassifier in enumerate(reqClassifiers):
                path = None
                if plotsPath is not None:
                    path = os.path.join(plotsPath, f"req_{j}")
                    os.makedirs(path, exist_ok=True)
                self.pdps[i].append(pdp.partialDependencePlot(
                    reqClassifier, X, [feature], "both", os.path.join(path, f"{feature}.png")
                ))

        self.summaryPdps = []
        for i, feature in enumerate(controllableFeaturesNames):
            path = None
            if plotsPath is not None:
                path = os.path.join(plotsPath, "summary")
                os.makedirs(path, exist_ok=True)
            self.summaryPdps.append(pdp.multiplyPdps(
                self.pdps[i], os.path.join(path, f"{feature}.png")
            ))

        print("Total offline preprocessing duration:", time.time() - preprocessingStartTime, "seconds\n" + "=" * 100)



    def generate_counterfactuals(self, row):
        """
        Efficiently generate counterfactuals that flip prediction from 0 → 1
        by adjusting controllable features in the direction of increasing
        predicted probability for class 1.
        """
        target_label = 1
        num_counterfactuals = 3
        num_features_to_change = len(self.controllableFeatureIndices)
        max_iterations = 10
        step_factor = 0.25  # controls adjustment magnitude
    
        counterfactuals = []
    
        # Precompute feature importance once (avoids redundant loops)
        feature_importance = np.zeros(num_features_to_change)
        for model in self.reqClassifiers:
            if hasattr(model, 'feature_importances_'):
                feature_importance += model.feature_importances_[self.controllableFeatureIndices]
            elif hasattr(model, 'coef_'):
                feature_importance += np.abs(model.coef_).flatten()[self.controllableFeatureIndices]
        feature_importance /= len(self.reqClassifiers)
    
        top_indices = np.argsort(feature_importance)[-num_features_to_change:]
    
        for _ in range(num_counterfactuals):
            cf = np.copy(row)
            base_prob = np.mean(vecPredictProba(self.reqClassifiers, [cf]), axis=0)[1]  # prob of class 1
    
            for _ in range(max_iterations):
                improved = False
                for i in top_indices:
                    f_idx = self.controllableFeatureIndices[i]
                    min_val, max_val = self.controllableFeatureDomains[i]
                    step_size = (max_val - min_val) * step_factor
    
                    # Try moving feature up and down
                    candidates = [
                        np.clip(cf[f_idx] + step_size, min_val, max_val),
                        np.clip(cf[f_idx] - step_size, min_val, max_val)
                    ]
    
                    best_prob = base_prob
                    best_val = cf[f_idx]
    
                    # Evaluate both directions
                    for val in candidates:
                        cf_candidate = np.copy(cf)
                        cf_candidate[f_idx] = val
                        prob = np.mean(vecPredictProba(self.reqClassifiers, [cf_candidate]), axis=0)[1]
    
                        if prob > best_prob:
                            best_prob, best_val = prob, val
    
                    # Apply best change
                    if best_prob > base_prob:
                        cf[f_idx] = best_val
                        base_prob = best_prob
                        improved = True
    
                # Stop if model flips confidently to class 1
                if (base_prob >= self.targetConfidence).all():
                    counterfactuals.append(cf[self.controllableFeatureIndices])
                    break
    
                # No further improvement — stop early
                if not improved:
                    break
    
        # Return all found counterfactuals or original point if none
        return np.array(counterfactuals) if counterfactuals else np.array([row[self.controllableFeatureIndices]])
    





    def refine_adaptations_with_pdp(self, adaptations, row):
        candidates = []
        top_k = 20
        threshold = 0.8  # Confidence threshold for early stopping
        max_possibilities = 10000
    
        # Precompute some frequently used values
        ext_features = row[self.externalFeatureIndices]
        n_reqs = len(self.reqClassifiers)
        n_features = len(self.controllableFeatureIndices)
    
        for adaptation in adaptations.copy():
            # Find nearest neighbor once
            #neighbor_index = np.ravel(self.knn.kneighbors([adaptation], 1, return_distance=False))[0]
            # Create full feature vector by combining adaptation (controllable features) with original external features
            full_feature_vector = np.copy(row)
            full_feature_vector[self.controllableFeatureIndices] = adaptation
            neighbor_index = np.ravel(self.knn.kneighbors([full_feature_vector], 1, return_distance=False))[0]
    
            # Compute PD maximals efficiently
            maximals = [
                pdp.getMaximalsOfLine(self.summaryPdps[i], neighbor_index)
                for i in range(n_features)
            ]
    
            # Downsample large PD grids to keep combinations tractable
            n_possibilities = np.prod([len(m) for m in maximals])
            while n_possibilities > max_possibilities:
                i = np.argmax([len(m) for m in maximals])
                maximals[i] = maximals[i][::2]
                n_possibilities = np.prod([len(m) for m in maximals])
    
            # Create Cartesian product of all PD possibilities
            possibilities = cartesian_product(*maximals)
    
            # Append external (non-controllable) features
            repeated_ext = np.repeat([ext_features], possibilities.shape[0], axis=0)
            possibilities = np.append(possibilities, repeated_ext, axis=1)
    
            # Compute probabilities for all requirements at once
            cand_probs = vecPredictProba(self.reqClassifiers, possibilities)
    
            # Compute aggregated confidence (max across requirements)
            max_confidences = np.max(cand_probs, axis=1)
    
            # Early stop if all are highly confident
            if np.any(max_confidences >= threshold):
                # Add directly to candidates
                candidates.extend(zip(possibilities, max_confidences))
                continue
    
            # Otherwise, add all candidates below threshold as well
            candidates.extend(zip(possibilities, max_confidences))
    
        # Sort once — by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)
    
        # Select top_k most confident candidates
        best_candidates = [c[0] for c in candidates[:top_k]]
    
        return np.unique(best_candidates, axis=0)


    
    def validate_and_select_best(self, adaptations):
        #Remove duplicate adaptations
        adaptations = np.unique(adaptations, axis=0)
        bestAdaptations, bestAdaptationsConfidence, bestAdaptationsScores = [], [], []
    
        for i, adaptation in enumerate(adaptations):
            confidence = vecPredictProba(self.reqClassifiers, [adaptation])[0]
            bestAdaptations.append(adaptation)
            bestAdaptationsConfidence.append(confidence)
            bestAdaptationsScores.append(self.optimizationScoreFunction(adaptation))
    
        if bestAdaptations:
            bestIndex = np.argmax(bestAdaptationsScores)
            finalAdaptation = bestAdaptations[bestIndex]
            finalConfidence = bestAdaptationsConfidence[bestIndex]
            finalScore = bestAdaptationsScores[bestIndex]
            
            return finalAdaptation, finalConfidence, finalScore
        else:
            return None, None, None
        


    def findAdaptation(self, row):
        counterfactuals = self.generate_counterfactuals(row)
        refined = self.refine_adaptations_with_pdp(counterfactuals, row)
        return self.validate_and_select_best(refined)

