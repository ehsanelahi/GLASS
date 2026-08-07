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
                 delta=0.5, plotsPath=None,
                 numCounterfactuals=3, maxIterations=10, stepFactor=0.25,
                 maxPossibilities=10000, topK=20, refinementThreshold=0.8):


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

        self.numCounterfactuals = numCounterfactuals
        self.maxIterations = maxIterations
        self.stepFactor = stepFactor
        self.maxPossibilities = maxPossibilities
        self.topK = topK
        self.refinementThreshold = refinementThreshold
    

        knn = KNeighborsClassifier()
        knn.fit(X.values, np.zeros((X.shape[0],)))
        self.knn = knn
        self.X_values = X.values  

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
        Generate counterfactuals by hill-climbing from num_counterfactuals distinct
        KNN-neighbor-seeded starting points, adjusting controllable features to
        increase predicted probability across all requirements.

        """
        num_counterfactuals = self.numCounterfactuals
        num_features_to_change = len(self.controllableFeatureIndices)
        max_iterations = self.maxIterations
        step_factor = self.stepFactor  # controls adjustment magnitude
        step_factor_floor = step_factor / 16  # finest step tried before giving up

        counterfactuals = []

        # Precompute feature importance once (avoids redundant loops)
        feature_importance = np.zeros(num_features_to_change)
        for model in self.reqClassifiers:
            if hasattr(model, 'feature_importances_'):
                feature_importance += model.feature_importances_[self.controllableFeatureIndices]
            elif hasattr(model, 'coef_'):
                feature_importance += np.abs(model.coef_).flatten()[self.controllableFeatureIndices]
        feature_importance /= len(self.reqClassifiers)

       
        top_indices = np.argsort(feature_importance)[::-1]

        '''

        Distinct real starting points for each restart, so numCounterfactuals
        restarts actually explore different regions instead of retracing the
        same deterministic hill-climb 

        '''
       
        n_neighbors_to_fetch = min(num_counterfactuals, self.X_values.shape[0])
        neighbor_indices = np.ravel(self.knn.kneighbors([row], n_neighbors_to_fetch, return_distance=False))

        for r in range(num_counterfactuals):
            cf = np.copy(row)
            if r < len(neighbor_indices):
                neighbor_row = self.X_values[neighbor_indices[r]]
                cf[self.controllableFeatureIndices] = neighbor_row[self.controllableFeatureIndices]

            
            confidences = vecPredictProba(self.reqClassifiers, [cf])[0]
            base_prob = np.mean(confidences)
            current_step_factor = step_factor

            best_cf_seen = np.copy(cf)
            best_prob_seen = base_prob

            for _ in range(max_iterations):
                improved = False
                for i in top_indices:
                    f_idx = self.controllableFeatureIndices[i]
                    min_val, max_val = self.controllableFeatureDomains[i]
                    step_size = (max_val - min_val) * current_step_factor

                    # Try moving feature up and down
                    candidates = [
                        np.clip(cf[f_idx] + step_size, min_val, max_val),
                        np.clip(cf[f_idx] - step_size, min_val, max_val)
                    ]

                    best_prob = base_prob
                    best_val = cf[f_idx]
                    best_confidences = confidences

                    # Evaluate both directions
                    for val in candidates:
                        cf_candidate = np.copy(cf)
                        cf_candidate[f_idx] = val
                        cand_confidences = vecPredictProba(self.reqClassifiers, [cf_candidate])[0]
                        prob = np.mean(cand_confidences)

                        if prob > best_prob:
                            best_prob, best_val, best_confidences = prob, val, cand_confidences

                    # Apply best change
                    if best_prob > base_prob:
                        cf[f_idx] = best_val
                        base_prob = best_prob
                        confidences = best_confidences
                        improved = True

                if base_prob > best_prob_seen:
                    best_prob_seen = base_prob
                    best_cf_seen = np.copy(cf)

                # Stop once every requirement meets its own target confidence --
                # no need to keep iterating this restart once it's fully satisfied
                if (confidences >= self.targetConfidence).all():
                    break

                # No improvement at the current resolution: shrink the step and
                # retry, only giving up once we're stuck at the finest resolution
                if not improved:
                    if current_step_factor <= step_factor_floor:
                        break
                    current_step_factor /= 2

            counterfactuals.append(best_cf_seen[self.controllableFeatureIndices])

        return np.array(counterfactuals)
    


    def refine_adaptations_with_pdp(self, adaptations, row):
        candidates = []
        top_k = self.topK
        threshold = self.refinementThreshold  # Confidence threshold for early stopping
        max_possibilities = self.maxPossibilities
    
        # Precompute some frequently used values
        ext_features = row[self.externalFeatureIndices]
        n_reqs = len(self.reqClassifiers)
        n_features = len(self.controllableFeatureIndices)
    
        for adaptation in adaptations.copy():
      
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
    
            # Append external features
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

