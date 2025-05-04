import numpy as np

def late_fusion(audio_probs, visual_probs, weights=(0.4, 0.6)):
    return audio_probs * weights[0] + visual_probs * weights[1]

def early_fusion(audio_features, visual_features):
    return np.concatenate([audio_features, visual_features], axis=1)

def confidence_weighted_fusion(audio_probs, visual_probs):
    audio_conf = np.max(audio_probs)
    visual_conf = np.max(visual_probs)
    total = audio_conf + visual_conf
    return (audio_probs * audio_conf + visual_probs * visual_conf) / total

def majority_vote(audio_pred, visual_pred):
    return audio_pred if audio_pred == visual_pred else visual_pred