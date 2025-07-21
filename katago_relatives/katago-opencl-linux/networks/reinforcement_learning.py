import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import torch.nn.functional as F  # Ajout de cet import manquant

# Ajouter ces imports supplémentaires
import math
import csv
import json
from datetime import datetime

# Ajout pour le typage
from typing import Union, Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import des modules - ordre modifié pour éviter les erreurs d'importation
from const import BLACK, WHITE, EMPTY
from plateau import Plateau
from moteursDeJeu import simulation_MoteurDeJeu
import MCTS_GO as mcts

# Import des modèles de réseaux de neurones
from go_policy_9x9_v5 import PolicyNet9x9, policy_loss_with_smoothing
from go_value_9x9_v3 import ValueNet9x9

# Configuration
BOARD_SIZE = 9

def parse_device(device_str):
    """Parse device string and return torch.device and multi-GPU flag."""
    if device_str == 'cpu':
        return torch.device('cpu'), False, 0
    elif device_str == 'cuda':
        # Use all available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            return torch.device('cuda'), True, num_gpus
        elif num_gpus == 1:
            return torch.device('cuda:0'), False, 1
        else:
            return torch.device('cpu'), False, 0
    elif device_str.startswith('cuda:'):
        idx = int(device_str.split(':')[1])
        if torch.cuda.device_count() > idx:
            return torch.device(device_str), False, 1
        else:
            return torch.device('cpu'), False, 0
    else:
        # Default fallback
        return torch.device('cpu'), False, 0

# Valeurs par défaut, seront écrasées par l'argument --device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MULTI_GPU = torch.cuda.device_count() > 1
NUM_GPUS = torch.cuda.device_count()

MEMORY_SIZE = 1000000  # Taille du buffer d'expérience
BATCH_INFERENCE = True  # Active l'inférence par lots pour MCTS
BATCH_SIZE = 512      # Taille des batchs d'apprentissage
MCTS_SIMULATIONS = 800  # Nombre de simulations MCTS par coup
MCTS_EVAL_SIMULATIONS = 3200 # Simulations MCTS pour évaluation des modèles
TEMPERATURE = 1.0     # Température pour la sélection des coups (diminue au fil du jeu)
C_PUCT = 1.0          # Paramètre d'exploration pour MCTS
NUM_SELF_PLAY_GAMES = 50  # Nombre de parties d'auto-jeu par itération (augmenté de 25 à 50)
NUM_EPOCHS = 25       # Nombre d'époques d'entraînement par itération (augmenté de 15 à 25)
NUM_ITERATIONS = 10000  # Nombre d'itérations du processus complet
DIRICHLET_ALPHA = 0.15  # Paramètre pour le bruit Dirichlet (exploration à la racine)
DIRICHLET_WEIGHT = 0.30  # Poids du bruit Dirichlet

# Augmenter les configurations et constantes
CHECKPOINT_FREQ = 5  # Fréquence de sauvegarde des checkpoints
REFRESH_FREQ = 50    # Fréquence de rafraîchissement du modèle
METRICS_HEADERS = ['Iteration', 'Policy_Loss', 'Value_Loss', 'Win_Rate', 'Self_Play_Games', 'Buffer_Size']

def print_gpu_memory_usage():
    """Affiche l'utilisation actuelle de la mémoire GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"💾 GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

def modified_policy_loss(policy_logits, mcts_probs, prev_policy_logits=None, kl_weight=0.02):
    """Version étendue avec terme de régularisation"""
    # Perte principale (votre implémentation actuelle)
    ce_loss = -(mcts_probs * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    
    # Terme de régularisation KL (optionnel)
    if prev_policy_logits is not None:
        log_policy = F.log_softmax(policy_logits, dim=1)
        prev_policy = F.softmax(prev_policy_logits, dim=1)
        # Vérifier que les formes correspondent avant de calculer la KL-divergence
        assert log_policy.shape == prev_policy.shape, f"Forme incompatible: {log_policy.shape} vs {prev_policy.shape}"
        kl_div = F.kl_div(log_policy, prev_policy, reduction='batchmean')
        return ce_loss + kl_weight * kl_div
    
    return ce_loss

def refresh_model(model, optimizer, device, model_type="policy"):
    """
    Effectue un rafraîchissement du modèle pour améliorer les performances.
    
    Args:
        model: Le modèle à rafraîchir
        optimizer: L'optimiseur associé au modèle
        device: L'appareil sur lequel le modèle est chargé
        model_type: Type de modèle (policy ou value)
    
    Returns:
        tuple: (modèle rafraîchi, optimiseur modifié)
    """
    print(f"🔄 Performing {model_type} model refreshing...")
    
    try:
        # 1. Sauvegarder les poids du modèle dans un fichier temporaire
        tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'refresh_tmp_{model_type}.pth')
        torch.save(model.state_dict(), tmp_path)
        
        # 2. Réinitialiser partiellement l'optimiseur
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                param_state = optimizer.state.get(p)
                if param_state is not None and 'momentum_buffer' in param_state:
                    # Réduire le momentum de moitié
                    param_state['momentum_buffer'] *= 0.5
        
        # 3. Recharger le modèle
        state_dict = torch.load(tmp_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # 4. Nettoyer le fichier temporaire
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        # 5. Libérer la mémoire
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ {model_type.capitalize()} model refreshed successfully")
        return model, optimizer
        
    except Exception as e:
        print(f"❌ Erreur lors du refresh du modèle {model_type}: {e}")
        # En cas d'erreur, retourner le modèle et l'optimiseur inchangés
        return model, optimizer

def apply_board_augmentation(state, policy=None):
    """Applique une augmentation aléatoire à un état de plateau et sa politique associée"""
    # Déterminer la transformation aléatoire (8 possibilités - groupe diédral D4)
    transform_type = np.random.randint(0, 8)
    k_rot = transform_type % 4  # Nombre de rotations de 90°
    do_flip = transform_type >= 4  # Si True, on fait un flip horizontal
    
    # Appliquer les rotations à l'état
    if k_rot > 0:
        state = np.rot90(state, k=k_rot, axes=(0, 1))
    
    # Appliquer le flip horizontal à l'état
    if do_flip:
        state = np.flip(state, axis=1)
    
    # Si une politique est fournie, la transformer également
    if policy is not None:
        bs = int(np.sqrt(len(policy) - 1))  # Taille du board (moins l'action pass)
        
        # Extraire et transformer la politique sur le board
        policy_board = policy[:-1].reshape(bs, bs)
        pass_prob = policy[-1]
        
        # Appliquer les mêmes transformations à la politique
        if k_rot > 0:
            policy_board = np.rot90(policy_board, k=k_rot)
        if do_flip:
            policy_board = np.fliplr(policy_board)
        
        # Reconstruire la politique complète
        new_policy = np.zeros_like(policy)
        new_policy[:-1] = policy_board.flatten()
        new_policy[-1] = pass_prob
        
        return state, new_policy
    
    return state

class ReinforcementLearningBuffer:
    """Buffer circulaire amélioré pour stocker les expériences d'auto-jeu"""
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
        self.recent_buffer = deque(maxlen=capacity // 4)  # Tampon pour données récentes
        self.iteration_count = 0
        
    def add(self, state, mcts_policy, outcome):
        """Ajoute une expérience au buffer principal et au buffer récent"""
        self.buffer.append((state, mcts_policy, outcome))
        self.recent_buffer.append((state, mcts_policy, outcome))
        
    def add_game(self, game_data):
        """Ajoute toutes les positions d'une partie aux buffers"""
        for state, policy, outcome in game_data:
            self.add(state, policy, outcome)
            
    def sample(self, batch_size, recent_ratio=0.3, augment=True):
        """
        Échantillonne avec un biais vers les données récentes et applique les 8 transformations
        
        Args:
            batch_size: Taille de l'échantillon final (après augmentation)
            recent_ratio: Proportion de données récentes à inclure
            augment: Si True, applique toutes les 8 transformations à chaque exemple
        """
        if len(self.buffer) == 0:
            return [], [], []
            
        # Réduire la taille d'échantillon initial si on va augmenter
        actual_batch_size = batch_size // 8 if augment else batch_size
        
        recent_size = min(int(actual_batch_size * recent_ratio), len(self.recent_buffer))
        old_size = actual_batch_size - recent_size
        
        original_states, original_policies, original_outcomes = [], [], []

        # Échantillonner des données récentes
        if recent_size > 0:
            recent_indices = np.random.choice(len(self.recent_buffer), recent_size, replace=False)
            for i in recent_indices:
                state, policy, outcome = self.recent_buffer[i]
                original_states.append(state)
                original_policies.append(policy)
                original_outcomes.append(outcome)
        
        # Échantillonner des données plus anciennes
        if old_size > 0 and len(self.buffer) > 0:
            old_indices = np.random.choice(len(self.buffer), min(old_size, len(self.buffer)), replace=False)
            for i in old_indices:
                state, policy, outcome = self.buffer[i]
                original_states.append(state)
                original_policies.append(policy)
                original_outcomes.append(outcome)
        
        # Si pas d'augmentation, renvoyer les données originales
        if not augment or not original_states:
            states_array = np.array(original_states, dtype=np.float32)
            policies_array = np.array(original_policies, dtype=np.float32)
            outcomes_array = np.array(original_outcomes, dtype=np.float32)
            return states_array, policies_array, outcomes_array
        
        # Appliquer les 8 transformations à chaque exemple
        augmented_states = []
        augmented_policies = []
        augmented_outcomes = []
        
        for state, policy, outcome in zip(original_states, original_policies, original_outcomes):
            # Appliquer les 8 transformations (4 rotations x 2 flips)
            for transform_type in range(8):
                k_rot = transform_type % 4  # Rotation de 0, 90, 180 ou 270 degrés
                do_flip = transform_type >= 4  # Flip horizontal ou non
                
                # Copier l'état pour le transformer
                transformed_state = state.copy()
                
                # Appliquer les rotations à l'état
                if k_rot > 0:
                    transformed_state = np.rot90(transformed_state, k=k_rot, axes=(0, 1))
                
                # Appliquer le flip horizontal à l'état
                if do_flip:
                    transformed_state = np.flip(transformed_state, axis=1)
                
                # Transformer la politique
                transformed_policy = policy.copy()
                bs = int(np.sqrt(len(policy) - 1))  # Taille du board (moins l'action pass)
                
                # Extraire et transformer la politique sur le board
                policy_board = transformed_policy[:-1].reshape(bs, bs)
                pass_prob = transformed_policy[-1]
                
                # Appliquer les mêmes transformations à la politique
                if k_rot > 0:
                    policy_board = np.rot90(policy_board, k=k_rot)
                if do_flip:
                    policy_board = np.fliplr(policy_board)
                
                # Reconstruire la politique complète
                transformed_policy = np.zeros_like(policy)
                transformed_policy[:-1] = policy_board.flatten()
                transformed_policy[-1] = pass_prob
                
                # Ajouter aux listes augmentées
                augmented_states.append(transformed_state)
                augmented_policies.append(transformed_policy)
                augmented_outcomes.append(outcome)
        
        return np.array(augmented_states), np.array(augmented_policies), np.array(augmented_outcomes)
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path):
        """Sauvegarde le buffer sur disque avec métadonnées"""
        metadata = {
            'size': len(self.buffer),
            'recent_size': len(self.recent_buffer),
            'capacity': self.buffer.maxlen,
            'iteration': self.iteration_count,
            'timestamp': datetime.now().isoformat()
        }
        
        # Sauvegarder les métadonnées séparément
        meta_path = path.replace('.pkl', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Sauvegarder les données du buffer
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
            
        print(f"Buffer sauvegardé: {len(self.buffer)} expériences, métadonnées dans {meta_path}")
    
    def load(self, path):
        """Charge un buffer depuis le disque avec vérification des métadonnées"""
        if os.path.exists(path):
            try:
                # Vérifier et charger les métadonnées
                meta_path = path.replace('.pkl', '_meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    self.iteration_count = metadata.get('iteration', 0)
                    print(f"Métadonnées chargées: itération {self.iteration_count}")
                
                # Charger le buffer
                with open(path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    
                # Réinitialiser et remplir le buffer
                self.buffer = deque(loaded_data, maxlen=self.buffer.maxlen)
                
                # Remplir le buffer récent avec les dernières entrées
                # Correction linter : protéger contre None
                n_recent = min(len(self.buffer), self.recent_buffer.maxlen or len(self.buffer))
                self.recent_buffer = deque(list(self.buffer)[-n_recent:], maxlen=self.recent_buffer.maxlen)
                
                print(f"Buffer chargé: {len(self.buffer)} expériences, {len(self.recent_buffer)} récentes")
            except Exception as e:
                print(f"Erreur lors du chargement du buffer: {e}")
                return False
            return True
        return False
    
    def partial_reset(self, keep_ratio=0.3):
        """
        Conserve seulement une partie des données les plus récentes
        
        Args:
            keep_ratio: Proportion des exemples à conserver (entre 0 et 1)
        """
        if len(self.buffer) == 0:
            return
            
        # Calculer combien d'exemples garder
        keep_size = max(int(len(self.buffer) * keep_ratio), len(self.recent_buffer))
        
        # Ne garder que les exemples les plus récents
        self.buffer = deque(list(self.buffer)[-keep_size:], maxlen=self.buffer.maxlen)
        
        # Rafraîchir également le buffer récent
        self.recent_buffer = deque(list(self.buffer)[-min(len(self.buffer), self.recent_buffer.maxlen or len(self.buffer)):], 
                                maxlen=self.recent_buffer.maxlen)

def plot_training_metrics(metrics_file, output_dir, iteration_dir=None):
    """
    Génère un ensemble de graphiques détaillés pour visualiser l'évolution de l'entraînement.
    
    Args:
        metrics_file: Chemin vers le fichier de métriques principal
        output_dir: Répertoire de sortie pour les graphiques
        iteration_dir: Répertoire de l'itération actuelle (optionnel)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import glob
    
    # Charger les métriques globales
    df = pd.read_csv(metrics_file)
    current_iteration = df['Iteration'].max()
    
    # Créer une figure avec 4 sous-graphiques
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Métriques d'apprentissage par renforcement (Itérations 1-{current_iteration})", fontsize=16)
    
    # 1. Graphique d'évolution des pertes - CONVERTIR EN NUMPY ARRAYS
    axs[0, 0].plot(df['Iteration'].to_numpy(), df['Policy_Loss'].to_numpy(), 'b-', label='Policy Loss')
    axs[0, 0].plot(df['Iteration'].to_numpy(), df['Value_Loss'].to_numpy(), 'r-', label='Value Loss')
    axs[0, 0].set_title('Évolution des pertes')
    axs[0, 0].set_xlabel('Itération')
    axs[0, 0].set_ylabel('Perte')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. Graphique de taux de victoire - CONVERTIR EN NUMPY ARRAYS
    axs[0, 1].plot(df['Iteration'].to_numpy(), df['Win_Rate'].to_numpy(), 'g-o', label='Taux de victoire')
    axs[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50%')
    axs[0, 1].set_title('Taux de victoire contre la version précédente')
    axs[0, 1].set_xlabel('Itération')
    axs[0, 1].set_ylabel('Taux de victoire')
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].legend()
    
    # 3. Croissance du buffer d'expérience - CONVERTIR EN NUMPY ARRAYS
    axs[1, 0].plot(df['Iteration'].to_numpy(), df['Buffer_Size'].to_numpy(), 'm-o')
    axs[1, 0].set_title('Croissance du buffer d\'expérience')
    axs[1, 0].set_xlabel('Itération')
    axs[1, 0].set_ylabel('Nombre d\'exemples')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 4. Détails de l'itération actuelle - pertes par époque
    if iteration_dir and os.path.exists(iteration_dir):
        # Chercher le fichier CSV de l'itération
        iteration_csv = os.path.join(iteration_dir, f'iteration_{current_iteration}.csv')
        if os.path.exists(iteration_csv):
            try:
                # Lire le contenu du fichier CSV d'itération
                with open(iteration_csv, 'r') as f:
                    content = f.readlines()
                
                # Trouver où commence la section Value Training
                value_start = -1
                for i, line in enumerate(content):
                    if 'Value Training' in line:
                        value_start = i
                        break
                
                if value_start > 0:
                    # Extraire et parser les données de Policy
                    policy_data = []
                    for i in range(2, value_start):
                        if len(content[i].strip()) > 0 and ',' in content[i]:
                            epoch, loss = content[i].strip().split(',')
                            if epoch.isdigit() and loss:
                                policy_data.append((int(epoch), float(loss)))
                    
                    # Extraire et parser les données de Value
                    value_data = []
                    for i in range(value_start + 2, len(content)):
                        if len(content[i].strip()) > 0 and ',' in content[i]:
                            epoch, loss = content[i].strip().split(',')
                            if epoch.isdigit() and loss:
                                value_data.append((int(epoch), float(loss)))
                    
                    # Tracer les courbes de pertes
                    if policy_data:
                        policy_epochs, policy_losses = zip(*policy_data)
                        axs[1, 1].plot(np.array(policy_epochs), np.array(policy_losses), 'b-', label='Policy Loss')
                    
                    if value_data:
                        value_epochs, value_losses = zip(*value_data)
                        axs[1, 1].plot(value_epochs, value_losses, 'r-', label='Value Loss')
                    
                    axs[1, 1].set_title(f'Pertes par époque (Itération {current_iteration})')
                    axs[1, 1].set_xlabel('Époque')
                    axs[1, 1].set_ylabel('Perte')
                    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
                    axs[1, 1].legend()
                else:
                    axs[1, 1].text(0.5, 0.5, "Format de fichier non reconnu", 
                                  horizontalalignment='center', verticalalignment='center')
            except Exception as e:
                print(f"Erreur lors de l'analyse des données d'époque: {str(e)}")
                axs[1, 1].text(0.5, 0.5, f"Erreur d'analyse: {str(e)}", 
                              horizontalalignment='center', verticalalignment='center')
        else:
            axs[1, 1].text(0.5, 0.5, f"Fichier d'itération non trouvé:\n{iteration_csv}", 
                          horizontalalignment='center', verticalalignment='center')
    else:
        axs[1, 1].text(0.5, 0.5, "Détails de l'itération non disponibles", 
                      horizontalalignment='center', verticalalignment='center')
    
    # Ajouter des statistiques supplémentaires
    if df.shape[0] > 0:
        stats_text = (f"Dernier Policy Loss: {df['Policy_Loss'].iloc[-1]:.4f}\n"
                      f"Dernier Value Loss: {df['Value_Loss'].iloc[-1]:.4f}\n"
                      f"Dernier Win Rate: {df['Win_Rate'].iloc[-1]:.1%}\n"
                      f"Parties d'auto-jeu: {df['Self_Play_Games'].iloc[-1]}\n"
                      f"Taille actuelle du buffer: {df['Buffer_Size'].iloc[-1]}")
        fig.text(0.01, 0.01, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    
    # Sauvegarder la figure
    plt.savefig(os.path.join(output_dir, 'rl_metrics.png'), dpi=150)
    
    # Si on a un dossier d'itération, sauvegarder également une version spécifique à l'itération
    if iteration_dir:
        plt.savefig(os.path.join(iteration_dir, f'rl_metrics_iter_{current_iteration}.png'), dpi=150)
    
    plt.close()
    
    print(f"Graphique de métriques généré et sauvegardé dans {output_dir}/rl_metrics.png")

def load_model_with_prefix_handling(model, checkpoint_path, device=DEVICE):
    """Charge un modèle en gérant les préfixes DataParallel"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Vérifier s'il y a un préfixe module.
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Supprimer le préfixe module.
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Modèle chargé avec adaptation de préfixe DataParallel")
    else:
        model.load_state_dict(state_dict, strict=False)
        print("Modèle chargé en mode standard non-strict")
    return model

def load_model(model_class, source=None, wrap_data_parallel=False, device=DEVICE):
    """
    Fonction générale pour charger un modèle avec différentes sources de poids
    
    Args:
        model_class: Classe du modèle à instancier (PolicyNet9x9 ou ValueNet9x9)
        source: Source des poids. Peut être:
                - Un chemin de fichier (str)
                - Un dictionnaire d'état (dict)
                - Un modèle existant (nn.Module)
                - None pour un modèle vierge
        wrap_data_parallel: Si True, enveloppe le modèle dans nn.DataParallel
        device: Appareil sur lequel charger le modèle
        
    Returns:
        Le modèle chargé
    """
    # 1. Créer une nouvelle instance du modèle
    model = model_class().to(device)
    
    # 2. Charger les poids si une source est fournie
    if source is not None:
        if isinstance(source, str):
            # Source est un chemin de fichier
            model = load_model_with_prefix_handling(model, source, device)
        elif isinstance(source, dict):
            # Source est un dictionnaire d'état (state_dict)
            model.load_state_dict(source, strict=False)
        elif hasattr(source, 'state_dict'):
            # Source est un autre modèle
            state_dict = source.state_dict()
            # Gérer le cas où le modèle source est enveloppé dans DataParallel
            if list(state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
    
    # 3. Envelopper dans DataParallel si nécessaire
    if wrap_data_parallel and USE_MULTI_GPU:
        model = nn.DataParallel(model)
    
    return model

def extract_model_metrics(model, model_type='policy'):
    """
    Extrait des métriques sur l'état interne d'un modèle.
    
    Args:
        model: Le modèle à analyser
        model_type: Type de modèle ('policy' ou 'value')
        
    Returns:
        dict: Dictionnaire de métriques
    """
    metrics = {}
    
    # Statistiques sur les poids
    with torch.no_grad():
        weight_norm_vals = []
        weight_abs_mean_vals = []
        weight_std_vals = []
        
        # Parcourir tous les paramètres
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Calculer les statistiques pour ce tenseur de poids
                weight_norm_vals.append(torch.norm(param).item())
                weight_abs_mean_vals.append(param.abs().mean().item())
                weight_std_vals.append(param.std().item())
        
        # Calculer les statistiques globales
        if weight_norm_vals:
            metrics['Weight_Norm'] = np.mean(weight_norm_vals)
            metrics['Weight_AbsMean'] = np.mean(weight_abs_mean_vals)
            metrics['Weight_StdDev'] = np.mean(weight_std_vals)
            metrics['Weight_Max'] = np.max(weight_norm_vals)
            metrics['Weight_Min'] = np.min(weight_norm_vals)
    
    return metrics

def plot_model_metrics(metrics_file, output_dir, model_type):
    """Génère des graphiques pour les métriques d'entraînement"""
    try:
        df = pd.read_csv(metrics_file)
        
        # Configurer le graphique
        plt.figure(figsize=(10, 6))
        plt.title(f'Métriques {model_type.capitalize()}')
        plt.grid(True)
        
        # Tracer chaque colonne de métriques, sauf Epoch
        for col in df.columns:
            if col != 'Epoch':
                # Utiliser .to_numpy() pour convertir en tableau numpy avant de tracer
                plt.plot(df['Epoch'].to_numpy(), df[col].to_numpy(), label=col)
        
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{model_type}_metrics.png'))
        plt.close()
    except Exception as e:
        print(f"Erreur lors de la génération du graphique: {e}")

def plot_combined_losses(output_dir):
    """
    Trace les courbes de loss de toutes les itérations sur deux graphiques 
    (un pour policy, un pour value)
    
    Args:
        output_dir: Répertoire contenant les dossiers d'itérations
    """
    # Récupérer tous les dossiers d'itérations
    iteration_dirs = [d for d in os.listdir(output_dir) if d.startswith('iteration_') and os.path.isdir(os.path.join(output_dir, d))]
    iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))
    
    all_policy_losses = {}
    all_value_losses = {}
    
    # Parcourir chaque dossier d'itération
    for iteration_dir in iteration_dirs:
        iteration_number = int(iteration_dir.split('_')[1])
        iteration_csv = os.path.join(output_dir, iteration_dir, f'iteration_{iteration_number}.csv')
        
        if not os.path.exists(iteration_csv):
            continue
            
        # Lire les données des fichiers CSV
        policy_losses = []
        value_losses = []
        
        with open(iteration_csv, 'r') as f:
            reader = csv.reader(f)
            
            # État initial: avant les données
            state = "init"
            
            for row in reader:
                if not row:  # Ligne vide
                    continue
                    
                if row[0] == 'Policy Training':
                    state = "policy_header"
                elif row[0] == 'Value Training':
                    state = "value_header"
                elif state == "policy_header" and row[0] == 'Epoch':
                    state = "policy_data"
                elif state == "value_header" and row[0] == 'Epoch':
                    state = "value_data"
                elif state == "policy_data":
                    try:
                        policy_losses.append(float(row[1]))
                    except (ValueError, IndexError):
                        pass
                elif state == "value_data":
                    try:
                        value_losses.append(float(row[1]))
                    except (ValueError, IndexError):
                        pass
        
        # Stocker les pertes pour cette itération
        if policy_losses:
            all_policy_losses[iteration_number] = policy_losses
        if value_losses:
            all_value_losses[iteration_number] = value_losses
    
    # Créer les graphiques
    plt.figure(figsize=(20, 10))
    
    # Graphique des pertes de politique
    plt.subplot(1, 2, 1)
    max_epochs = max([len(losses) for losses in all_policy_losses.values()]) if all_policy_losses else 0
    color_map = plt.get_cmap('viridis')(np.linspace(0, 1, len(all_policy_losses)))
    
    for idx, (iteration, losses) in enumerate(sorted(all_policy_losses.items())):
        plt.plot(range(1, len(losses) + 1), losses, label=f'Itération {iteration}', 
                color=color_map[idx], alpha=0.8)
    
    plt.title('Pertes de politique par époque pour toutes les itérations', fontsize=14)
    plt.xlabel('Époque', fontsize=12)
    plt.ylabel('Perte de Policy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, max_epochs)
    
    # Ajouter une légende avec plusieurs colonnes si nombreuses itérations
    if len(all_policy_losses) > 10:
        plt.legend(ncol=3, fontsize=8)
    else:
        plt.legend(fontsize=10)
    
    # Graphique des pertes de valeur
    plt.subplot(1, 2, 2)
    max_epochs = max([len(losses) for losses in all_value_losses.values()]) if all_value_losses else 0
    color_map = plt.get_cmap('viridis')(np.linspace(0, 1, len(all_value_losses)))
    
    for idx, (iteration, losses) in enumerate(sorted(all_value_losses.items())):
        plt.plot(range(1, len(losses) + 1), losses, label=f'Itération {iteration}', 
                color=color_map[idx], alpha=0.8)
    
    plt.title('Pertes de valeur par époque pour toutes les itérations', fontsize=14)
    plt.xlabel('Époque', fontsize=12)
    plt.ylabel('Perte de Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, max_epochs)
    
    # Ajouter une légende avec plusieurs colonnes si nombreuses itérations
    if len(all_value_losses) > 10:
        plt.legend(ncol=3, fontsize=8)
    else:
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_losses.png'), dpi=300)
    plt.close()
    
    print(f"Graphique combiné des pertes sauvegardé dans {os.path.join(output_dir, 'combined_losses.png')}")

def run_rl_training(policy_model, value_model, buffer, output_dir, 
                    start_iteration=1, iterations=100, games_per_iteration=100,
                    epochs_per_iteration=20, metrics_history=None, mcts_simulations=MCTS_SIMULATIONS, eval_simulations=MCTS_EVAL_SIMULATIONS, eval_games=15):
    """
    Fonction principale d'entraînement par renforcement avec gestion avancée et reprise
    """

    policy_losses_per_epoch = []
    value_losses_per_epoch = []

    # Création des optimiseurs de manière séparée pour faciliter la reprise
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=0.00001)  # Réduit de 0.0001 à 0.00001
    value_optimizer = optim.Adam(value_model.parameters(), lr=0.00001)    # Réduit de 0.0001 à 0.00001
    
    policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        policy_optimizer, factor=0.7, patience=5, min_lr=1e-7  # Amélioré
    )
    value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        value_optimizer, factor=0.7, patience=5, min_lr=1e-7   # Amélioré
    )
    
    # Initialisation de l'historique des métriques
    if metrics_history is None:
        metrics_history = {
            'policy_loss': [],
            'value_loss': [],
            'win_rate': [],
            'buffer_size': []
        }
    
    # AJOUTEZ CES LIGNES ICI
    previous_policy_state = None
    previous_value_state = None
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Fichiers de sauvegarde des métriques
    metrics_csv = os.path.join(output_dir, 'metrics.csv')
    
    # Créer ou vérifier le fichier CSV
    if start_iteration == 1 or not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(METRICS_HEADERS)
    
    # Sauvegarder la config
    config = {
        'iterations': iterations,
        'games_per_iteration': games_per_iteration,
        'epochs_per_iteration': epochs_per_iteration,
        'device': str(DEVICE),
        'start_time': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Boucle principale d'apprentissage par renforcement
    for iteration in range(start_iteration, iterations + 1):
        print(f"\n{'='*30}")
        print(f"=== Itération {iteration}/{iterations} ===")
        print(f"{'='*30}")

        # Réinitialiser les listes de pertes au début de chaque itération
        policy_losses_per_epoch = []
        value_losses_per_epoch = []
        
        # 1. Libérer la mémoire avant de commencer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_gpu_memory_usage()
        
        # 2. Générer des parties par auto-jeu
        print("Génération de parties par auto-jeu...")
        
        # Créer le dossier d'itération avant de générer les parties
        iteration_dir = os.path.join(output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Chemin pour le fichier SGF de cette itération
        sgf_path = os.path.join(iteration_dir, f"selfplay_games.sgfs")
        
        training_data = self_play(policy_model, value_model, num_games=games_per_iteration, mcts_simulations=mcts_simulations, save_sgf_path=sgf_path)
        
        print(f"Parties d'auto-jeu sauvegardées dans: {sgf_path}")
            
        # 1.5 Rafraîchissement périodique du buffer (NOUVEAU CODE)
        # Raffinage adaptatif du buffer en fonction de sa taille
        if iteration % 10 == 0 and len(buffer) > 0:
            # Calcul d'un ratio de conservation adaptatif
            if len(buffer) > 200000:
                keep_ratio = max(0.3, 100000 / len(buffer))
                print(f"Rafraîchissement important du buffer (itération {iteration})...")
            elif len(buffer) > 50000:
                keep_ratio = 0.5
                print(f"Rafraîchissement modéré du buffer (itération {iteration})...")
            else:
                keep_ratio = 0.7
                print(f"Rafraîchissement léger du buffer (itération {iteration})...")
                
            old_size = len(buffer)
            buffer.partial_reset(keep_ratio=keep_ratio)
            print(f"Buffer réduit: {old_size} → {len(buffer)} exemples")
        
        # 3. Ajouter les données au buffer d'expérience
        for state, policy, outcome in training_data:
            buffer.add(state, policy, outcome)
        buffer.iteration_count = iteration
        
        # 4. Sauvegarder une copie du modèle actuel pour l'évaluation
        if previous_policy_state is None:
            # Première itération - utiliser les poids actuels
            previous_policy = load_model(PolicyNet9x9, policy_model)
            previous_value = load_model(ValueNet9x9, value_model)
        else:
            # Utiliser les poids sauvegardés de l'itération précédente
            previous_policy = load_model(PolicyNet9x9, previous_policy_state)
            previous_value = load_model(ValueNet9x9, previous_value_state)

        
        # 5. Entraîner les modèles avec refreshing périodique si nécessaire

        # Créer une copie du modèle précédent pour la régularisation KL
        if previous_policy_state is not None:
            prev_policy_model = load_model(PolicyNet9x9, previous_policy_state)
            prev_policy_model.eval()  # Mode évaluation pour éviter les calculs de gradient
        else:
            prev_policy_model = None

        print(f"Entraînement du réseau de politique ({epochs_per_iteration} époques)...")
        total_policy_loss = 0
        policy_model.train()
        
        for epoch in range(epochs_per_iteration):
            epoch_loss = 0
            num_batches = 0
            
            # Calculer le nombre de batchs de manière plus robuste
            num_batches_per_epoch = max(1, len(buffer) // BATCH_SIZE)
            
            # Échantillonner des mini-batchs du buffer
            for _ in range(num_batches_per_epoch):
                states, mcts_policies, _ = buffer.sample(BATCH_SIZE, augment=True)
                
                # Vérifier que nous avons des données
                if len(states) == 0:
                    continue
                
                # Préparer les entrées pour le réseau
                state_tensors = torch.from_numpy(states).float().to(DEVICE)
                policy_targets = torch.from_numpy(mcts_policies).float().to(DEVICE)
                
                # Forward pass
                # Vérifier la forme des données d'entrée
                if state_tensors.shape[1] != 2:
                    batch_size = state_tensors.shape[0]
                    
                    # Si le tensor est au mauvais format, le convertir
                    if len(state_tensors.shape) == 4 and state_tensors.shape[1] > 2:
                        # Créer deux canaux: un pour les pierres noires et un pour les blanches
                        black_stones = (state_tensors[:, 0, :, :] == BLACK).float().unsqueeze(1)  # Canal pour pierres noires
                        white_stones = (state_tensors[:, 0, :, :] == WHITE).float().unsqueeze(1)  # Canal pour pierres blanches
                        state_tensors = torch.cat([black_stones, white_stones], dim=1)  # [batch_size, 2, 9, 9]
                    elif len(state_tensors.shape) == 3:  # Format [batch_size, 9, 9]
                        black_stones = (state_tensors == BLACK).float().unsqueeze(1)
                        white_stones = (state_tensors == WHITE).float().unsqueeze(1)
                        state_tensors = torch.cat([black_stones, white_stones], dim=1)

                # Une fois correctement formaté, faire passer par le modèle
                policy_logits = policy_model(state_tensors)
                if prev_policy_model is not None:
                    with torch.no_grad():
                        prev_policy_logits = prev_policy_model(state_tensors)
                    loss = modified_policy_loss(policy_logits, policy_targets, prev_policy_logits, kl_weight=0.05)  # Augmenté de 0.02 à 0.05
                else:
                    loss = policy_loss_with_smoothing(policy_logits, policy_targets, label_smoothing=0.1)  # Augmenté de 0.05 à 0.1
                
                # Backward pass
                policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)  # Réduit de 2.0 à 1.0
                policy_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Refreshing périodique du modèle
            if (epoch + 1) % REFRESH_FREQ == 0:
                policy_model, policy_optimizer = refresh_model(policy_model, policy_optimizer, DEVICE, "policy")
            
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            total_policy_loss += avg_epoch_loss
            print(f"  Epoch {epoch+1}/{epochs_per_iteration}, Loss: {avg_epoch_loss:.4f}")
            policy_losses_per_epoch.append(avg_epoch_loss)
            
        avg_policy_loss = total_policy_loss / epochs_per_iteration
        policy_scheduler.step(avg_policy_loss)
        
        print(f"Entraînement du réseau de valeur ({epochs_per_iteration} époques)...")
        total_value_loss = 0
        value_model.train()
        
        for epoch in range(epochs_per_iteration):
            epoch_loss = 0
            num_batches = 0
            
            # Calculer le nombre de batchs de manière plus robuste
            num_batches_per_epoch = max(1, len(buffer) // BATCH_SIZE)
            
            # Échantillonner des mini-batchs du buffer
            for _ in range(num_batches_per_epoch):
                states, _, outcomes = buffer.sample(BATCH_SIZE, augment=True)

                # Vérifier que nous avons des données
                if len(states) == 0:
                    continue

                # Préparer les entrées pour le réseau
                state_tensors = torch.from_numpy(states).float().to(DEVICE)
                value_targets = torch.from_numpy(outcomes).float().to(DEVICE)
                
                # ✅ CORRECTION: S'assurer que les cibles sont dans [0,1]
                value_targets = torch.clamp(value_targets, 0.0, 1.0)
                
                # Vérifier la forme des données d'entrée
                if state_tensors.shape[1] != 2:
                    batch_size = state_tensors.shape[0]
                    
                    # Si le tensor est au mauvais format, le convertir
                    if len(state_tensors.shape) == 4 and state_tensors.shape[1] > 2:
                        # Créer deux canaux: un pour les pierres noires et un pour les blanches
                        black_stones = (state_tensors[:, 0, :, :] == BLACK).float().unsqueeze(1)  # Canal pour pierres noires
                        white_stones = (state_tensors[:, 0, :, :] == WHITE).float().unsqueeze(1)  # Canal pour pierres blanches
                        state_tensors = torch.cat([black_stones, white_stones], dim=1)  # [batch_size, 2, 9, 9]
                    elif len(state_tensors.shape) == 3:  # Format [batch_size, 9, 9]
                        black_stones = (state_tensors == BLACK).float().unsqueeze(1)
                        white_stones = (state_tensors == WHITE).float().unsqueeze(1)
                        state_tensors = torch.cat([black_stones, white_stones], dim=1)

                # Forward pass
                value_preds = value_model(state_tensors)
                value_targets = value_targets.squeeze(-1)  # Convertit [226, 1] en [226]
                criterion = nn.MSELoss()
                loss = criterion(value_preds, value_targets)
                
                # Backward pass
                value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)  # Réduit de 2.0 à 1.0
                value_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Refreshing périodique du modèle
            if (epoch + 1) % REFRESH_FREQ == 0:
                value_model, value_optimizer = refresh_model(value_model, value_optimizer, DEVICE, "value")
            
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            total_value_loss += avg_epoch_loss
            print(f"  Epoch {epoch+1}/{epochs_per_iteration}, Loss: {avg_epoch_loss:.4f}")
            value_losses_per_epoch.append(avg_epoch_loss)
        
        avg_value_loss = total_value_loss / epochs_per_iteration
        value_scheduler.step(avg_value_loss)
        
        # Afficher l'utilisation mémoire après l'entraînement
        if torch.cuda.is_available():
            print_gpu_memory_usage()
        
        # 6. Évaluer les progrès
        print("Évaluation des nouveaux modèles...")
        
        # Chemin pour le fichier SGF des parties d'évaluation
        eval_sgf_path = os.path.join(iteration_dir, f"evaluation_games.sgfs")
        
        win_rate = evaluate_models(policy_model, value_model, previous_policy, previous_value, num_games=eval_games, eval_simulations=eval_simulations, save_sgf_path=eval_sgf_path)
        
        print(f"Parties d'évaluation sauvegardées dans: {eval_sgf_path}")
        
        # 7. Mettre à jour et sauvegarder les métriques
        metrics_history['policy_loss'].append(avg_policy_loss)
        metrics_history['value_loss'].append(avg_value_loss)
        metrics_history['win_rate'].append(win_rate)
        metrics_history['buffer_size'].append(len(buffer))
        
       # 9. Sauvegarder les modèles et le buffer
        # Le dossier d'itération a déjà été créé plus tôt

        # Extraire les state dicts une seule fois (avec gestion multi-GPU)
        policy_state_dict = policy_model.module.state_dict() if USE_MULTI_GPU else policy_model.state_dict()
        value_state_dict = value_model.module.state_dict() if USE_MULTI_GPU else value_model.state_dict()

        # Sauvegarder les modèles individuels
        policy_path = os.path.join(iteration_dir, 'policy_model.pth')
        value_path = os.path.join(iteration_dir, 'value_model.pth')
        torch.save(policy_state_dict, policy_path)
        torch.save(value_state_dict, value_path)

        # NOUVEAU: Extraire les métriques des modèles
        policy_model_metrics = extract_model_metrics(policy_model, 'policy')
        value_model_metrics = extract_model_metrics(value_model, 'value')
        
        # Ajouter les métriques d'entraînement
        policy_model_metrics['Epoch'] = iteration
        policy_model_metrics['Loss'] = avg_policy_loss
        policy_model_metrics['Train_Loss'] = avg_policy_loss
        policy_model_metrics['Win_Rate'] = win_rate
        policy_model_metrics['Learning_Rate'] = policy_optimizer.param_groups[0]['lr']
        
        value_model_metrics['Epoch'] = iteration
        value_model_metrics['Loss'] = avg_value_loss
        value_model_metrics['Train_Loss'] = avg_value_loss
        value_model_metrics['Win_Rate'] = win_rate
        value_model_metrics['Learning_Rate'] = value_optimizer.param_groups[0]['lr']
        
        # Sauvegarder les métriques dans des fichiers CSV par modèle
        policy_metrics_csv = os.path.join(iteration_dir, 'policy_metrics.csv')
        value_metrics_csv = os.path.join(iteration_dir, 'value_metrics.csv')
        
        # Écrire les métriques dans les fichiers CSV
        with open(policy_metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(policy_model_metrics.keys())
            writer.writerow(policy_model_metrics.values())
        
        with open(value_metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(value_model_metrics.keys())
            writer.writerow(value_model_metrics.values())
        
        # Générer les graphiques des métriques
        plot_model_metrics(policy_metrics_csv, iteration_dir, 'policy')
        plot_model_metrics(value_metrics_csv, iteration_dir, 'value')
        
        # Ajouter la ligne aux métriques CSV globales
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                f"{avg_policy_loss:.6f}",
                f"{avg_value_loss:.6f}",
                f"{win_rate:.6f}",
                games_per_iteration,
                len(buffer)
            ])
        
        # Copier également les métriques globales dans l'itération
        iteration_metrics_csv = os.path.join(iteration_dir, 'metrics.csv')
        with open(iteration_metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(METRICS_HEADERS)
            writer.writerow([
                iteration,
                f"{avg_policy_loss:.6f}",
                f"{avg_value_loss:.6f}",
                f"{win_rate:.6f}",
                games_per_iteration,
                len(buffer)
            ])
        
        # Sauvegarde des pertes par époque
        iteration_csv = os.path.join(iteration_dir, f'iteration_{iteration}.csv')
        with open(iteration_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Policy Training'])
            writer.writerow(['Epoch', 'Loss'])
            for i, loss in enumerate(policy_losses_per_epoch, 1):
                writer.writerow([i, f"{loss:.4f}"])
                
            writer.writerow([])  # Ligne vide entre les sections
            writer.writerow(['Value Training'])
            writer.writerow(['Epoch', 'Loss'])
            for i, loss in enumerate(value_losses_per_epoch, 1):
                writer.writerow([i, f"{loss:.4f}"])
        
        # Créer les visualisations des métriques globales
        plot_training_metrics(metrics_csv, output_dir, iteration_dir)        # Ajouter ici la visualisation combinée de toutes les itérations
        plot_combined_losses(output_dir)
        # Créer des visualisations spécifiques à l'itération
        plot_training_metrics(iteration_metrics_csv, iteration_dir)
        
        # Sauvegarder le buffer périodiquement
        buffer_path = os.path.join(iteration_dir, 'experience_buffer.pkl')
        buffer.save(buffer_path)

        # Sauvegarder un checkpoint complet pour reprise
        checkpoint_path = os.path.join(output_dir, 'latest_checkpoint.pth')
        torch.save({
            'iteration': iteration,
            'policy_model': policy_state_dict,
            'value_model': value_state_dict,
            'buffer_path': buffer_path,
            'metrics': metrics_history,
            'config': config,
            'output_dir': output_dir
        }, checkpoint_path)
        print(f"Checkpoint complet sauvegardé: {checkpoint_path}")
        
        # AJOUTEZ CES LIGNES ICI (juste avant la fin de la boucle for)
        previous_policy_state = policy_model.module.state_dict() if USE_MULTI_GPU else policy_model.state_dict()
        previous_value_state = value_model.module.state_dict() if USE_MULTI_GPU else value_model.state_dict()
        
        
    print("\nApprentissage par renforcement terminé!")
    print(f"Modèles et métriques sauvegardés dans {output_dir}")

def resume_rl_training(checkpoint_path, iterations=None, games_per_iteration=None):
    """
    Reprend un entraînement par renforcement interrompu
    
    Args:
        checkpoint_path: Chemin vers le checkpoint à restaurer
        iterations: Nombre d'itérations à exécuter (si None, utilise la valeur précédente)
        games_per_iteration: Nombre de parties par itération (si None, utilise la valeur précédente)
    """
    print(f"Reprise de l'entraînement depuis: {checkpoint_path}")
    
    try:
        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Vérifier les composants requis
        required_keys = ['policy_model', 'value_model', 'buffer_path', 'iteration', 
                         'metrics', 'config', 'output_dir']
        missing = [key for key in required_keys if key not in checkpoint]
        if missing:
            raise ValueError(f"Composants manquants dans le checkpoint: {missing}")
        
        # Extraire configuration et état
        config = checkpoint['config']
        current_iteration = checkpoint['iteration']
        output_dir = checkpoint['output_dir']
        
        # Mise à jour des paramètres spécifiés
        if iterations is not None:
            config['iterations'] = iterations
        if games_per_iteration is not None:
            config['games_per_iteration'] = games_per_iteration
        
        # Recréer les modèles
        policy_model = load_model(PolicyNet9x9, checkpoint['policy_model'], wrap_data_parallel=USE_MULTI_GPU)
        value_model = load_model(ValueNet9x9, checkpoint['value_model'], wrap_data_parallel=USE_MULTI_GPU)
        print(f"Modèles repris" + (f" distribués sur {NUM_GPUS} GPUs" if USE_MULTI_GPU else ""))

        # Recréer le buffer et le charger
        buffer = ReinforcementLearningBuffer()
        buffer_loaded = buffer.load(checkpoint['buffer_path'])
        if not buffer_loaded:
            print("Attention: Buffer non chargé. Utilisation d'un buffer vide.")
        
        # Récupérer l'historique des métriques
        metrics_history = checkpoint['metrics']
        
        print(f"Entraînement repris à l'itération {current_iteration}/{config['iterations']}")
        print(f"Output directory: {output_dir}")
        
        # Lancer l'entraînement avec les paramètres restaurés
        run_rl_training(
            policy_model=policy_model,
            value_model=value_model,
            buffer=buffer,
            output_dir=output_dir,
            start_iteration=current_iteration + 1,
            iterations=config['iterations'],
            games_per_iteration=config['games_per_iteration'],
            epochs_per_iteration=config['epochs_per_iteration'],
            metrics_history=metrics_history
        )
        
    except Exception as e:
        print(f"Erreur lors de la reprise de l'entraînement: {e}")
        import traceback
        traceback.print_exc()

def generate_game(policy_model, value_model, temperature_func: Union[float, Callable[[int], float]] = TEMPERATURE, iterations=MCTS_SIMULATIONS):
    """Génère une partie complète par auto-jeu avec les modèles actuels"""
    # Nouveau: Varier la graine aléatoire pour chaque partie
    seed = int(time.time() * 1000000) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)  # Utiliser la même graine pour numpy
    torch.manual_seed(seed)  # Ajouter la graine pour torch
    
    # Définir une limite maximale de coups pour éviter les parties infinies
    max_moves = BOARD_SIZE * BOARD_SIZE * 2  # Environ 160 coups max sur 9x9
    
    # Ajouter une chance de varier le bruit Dirichlet pour maintenir la diversité
    dirichlet_alpha = DIRICHLET_ALPHA * (0.5 + 1.0 * random.random())  # Plus de variation
    dirichlet_weight = DIRICHLET_WEIGHT * (0.5 + 1.0 * random.random())  # Plus de variation
    
    # Initialiser le plateau et le moteur de jeu
    plateau = Plateau(BOARD_SIZE)
    game_history = []
    moves = []  # Pour le fichier SGF
    moteur = simulation_MoteurDeJeu(plateau)
    move_num = 0  # Compteur de coups pour la température
    
    # Préparer MCTS pour utiliser les réseaux de neurones
    mcts.enable_policy_value_networks(iterations=iterations, exploration=C_PUCT, 
                               policy_model_ext=policy_model, value_model_ext=value_model)
    
    # Utiliser torch.no_grad pour améliorer les performances d'inférence
    with torch.no_grad():
        while not moteur.end_game() and move_num < max_moves:
            # Obtenir la température actuelle en fonction du numéro de coup
            current_temp = temperature_func(move_num) if callable(temperature_func) else temperature_func
            
            current_player = moteur.get_turn()
            board = moteur.get_plateau().get_plateau()
            
            # Obtenir tous les coups légaux
            legal_moves = moteur.liste_coups_jouable()
            
            # Traiter les coordonnées en index linéaires pour la politique
            move_indices = {}
            for move in legal_moves:
                if move is None:
                    idx = BOARD_SIZE * BOARD_SIZE  # Indice de passe
                else:
                    i, j = move
                    idx = i * BOARD_SIZE + j
                move_indices[move] = idx
            
            # Ajouter du bruit de Dirichlet à la racine uniquement si move_num < 30
            use_noise = move_num < 30
            
            # Parfois jouer un coup aléatoire (exploration)
            if random.random() < 0.1 and move_num < 15:  # 5% de chance au début de la partie
                if legal_moves:
                    selected_move = random.choice(legal_moves)
                    # print(f"🎲 Coup aléatoire {selected_move} (coup {move_num})")
                    # Toujours exécuter MCTS pour générer la politique d'apprentissage
                    _, arbre = mcts.MCTS(moteur, current_player, use_tqdm=False, 
                                       use_dirichlet=use_noise,
                                       dirichlet_alpha=dirichlet_alpha,
                                       dirichlet_weight=dirichlet_weight,
                                       autorise_pass=False)
                else:
                    selected_move, arbre = mcts.MCTS(moteur, current_player, use_tqdm=False,
                                                 use_dirichlet=use_noise,
                                                 dirichlet_alpha=dirichlet_alpha,
                                                 dirichlet_weight=dirichlet_weight,
                                                 autorise_pass=False)
            else:
                selected_move, arbre = mcts.MCTS(moteur, current_player, use_tqdm=False,
                                              use_dirichlet=use_noise,
                                              dirichlet_alpha=dirichlet_alpha,
                                              dirichlet_weight=dirichlet_weight,
                                              autorise_pass=False)
            
            # Enregistrer le coup pour le fichier SGF
            moves.append((current_player, selected_move))
            
            root = arbre.get_racine()
            children = arbre.get_enfants(root)
            
            # Construire la politique MCTS
            total_visits = sum(child.get_nombre_simulation() for child in children)
            mcts_policy = np.zeros(BOARD_SIZE * BOARD_SIZE + 1)  # +1 pour le coup "pass"
            
            for child in children:
                move = child.get_coup()
                visits = child.get_nombre_simulation()
                if move in move_indices:
                    mcts_policy[move_indices[move]] = visits / total_visits
            
            # Appliquer la température
            if current_temp != 1.0:
                mcts_policy = mcts_policy ** (1 / current_temp)
                mcts_policy /= np.sum(mcts_policy)
            
            # Sauvegarder l'état actuel et la politique MCTS
            game_history.append({
                'board': board.copy(),
                'player': current_player,
                'policy': mcts_policy.copy(),
                'move': selected_move
            })
            
            # Jouer le coup et incrémenter le compteur
            moteur.jouer_coup(selected_move)
            move_num += 1
            
            # Libérer la mémoire périodiquement
            if move_num % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Déterminer le résultat final
    moteur.get_plateau().actualise_points_territoires()
    points_noir, points_blanc = moteur.get_plateau().get_score_total_couple()
    result = 1.0 if points_noir > points_blanc else 0.0 if points_blanc > points_noir else 0.5
    
    # Déterminer le gagnant pour le fichier SGF
    if result == 1.0:
        winner = BLACK
    elif result == 0.0:
        winner = WHITE
    else:
        winner = None  # Match nul
    
    # Transformer l'historique en données d'entraînement
    training_data = []
    
    for entry in game_history:
        player = entry['player']
        # Ajuster la récompense selon le joueur
        outcome = result if player == BLACK else (1.0 - result)
        
        # Ajouter les données d'entraînement
        training_data.append((
            entry['board'].copy(),
            entry['policy'].copy(),
            outcome
        ))
    
    # Libérer la mémoire à la fin
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Retourner les données d'entraînement ET les informations pour le fichier SGF
    return training_data, moves, winner, points_noir, points_blanc

def save_game_to_sgf(moves, winner, points_noir, points_blanc, path, append=False, black_name="RL_Agent_Black", white_name="RL_Agent_White"):
    """
    Sauvegarde une partie au format SGF avec le score précis
    
    Args:
        moves: Liste des coups joués
        winner: Le gagnant de la partie (BLACK, WHITE, ou None pour match nul)
        points_noir: Score final de Noir
        points_blanc: Score final de Blanc
        path: Chemin où sauvegarder le fichier SGF
        append: Si True, ajoute la partie à un fichier existant
        black_name: Nom du joueur Noir (par défaut "RL_Agent_Black")
        white_name: Nom du joueur Blanc (par défaut "RL_Agent_White")
    """
    # Préparer la chaîne de résultat avec le score
    score_diff = abs(points_noir - points_blanc)
    if winner == BLACK:
        result = f"B+{score_diff:.1f}"
    elif winner == WHITE:
        result = f"W+{score_diff:.1f}"
    else:
        result = "0" # Match nul
    
    # Pour un format standard de collection SGF
    if append:
        try:
            # Lire le fichier existant
            with open(path, 'r') as f:
                content = f.read()
            
            # Vérifier si c'est déjà une collection
            if content.strip().startswith('(;') and content.strip().endswith(')'):
                # Retirer la dernière parenthèse fermante
                content = content.rstrip('\n)')
                
                # Écrire le fichier avec la nouvelle partie
                with open(path, 'w') as f:
                    f.write(content)
                    f.write("\n(;FF[4]GM[1]SZ[{}]PB[{}]PW[{}]".format(
                        BOARD_SIZE, black_name, white_name))
                    
                    # Ajouter le résultat avec le score
                    f.write(f"RE[{result}]")
                        
                    # Ajouter les coups
                    for player, move in moves:
                        if player == BLACK:
                            prefix = ";B"
                        else:
                            prefix = ";W"
                            
                        if move is None:
                            f.write("{}[]".format(prefix))  # Passe
                        else:
                            x, y = move
                            # Convertir en coordonnées SGF
                            sgf_x = chr(97 + x)
                            sgf_y = chr(97 + y)
                            f.write("{}[{}{}]".format(prefix, sgf_x, sgf_y))
                    
                    f.write(")")  # Fermeture de la partie
                    f.write("\n)")  # Fermeture de la collection
            else:
                # Premier ajout à un fichier non-collection ou format incorrect
                with open(path, 'w') as f:
                    f.write("(")  # Début de collection
                    f.write("\n(;FF[4]GM[1]SZ[{}]PB[{}]PW[{}]".format(
                        BOARD_SIZE, black_name, white_name))
                    
                    # Ajouter le résultat avec le score
                    f.write(f"RE[{result}]")
                    
                    # Ajouter les coups
                    for player, move in moves:
                        if player == BLACK:
                            prefix = ";B"
                        else:
                            prefix = ";W"
                            
                        if move is None:
                            f.write("{}[]".format(prefix))  # Passe
                        else:
                            x, y = move
                            # Convertir en coordonnées SGF
                            sgf_x = chr(97 + x)
                            sgf_y = chr(97 + y)
                            f.write("{}[{}{}]".format(prefix, sgf_x, sgf_y))
                    
                    f.write(")")  # Fermeture de la partie
                    f.write("\n)")  # Fermeture de la collection
        except FileNotFoundError:
            # Le fichier n'existe pas encore, créer comme nouvelle collection
            with open(path, 'w') as f:
                f.write("(")  # Début de collection
                f.write("\n(;FF[4]GM[1]SZ[{}]PB[{}]PW[{}]".format(
                    BOARD_SIZE, black_name, white_name))
                
                # Ajouter le résultat avec le score
                f.write(f"RE[{result}]")
                
                # Ajouter les coups
                for player, move in moves:
                    if player == BLACK:
                        prefix = ";B"
                    else:
                        prefix = ";W"
                        
                    if move is None:
                        f.write("{}[]".format(prefix))  # Passe
                    else:
                        x, y = move
                        # Convertir en coordonnées SGF
                        sgf_x = chr(97 + x)
                        sgf_y = chr(97 + y)
                        f.write("{}[{}{}]".format(prefix, sgf_x, sgf_y))
                
                f.write(")")  # Fermeture de la partie
                f.write("\n)")  # Fermeture de la collection
    else:
        # Nouvelle partie simple (non collection)
        with open(path, 'w') as f:
            f.write("(;FF[4]GM[1]SZ[{}]PB[{}]PW[{}]".format(
                BOARD_SIZE, black_name, white_name))
            
            # Ajouter le résultat avec le score
            f.write(f"RE[{result}]")
            
            # Ajouter les coups
            for player, move in moves:
                if player == BLACK:
                    prefix = ";B"
                else:
                    prefix = ";W"
                    
                if move is None:
                    f.write("{}[]".format(prefix))  # Passe
                else:
                    x, y = move
                    # Convertir en coordonnées SGF
                    sgf_x = chr(97 + x)
                    sgf_y = chr(97 + y)
                    f.write("{}[{}{}]".format(prefix, sgf_x, sgf_y))
            
            f.write(")")  # Fermeture de la partie

def self_play(policy_model, value_model, num_games=NUM_SELF_PLAY_GAMES, mcts_simulations=MCTS_SIMULATIONS, save_sgf_path=None):
    """Génère plusieurs parties par auto-jeu et retourne les données d'entraînement"""
    all_training_data = []
    # Mettre les modèles en mode évaluation
    policy_model.eval()
    value_model.eval()
    
    successful_games = 0
    
    for i in tqdm(range(num_games), desc="Auto-jeu"):
        try:
            # Temperature décroissante: commence à 1.0, diminue à 0.1 après 50 coups
            temperature = lambda move_num: 1.0 if move_num < 80 else 0.8 if move_num < 120 else 0.6  # Plus de température
            game_data, moves, winner, points_noir, points_blanc = generate_game(policy_model, value_model, temperature, iterations=mcts_simulations)
            all_training_data.extend(game_data)
            successful_games += 1
            
            # Sauvegarder la partie au format SGF si un chemin est fourni
            if save_sgf_path:
                save_game_to_sgf(moves, winner, points_noir, points_blanc, save_sgf_path, append=(successful_games > 1))
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la partie {i+1}: {e}")
            continue
        
        # Libérer la mémoire régulièrement pour éviter les fuites
        if i > 0 and i % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"✅ Parties réussies: {successful_games}/{num_games}")
    print(f"📊 Données générées: {len(all_training_data)} positions")
    return all_training_data

def evaluate_models(current_policy, current_value, previous_policy, previous_value, num_games=10, eval_simulations=3200, save_sgf_path=None):
    # Sauvegarder l'état d'entraînement actuel
    current_policy_training = current_policy.training
    current_value_training = current_value.training
    previous_policy_training = previous_policy.training
    previous_value_training = previous_value.training
    
    # Mettre en mode évaluation
    current_policy.eval()
    current_value.eval()
    previous_policy.eval()
    previous_value.eval()

    wins_current = 0
    wins_previous = 0
    draws = 0
    successful_games = 0
    
    # Ajouter une barre de progression pour les parties d'évaluation
    for i in tqdm(range(num_games), desc="Évaluation des modèles"):
        try:
            # Jouer une partie en alternant les côtés
            if i % 2 == 0:
                # Current joue Noir, Previous joue Blanc
                first_player = (current_policy, current_value)
                second_player = (previous_policy, previous_value)
                print(f"Évaluation: partie {i+1}/{num_games} (Actuel=Noir vs Précédent=Blanc)", end="", flush=True)
                result, moves, winner, points_noir, points_blanc = play_evaluation_game(first_player, second_player, eval_simulations)
                
                # Sauvegarder la partie au format SGF si un chemin est fourni
                if save_sgf_path:
                    save_game_to_sgf(moves, winner, points_noir, points_blanc, save_sgf_path, 
                                       append=(successful_games > 0), black_name="Current_Model", white_name="Previous_Model")
                
                if result > 0:
                    wins_current += 1
                    print(f" - Victoire du modèle actuel ({wins_current}-{wins_previous}-{draws})")
                elif result == 0.0:
                    wins_previous += 1
                    print(f" - Victoire du modèle précédent ({wins_current}-{wins_previous}-{draws})")
                else:
                    draws += 1
                    print(f" - Match nul ({wins_current}-{wins_previous}-{draws})")
            else:
                # Previous joue Noir, Current joue Blanc
                first_player = (previous_policy, previous_value)
                second_player = (current_policy, current_value)
                print(f"Évaluation: partie {i+1}/{num_games} (Précédent=Noir vs Actuel=Blanc)", end="", flush=True)
                result, moves, winner, points_noir, points_blanc = play_evaluation_game(first_player, second_player, eval_simulations)
                
                # Sauvegarder la partie au format SGF si un chemin est fourni
                if save_sgf_path:
                    save_game_to_sgf(moves, winner, points_noir, points_blanc, save_sgf_path, 
                                       append=(successful_games > 0), black_name="Previous_Model", white_name="Current_Model")
                
                if result > 0:
                    wins_previous += 1
                    print(f" - Victoire du modèle précédent ({wins_current}-{wins_previous}-{draws})")
                elif result == 0.0:
                    wins_current += 1
                    print(f" - Victoire du modèle actuel ({wins_current}-{wins_previous}-{draws})")
                else:
                    draws += 1
                    print(f" - Match nul ({wins_current}-{wins_previous}-{draws})")
            
            successful_games += 1
            
        except Exception as e:
            print(f"❌ Erreur lors de l'évaluation de la partie {i+1}: {e}")
            continue
    
    # Calculer le taux de victoire seulement si des parties ont été jouées
    if successful_games > 0:
        win_rate = wins_current / successful_games
        print(f"✅ Évaluation terminée: {wins_current} victoires, {wins_previous} défaites, {draws} égalités ({successful_games} parties réussies)")
        print(f"📊 Taux de victoire du nouveau modèle: {win_rate:.2%}")
    else:
        win_rate = 0.5  # Valeur par défaut si aucune partie n'a réussi
        print(f"⚠️  Aucune partie d'évaluation n'a réussi, taux de victoire par défaut: {win_rate:.2%}")
    
    # Restaurer l'état d'entraînement
    if current_policy_training:
        current_policy.train()
    if current_value_training:
        current_value.train()
    if previous_policy_training:
        previous_policy.train()
    if previous_value_training:
        previous_value.train()

    return win_rate

def play_evaluation_game(black_player, white_player, eval_simulations=3200, temperature=0.3):
    """Joue une partie d'évaluation entre deux joueurs avec stochasticité contrôlée"""
    # Utiliser une graine différente pour chaque partie
    partie_id = random.randint(0, 10000)
    random.seed(partie_id)
    np.random.seed(partie_id)
    torch.manual_seed(partie_id)
    
    plateau = Plateau(BOARD_SIZE)
    moteur = simulation_MoteurDeJeu(plateau)
    
    players = {BLACK: black_player, WHITE: white_player}
    moves = []  # Pour le fichier SGF
    
    move_count = 0
    max_moves = BOARD_SIZE * BOARD_SIZE * 3  # Limite raisonnable

    with torch.no_grad():
        while not moteur.end_game() and move_count < max_moves:
            current_player = moteur.get_turn()
            current_policy_model, current_value_model = players[current_player]

            # Utiliser la fonction d'activation qui prend les modèles externes
            mcts.enable_policy_value_networks(
                iterations=eval_simulations,
                exploration=0.5,  # Valeur constante pour l'évaluation
                policy_model_ext=current_policy_model, 
                value_model_ext=current_value_model
            )

            # Utiliser MCTS avec un peu de bruit Dirichlet pour garantir la diversité
            move, tree = mcts.MCTS(moteur, current_player, use_tqdm=False,
                                   use_dirichlet=True, 
                                   dirichlet_alpha=0.03, 
                                   dirichlet_weight=0.1,
                                   autorise_pass=False)

            # Avant de jouer, sauvegarder l'état
            turn_before = moteur.get_turn()
            plateau_before = moteur.get_plateau().get_plateau().copy()

            # Tenter de jouer le coup
            moteur.jouer_coup(move)

            # Vérifier si le coup a été accepté (le tour a changé ou le plateau a changé)
            turn_after = moteur.get_turn()
            plateau_after = moteur.get_plateau().get_plateau()

            if (turn_after != turn_before) or (not np.array_equal(plateau_before, plateau_after)):
                moves.append((current_player, move))
                move_count += 1
            else:
                print(f"⚠️ Coup illégal ignoré en évaluation : {move} pour le joueur {current_player}")
                # Essayer de passer si ce n'était pas déjà un pass
                if move is not None:
                    moteur.jouer_coup(None)
                    moves.append((current_player, None))
                    move_count += 1
                else:
                    # Si même passer ne marche pas, on arrête la partie
                    print("Aucun coup légal possible, fin de partie forcée.")
                    break
    
    # Déterminer le résultat
    moteur.get_plateau().actualise_points_territoires()
    points_noir, points_blanc = moteur.get_plateau().get_score_total_couple()
    
    # Déterminer le gagnant pour le fichier SGF
    if points_noir > points_blanc:
        winner = BLACK
        result = 1.0  # Noir gagne
    elif points_blanc > points_noir:
        winner = WHITE
        result = 0.0  # Blanc gagne
    else:
        winner = None  # Match nul
        result = 0.5  # Égalité
    
    # Retourner le résultat ET les informations pour le fichier SGF
    return result, moves, winner, points_noir, points_blanc

# python3 reinforcement_learning.py --iterations 100 --games 75 --epochs 12

def main():
    parser = argparse.ArgumentParser(description="Apprentissage par renforcement pour le jeu de Go")
    parser.add_argument('--iterations', type=int, default=NUM_ITERATIONS, help='Nombre d\'itérations d\'apprentissage')
    parser.add_argument('--games', type=int, default=NUM_SELF_PLAY_GAMES, help='Nombre de parties d\'auto-jeu par itération')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Nombre d\'époques d\'entraînement par itération')
    parser.add_argument('--mcts_sims', type=int, default=MCTS_SIMULATIONS, help='Nombre de simulations MCTS par coup')
    parser.add_argument('--eval_simulations', type=int, default=3200, help='Nombre de simulations MCTS pour l\'évaluation des modèles')
    parser.add_argument('--eval_games', type=int, default=15, help='Nombre de parties d\'évaluation')
    parser.add_argument('--load_buffer', type=str, default=None, help='Chemin du buffer à charger')
    parser.add_argument('--load_policy', type=str, default="training_logs_policy9x9_v5_2025-06-06_18-21-11/" \
    "best_policynet_9x9.pth", help='Chemin du modèle de politique à charger')
    parser.add_argument('--load_value', type=str, default="training_logs_value9x9_v3_2025-06-06_18-21-00/" \
    "best_valuenet_9x9.pth", help='Chemin du modèle de valeur à charger')
    parser.add_argument('--resume', type=str, default=None, help='Reprendre l\'entraînement depuis un checkpoint complet')
    parser.add_argument('--device', type=str, default=None, help='Device à utiliser: cpu, cuda, cuda:0, cuda:1, etc.')
    
    args = parser.parse_args()
    
    # Gestion du device
    global DEVICE, USE_MULTI_GPU, NUM_GPUS
    if args.device:
        DEVICE, USE_MULTI_GPU, NUM_GPUS = parse_device(args.device)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        NUM_GPUS = torch.cuda.device_count()
        USE_MULTI_GPU = NUM_GPUS > 1 if DEVICE.type == 'cuda' else False

    if USE_MULTI_GPU:
        print(f"🚀 Utilisation de {NUM_GPUS} GPUs sur {DEVICE}")
    else:
        print(f"💻 Utilisation de: {DEVICE}")
    
    # Traiter le cas de reprise d'un checkpoint complet
    if args.resume:
        resume_rl_training(
            checkpoint_path=args.resume,
            iterations=args.iterations if args.iterations != NUM_ITERATIONS else None,
            games_per_iteration=args.games if args.games != NUM_SELF_PLAY_GAMES else None
        )
        return
    
    # Création du dossier de sortie
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"rl_training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Pour le modèle de politique
    if args.load_policy:
        print(f"Chargement du modèle de politique depuis {args.load_policy}")
        policy_model = load_model(PolicyNet9x9, args.load_policy, wrap_data_parallel=USE_MULTI_GPU, device=DEVICE)
        print("Modèle de politique chargé avec succès")

    # Pour le modèle de valeur
    if args.load_value:
        print(f"Chargement du modèle de valeur depuis {args.load_value}")
        value_model = load_model(ValueNet9x9, args.load_value, wrap_data_parallel=USE_MULTI_GPU, device=DEVICE)
        print("Modèle de valeur chargé avec succès")

    if USE_MULTI_GPU:
        print(f"Modèles distribués sur {NUM_GPUS} GPUs")
    
    # Initialiser ou charger le buffer d'expérience
    experience_buffer = ReinforcementLearningBuffer()
    if args.load_buffer:
        experience_buffer.load(args.load_buffer)
    
    # Lancer l'entraînement
    run_rl_training(
        policy_model=policy_model,
        value_model=value_model,
        buffer=experience_buffer,
        output_dir=output_dir,
        iterations=args.iterations,
        games_per_iteration=args.games,
        epochs_per_iteration=args.epochs,
        mcts_simulations=args.mcts_sims,
        eval_simulations=args.eval_simulations,
        eval_games=args.eval_games
    )

if __name__ == "__main__":
    main()