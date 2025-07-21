"""
Module d'implémentation de Monte Carlo Tree Search (MCTS) pour le jeu de Go.
Intègre des optimisations avec réseaux de neurones (Policy et Value).
"""

import random
import numpy as np
import os
import time
import threading
import concurrent.futures
from tqdm import *
import multiprocessing as mp
from const import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
import torch.nn.functional as F
from noeud_go import Noeud
from arbre_go import Arbre
import moteursDeJeu as mtdj

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os

import sys
import inspect

# Ajouter le chemin spécifique pour les réseaux de neurones
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "katago_relatives", 
                             "katago-opencl-linux", 
                             "networks"))

# Import des modèles de réseaux de neurones
from go_policy_9x9_v5 import PolicyNet9x9
from go_value_9x9_v3 import ValueNet9x9


COLORMAP = False  # Afficher la colormap des scores sur le plateau
# Paramètres d'optimisation pour réduire les calculs
NB_ITERATIONS_MCTS = 2500   # Réduire le nombre d'itérations par défaut
# MAX_ROLLOUT_MOVES = 10  # Limiter le nombre de coups dans les simulations
NB_COUPS_SELECTIONNES = 15  # Nombre de coups à sélectionner lors de l'expansion / AlphaZero, pour 19x19, utilisait 20, mais pour 9x9, on peut réduire à 15
C_PUCT = 1.1              # Paramètre d'exploration PUCT pour la politique

# Variables de contrôle pour utiliser ou non les réseaux de neurones
POLICY = True  # Réseau de politique pour guider les sélections et expansions
VALUE = True   # Réseau de valeur pour évaluer les positions
policy_model = None  # Sera défini soit par injection externe, soit par chargement de fichier
value_model = None   # Sera défini soit par injection externe, soit par chargement de fichier
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Ajout de cette variable manquante

# Définition des chemins de base
base_dir = os.path.dirname(os.path.abspath(__file__))

# Variables pour les chemins de fichiers (utilisées uniquement si pas de modèles injectés)
policy_model_path = os.path.join(base_dir, 
                               "katago_relatives", 
                               "katago-opencl-linux",  
                               "networks",
                               "training_logs_policy9x9_v5_2025-06-06_18-21-11_resumed",
                               "best_policynet_9x9.pth")

value_model_path = os.path.join(base_dir, 
                               "katago_relatives", 
                               "katago-opencl-linux",  
                               "networks",
                               "training_logs_value9x9_v3_2025-06-06_18-21-00",
                               "best_valuenet_9x9.pth")

def reset_random_seeds(seed=None):
    """Réinitialise les graines aléatoires pour assurer la diversité des parties"""
    if seed is None:
        seed = int(time.time() * 1000) % 10000  # Valeur entre 0 et 9999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed

class ValueNet9x9_ancien(nn.Module):
    def __init__(self, board_size=9, n_blocks=4, filters=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        self.resblocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(filters, filters, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        self.value_fc = nn.Sequential(
            nn.Linear(filters * board_size * board_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.resblocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        value = self.value_fc(x).squeeze(-1)
        return value


class PolicyNet9x9_ancien(nn.Module):
    def __init__(self, board_size=9, n_blocks=4, filters=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        self.resblocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(filters, filters, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Linear(filters * board_size * board_size, board_size * board_size + 1)
    def forward(self, x):
        x = self.stem(x)
        x = self.resblocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        policy_logits = self.policy_fc(x)
        return policy_logits


class ResidualBlock_ancien(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


def load_model_with_prefix_handling(model, checkpoint_path, device=torch.device("cpu")):
    """Charge un modèle en gérant les préfixes DataParallel"""
    try:
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
        else:
            model.load_state_dict(state_dict, strict=False)
        
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
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


def load_policy_model(external_model=None, model_path=None):
    """
    Charge le modèle de politique, avec possibilité d'injecter un modèle externe.
    
    Args:
        external_model: Un modèle externe à utiliser (prioritaire)
        model_path: Chemin vers un modèle à charger si aucun modèle externe n'est fourni
    
    Returns:
        Le modèle de politique chargé
    """
    global policy_model
    
    # Si un modèle externe est fourni, l'utiliser
    if external_model is not None:
        policy_model = external_model
        return policy_model
    
    # Si aucun modèle n'est chargé, en créer un nouveau
    if policy_model is None:
        try:
            # Utiliser le chemin spécifié ou le chemin par défaut
            path_to_use = model_path if model_path else policy_model_path
            policy_model = PolicyNet9x9().to(DEVICE)
            policy_model = load_model_with_prefix_handling(policy_model, path_to_use, DEVICE)
            policy_model.eval()
            print(f"Modèle de politique chargé depuis {path_to_use}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle de politique: {e}")
            policy_model = PolicyNet9x9().to(DEVICE)  # Modèle vierge en cas d'erreur
            
    return policy_model


def load_value_model(external_model=None, model_path=None):
    """
    Charge le modèle de valeur, avec possibilité d'injecter un modèle externe.
    
    Args:
        external_model: Un modèle externe à utiliser (prioritaire)
        model_path: Chemin vers un modèle à charger si aucun modèle externe n'est fourni
    
    Returns:
        Le modèle de valeur chargé
    """
    global value_model
    
    # Si un modèle externe est fourni, l'utiliser
    if external_model is not None:
        value_model = external_model
        return value_model
    
    # Si aucun modèle n'est chargé, en créer un nouveau
    if value_model is None:
        try:
            # Utiliser le chemin spécifié ou le chemin par défaut
            path_to_use = model_path if model_path else value_model_path
            value_model = ValueNet9x9().to(DEVICE)
            value_model = load_model_with_prefix_handling(value_model, path_to_use, DEVICE)
            value_model.eval()
            print(f"Modèle de valeur chargé depuis {path_to_use}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle de valeur: {e}")
            value_model = ValueNet9x9().to(DEVICE)  # Modèle vierge en cas d'erreur
            
    return value_model


def get_policy_probs(board_tensor):
    """
    Utilise le modèle de politique pour obtenir les probabilités d'action.
    """
    try:
        model = load_policy_model()
        
        # S'assurer explicitement que le modèle est en mode évaluation
        model.eval()
        
        with torch.no_grad():
            board_tensor = board_tensor.to(next(model.parameters()).device)
            
            # Vérifier les dimensions
            if board_tensor.shape[1] != 2:
                print(f"Erreur: le tenseur doit avoir 2 canaux, mais a shape {board_tensor.shape}")
                return None
                
            # Forcer le mode d'évaluation pour les modules BatchNorm
            def set_bn_eval(module):
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    
            # Appliquer cette fonction à tous les modules
            model.apply(set_bn_eval)
            
            policy_logits = model(board_tensor)
            policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0, :]
            
        return policy_probs  # Retourne les probabilités pour toutes les actions possibles
    except Exception as e:
        print(f"Erreur lors de l'obtention des probabilités de politique : {e}")
        # En cas d'erreur, retourner une politique uniforme comme fallback
        board_size = board_tensor.shape[2]
        uniform_probs = np.ones(board_size * board_size + 1) / (board_size * board_size + 1)
        return uniform_probs


def get_position_value(board_tensor, player_perspective):
    """
    Utilise le modèle de valeur pour évaluer une position.
    """
    try:
        model = load_value_model()
        
        # S'assurer explicitement que le modèle est en mode évaluation
        model.eval()
        
        with torch.no_grad():
            board_tensor = board_tensor.to(next(model.parameters()).device)
            
            # Vérifier les dimensions
            if board_tensor.shape[1] != 2:
                print(f"Erreur: le tenseur doit avoir 2 canaux, mais a shape {board_tensor.shape}")
                return 0.5
                
            # Forcer le mode d'évaluation pour les modules BatchNorm
            def set_bn_eval(module):
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    
            # Appliquer cette fonction à tous les modules
            model.apply(set_bn_eval)
            
            value = model(board_tensor).cpu().numpy()
            
            return float(value)
    except Exception as e:
        print(f"Erreur lors de l'évaluation de la position : {e}")
        return 0.5  # Valeur par défaut


def selection(tree:Arbre):
    """
    Fonction de sélection pour MCTS classique (sans réseaux de neurones).
    Sélectionne un nœud en utilisant l'algorithme Upper Confidence Bound (UCB).
    """
    root = tree.get_racine() #racine de l'arbre
    childs = tree.get_enfants(root)
    N = root.get_nombre_simulation()
    
    node_max = root
    path = [root]

    while childs != []: #tant que je ne suis pas sur une feuille
        UCB_max = -np.inf
        node_max = childs[0]
        for node in childs:
            n_i = node.get_nombre_simulation()
            w_i = node.get_value()

            # Formule UCB classique
            if n_i == 0:
                UCB_i = np.inf
            else:
                UCB_i = w_i/n_i + np.sqrt(2)*np.sqrt(np.log(N)/n_i)

            if UCB_i > UCB_max:
                UCB_max = UCB_i
                node_max = node

        path.append(node_max)

        N = node_max.get_nombre_simulation()
        childs = tree.get_enfants(node_max)
        
    return path, node_max


def expansion(parent:Noeud, arbre:Arbre):
    """
    Fonction d'expansion pour MCTS classique (sans réseaux de neurones).
    Ajoute tous les nœuds enfants possibles à partir du nœud parent.
    """
    moteur_parent = parent.get_etat()

    if not moteur_parent.end_game():
        joueur_parent = parent.get_joueur()
        liste_coups = moteur_parent.liste_coups_jouable()
        
        while liste_coups:
            coup = liste_coups.pop()
            moteur_enfant = moteur_parent.clone()
            moteur_enfant.jouer_coup(coup)
            
            # Créer le nouveau nœud
            arbre.add_noeud(parent, moteur_enfant, next_player(joueur_parent), coup)


def board_to_tensor(board):
    size = board.shape[0]
    x = np.zeros((1, 2, size, size), dtype=np.float32)
    x[0, 0] = (board == 1)
    x[0, 1] = (board == 2)
    return torch.from_numpy(x)


def index_to_move(idx, size):
    if idx == size*size:
        return None
    return (idx // size, idx % size)


def move_to_index(move, board_size):
    """
    Mappe un coup en index dans [0, board_size*board_size] :
    - (i, j) -> i * board_size + j
      - None   -> board_size * board_size (pass)
    """
    if move is None:
        return board_size * board_size
    i, j = move
    return i * board_size + j


def rollout(tuple):
    """
    Fonction de simulation (rollout) pour MCTS classique (sans réseaux de neurones).
    Effectue une simulation aléatoire à partir d'un nœud jusqu'à la fin de la partie.
    """
    noeud, ai_turn = tuple
    moteur_parent = noeud.get_etat()
    
    # Simulation classique
    moteur_modif = moteur_parent.clone()
    
    # Pour accélérer le rollout, limiter le nombre de coups à jouer
    # max_coups = MAX_ROLLOUT_MOVES  # Utiliser la constante globale
    # nb_coups = 0
    
    # Simulation avec coups aléatoires
    while not moteur_modif.end_game(): # and nb_coups < max_coups:
        coups_possibles = moteur_modif.liste_coups_jouable()
        if not coups_possibles:  # Si pas de coups possibles, terminer la simulation
            break
            
        coup = random.choice(coups_possibles)
        moteur_modif.jouer_coup(coup)
        # nb_coups += 1

    moteur_modif.get_plateau().actualise_points_territoires()
    points_noir, points_blanc = moteur_modif.get_plateau().get_score_total_couple()

    return 1 if (points_blanc - points_noir > 0) == (ai_turn == WHITE) else 0


def retropropagation(arbre, chemin, value):
    """
    Fonction de rétropropagation pour MCTS.
    Met à jour les statistiques des nœuds visités.
    """
    for noeud in chemin:
        noeud.actualise_simulation_et_value(value)  # value = 0 si la partie est perdue
        if value > 0:
            noeud.add_win()
        else:
            noeud.add_loose()
    return arbre


def next_player(player):
    """Retourne l'identifiant du joueur suivant."""
    return BLACK if player == WHITE else WHITE


def selection_with_policy(tree:Arbre):
    """
    Fonction de sélection avancée utilisant un réseau de politique.
    Utilise la formule PUCT (Polynomial Upper Confidence Trees) au lieu de UCB.
    """
    root = tree.get_racine() #racine de l'arbre
    childs = tree.get_enfants(root)
    N = root.get_nombre_simulation()
    
    node_max = root
    path = [root]

    while childs != []: #tant que je ne suis pas sur une feuille
        UCB_max = -np.inf
        node_max = childs[0]
        for node in childs:
            n_i = node.get_nombre_simulation()
            w_i = node.get_value()

            # Formule PUCT (Polynomial Upper Confidence Trees)
            c_puct = C_PUCT  # Paramètre d'exploration
            
            if n_i == 0:
                UCB_i = np.inf
            else:
                # UCT(s,a) = Q(s,a) + c*P(s,a)* √N(s)/(1+N(s,a))
                UCB_i = w_i/n_i + c_puct * node.get_prior() * np.sqrt(N) / (1 + n_i)

            if UCB_i > UCB_max:
                UCB_max = UCB_i
                node_max = node

        path.append(node_max)

        N = node_max.get_nombre_simulation()
        childs = tree.get_enfants(node_max)
        
    return path, node_max


def expansion_with_policy(parent:Noeud, arbre:Arbre, top_k=NB_COUPS_SELECTIONNES):
    """
    Fonction d'expansion sélective utilisant un réseau de politique.
    Ne garde que les top-k coups les plus prometteurs selon le réseau.
    
    Args:
        parent: Le nœud parent à partir duquel effectuer l'expansion
        arbre: L'arbre MCTS
        top_k: Nombre de coups à conserver (les plus probables)
    """
    moteur_parent = parent.get_etat()

    if not moteur_parent.end_game():
        joueur_parent = parent.get_joueur()
        liste_coups = moteur_parent.liste_coups_jouable()
        
        # Obtenir le plateau
        plateau = moteur_parent.get_plateau().get_plateau()
        board_size = plateau.shape[0]
        
        # Obtenir les probabilités d'action
        board_tensor = board_to_tensor(plateau)
        probs = get_policy_probs(board_tensor)
        
        if probs is None:
            # Fallback: utiliser l'expansion normale si le réseau échoue
            return expansion(parent, arbre)
        
        # Créer un dictionnaire de priorités pour chaque coup
        priors = {}
        for coup in liste_coups:
            if coup is None:  # Passe
                idx = board_size * board_size
            else:
                i, j = coup
                idx = i * board_size + j
            
            # Vérifier que l'index est valide
            if idx < len(probs):
                priors[coup] = probs[idx]
            else:
                priors[coup] = 0.001  # Valeur par défaut faible
        
        # Trier les coups par probabilité décroissante et ne garder que les top_k
        top_moves = sorted(priors.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Créer les nœuds enfants uniquement pour les coups sélectionnés
        for coup, prior_value in top_moves:
            moteur_enfant = moteur_parent.clone()
            moteur_enfant.jouer_coup(coup)
            
            # Créer le nouveau nœud
            nouveau_noeud = arbre.add_noeud(parent, moteur_enfant, next_player(joueur_parent), coup)
            
            # Initialiser P(s,a)
            nouveau_noeud.set_prior(prior_value)


def rollout_with_value_2(tuple):
    """
    Fonction de simulation (rollout) pour MCTS classique (sans réseaux de neurones).
    Effectue une simulation aléatoire à partir d'un nœud jusqu'à la fin de la partie.
    """
    noeud, ai_turn = tuple
    moteur_parent = noeud.get_etat()
    top_k = 5   
    # Simulation classique
    moteur_modif = moteur_parent.clone()
    
    # Pour accélérer le rollout, limiter le nombre de coups à jouer
    # Calculate max_coups based on remaining stones needed to fill half the board
    plateau = moteur_modif.get_plateau().get_plateau()
    board_size = plateau.shape[0]  # Assuming it's a square board
    total_positions = board_size * board_size
    current_stones = np.sum(plateau > 0)  # Count all stones (values greater than 0)
    half_board = total_positions // 2  # Integer division to get half
    max_coups = max(1, half_board - current_stones)  # Make sure it's at least 1
    max_coups = 10000
    nb_coups = 0

    # Simulation avec coups aléatoires
    while not moteur_modif.end_game() and nb_coups < max_coups:
        coups_possibles = moteur_modif.liste_coups_jouable()
        if not coups_possibles:  # Si pas de coups possibles, terminer la simulation
            break
        """
        coup = random.choice(coups_possibles)
        
        moteur_modif.jouer_coup(coup)
        nb_coups += 1
        """
        

        # Obtenir le plateau
        plateau = moteur_modif.get_plateau().get_plateau()  # Modification ici: moteur_modif au lieu de moteur_parent
        board_size = plateau.shape[0]
        
        # Obtenir les probabilités d'action
        board_tensor = board_to_tensor(plateau)
        probs = get_policy_probs(board_tensor)
        
        # Créer un dictionnaire de priorités pour chaque coup
        priors = {}
        for coup in coups_possibles:
            if coup is None:  # Passe
                idx = board_size * board_size
            else:
                i, j = coup
                idx = i * board_size + j
            
            # Vérifier que l'index est valide
            if idx < len(probs):
                priors[coup] = probs[idx]
            else:
                priors[coup] = 0.001  # Valeur par défaut faible
        
        # Trier les coups par probabilité décroissante et ne garder que les top_k
        top_moves = sorted(priors.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Correction ici: extraire seulement le coup du tuple (coup, prior_value)
        coup_et_prior = random.choice(top_moves)
        coup = coup_et_prior[0]  # Extraire seulement le coup
        
        moteur_modif.jouer_coup(coup)
        nb_coups += 1
        


    # Obtenir le plateau final pour l'évaluation
    plateau = moteur_modif.get_plateau().get_plateau()  # Aussi modifié: utiliser moteur_modif
    board_tensor = board_to_tensor(plateau)
    
    # Obtenir la valeur depuis le réseau de valeur
    value = get_position_value(board_tensor, ai_turn)
    
    # print(plateau)
    # print("Valeur de la position:", value, "pour le joueur", "NOIR" if ai_turn == BLACK else "BLANC")

    # moteur_modif.get_plateau().actualise_points_territoires()
    # points_noir, points_blanc = moteur_modif.get_plateau().get_score_total_couple()

    # print("Score - Noir:", points_noir, "Blanc:", points_blanc)

    return float(value)


def rollout_with_value(tuple):
    """
    Fonction de simulation utilisant un réseau de valeur.
    Remplace la simulation par une évaluation directe de la position.
    """
    noeud, ai_turn = tuple
    moteur_parent = noeud.get_etat()
    
    # Obtenir le plateau
    plateau = moteur_parent.get_plateau().get_plateau()
    board_tensor = board_to_tensor(plateau)
    
    try:
        # Obtenir la valeur depuis le réseau de valeur
        value = get_position_value(board_tensor, ai_turn)
        
        # Vérifier si la valeur est None ou invalide
        if value is None or not np.isfinite(value).all():
            print(f"Valeur invalide détectée ({value}), utilisation d'une valeur par défaut")
            return 0.5  # Valeur neutre comme fallback
        
        # Ajouter un petit bruit aléatoire pour éviter le déterminisme
        noise = (random.random() - 0.5) * 0.05  # Bruit entre -0.025 et +0.025
        value = max(0.0, min(1.0, float(value) + noise))  # Garder value entre 0 et 1
        
        return value
    except Exception as e:
        print(f"Erreur dans rollout_with_value: {e}")
        return 0.5  # Valeur par défaut en cas d'erreur


def MCTS(moteur_obj, ai_turn:int, use_tqdm=True, use_dirichlet=True, 
        dirichlet_alpha=0.03, dirichlet_weight=0.25, temperature=None, autorise_pass=True):
    """
    Fonction principale de Monte Carlo Tree Search (MCTS).
    
    Args:
        moteur_obj: L'objet moteur de jeu contenant l'état actuel
        ai_turn: Le joueur dont c'est le tour (NOIR ou BLANC)
        use_tqdm: Afficher une barre de progression
        use_dirichlet: Appliquer du bruit de Dirichlet
        dirichlet_alpha: Paramètre alpha pour le bruit de Dirichlet
        dirichlet_weight: Poids du bruit dans la distribution finale
        temperature: Température pour la sélection (None = dynamique basée sur avancement)
        autorise_pass: Si False, le coup "pass" n'est autorisé que si le joueur gagne en passant
        
    Returns:
        Le coup choisi par l'algorithme
    """
    global POLICY, VALUE
    
    # Réinitialiser les seeds aléatoires pour chaque appel MCTS
    random_seed = int(time.time() * 1000) % 10000
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Initialisation de l'arbre
    arbre = Arbre({})
    moteur_racine = moteur_obj.clone()
    arbre.add_noeud(None, moteur_racine, next_player(ai_turn), None)

    # Utiliser la constante globale pour le nombre d'itérations
    nb_iterations = NB_ITERATIONS_MCTS

    # Choisir les fonctions à utiliser en fonction des variables globales
    select_func = selection
    expand_func = expansion
    rollout_func = rollout

    # Adapter les fonctions selon les réseaux activés
    if POLICY:
        select_func = selection_with_policy
        expand_func = expansion_with_policy

    if VALUE:
        rollout_func = rollout_with_value

    # Exécution des itérations MCTS
    iteration_range = range(nb_iterations)
    if use_tqdm:
        iteration_range = tqdm(iteration_range)
        
    for i in iteration_range:
        path, noeud = select_func(arbre)
        if noeud.get_nombre_simulation() > 0:
            expand_func(noeud, arbre)
            path, noeud = select_func(arbre)

        value = rollout_func((noeud, ai_turn))

        retropropagation(arbre, path, value)

    root = arbre.get_racine()
    childs = arbre.get_enfants(root)

    # Sélection du coup à la fin du MCTS
    plateau = moteur_racine.get_plateau().get_plateau()
    board_size = plateau.shape[0]
    move_count = np.sum(plateau > 0)  # Nombre de pierres sur le plateau
    
    # Utiliser température fournie ou calculer dynamiquement
    if temperature is None:
        # Calcul dynamique de la température selon l'avancement de la partie
        temperature = max(0.5, 1.0 - (move_count / (board_size * board_size * 0.8)))
    
    # Calculer les scores basés sur les visites
    scores = {}
    total_visits = sum(node.get_nombre_simulation() for node in childs)

    for node in childs:
        n_i = node.get_nombre_simulation()
        coup = node.get_coup()
        
        # Utiliser une pénalité progressive pour "passer" selon l'étape de la partie
        if coup is None:  # Si c'est l'action "passer"
            # Début de partie: fortement pénalisé
            if move_count < board_size * board_size * 0.3:
                visit_factor = 0.001
            # Milieu de partie: modérément pénalisé
            elif move_count < board_size * board_size * 0.6:
                visit_factor = 0.1
            # Fin de partie: pas de pénalité
            else:
                visit_factor = 1.0
        else:
            visit_factor = 1.0
        
        # Appliquer la température (T→0 = déterministe, T→∞ = uniforme)
        score = (n_i / total_visits) ** (1 / temperature) * visit_factor
        scores[node] = score

    # Ajouter du bruit de Dirichlet si demandé
    if use_dirichlet:
        noise = np.random.dirichlet([dirichlet_alpha] * len(childs))
        for i, node in enumerate(childs):
            scores[node] = (1 - dirichlet_weight) * scores[node] + dirichlet_weight * noise[i]
    
    # --- FILTRAGE DU COUP PASSER SI DEMANDÉ ---
    if not autorise_pass:
        # On ne garde le coup "pass" que si le joueur gagne en passant
        filtered_nodes = []
        for node in scores:
            coup = node.get_coup()
            if coup is not None:
                filtered_nodes.append(node)
            else:
                # Simuler le pass
                moteur_simule = moteur_racine.clone()
                moteur_simule.jouer_coup(None)
                moteur_simule.get_plateau().actualise_points_territoires()
                points_noir, points_blanc = moteur_simule.get_plateau().get_score_total_couple()
                if ai_turn == BLACK:
                    if points_noir > points_blanc:
                        filtered_nodes.append(node)
                else:
                    if points_blanc > points_noir:
                        filtered_nodes.append(node)
        # Si aucun coup n'est possible (très rare), on garde tout de même tous les coups
        if filtered_nodes:
            nodes = filtered_nodes
            probs = np.array([scores[n] for n in nodes])
        else:
            nodes = list(scores.keys())
            probs = np.array(list(scores.values()))
    else:
        nodes = list(scores.keys())
        probs = np.array(list(scores.values()))
    # --- FIN FILTRAGE ---

    if probs.sum() > 0:  # Éviter division par zéro
        probs = probs / probs.sum()
        selected_child = np.random.choice(nodes, p=probs)
    else:
        selected_child = max(scores, key=scores.get)

    # Vérifier si on doit afficher la colormap
    if hasattr(moteur_obj, '_MoteurDeJeu__gobanGraphique') and hasattr(moteur_obj._MoteurDeJeu__gobanGraphique, 'get_show_colormap'):
        if moteur_obj._MoteurDeJeu__gobanGraphique.get_show_colormap():
            afficher_colormap(moteur_racine.get_plateau().get_plateau(), childs, selected_case=selected_child.get_coup())

    return selected_child.get_coup(), arbre


def afficher_plateau_avec_scores(plateau, childs):
    """Affiche le plateau avec les scores w_i/n_i de chaque case."""
    scores = np.zeros_like(plateau, dtype=float)  # Créer un tableau de scores de la même taille que le plateau
    
    for node in childs:
        coup = node.get_coup()  # Le coup joué par cet enfant (sous forme de (x, y))
        if coup is None:  # Ignorer le cas où le coup est None (passer)
            continue
            
        n_i = node.get_nombre_simulation()
        w_i = node.get_value()

        if n_i > 0:  # Assurer que n_i n'est pas 0 pour éviter la division par zéro
            score = w_i / n_i
            x, y = coup  # On suppose que coup est un tuple (x, y) représentant une case
            scores[x, y] = score  # Attribuer le score à la case correspondante

    # Affichage du plateau avec les scores
    for i in range(plateau.shape[0]):  # Itérer sur les lignes du plateau
        row_display = []
        for j in range(plateau.shape[1]):  # Itérer sur les colonnes du plateau
            case_value = plateau[i, j]  # Valeur de la case (0, 1, ou 2)
            score = scores[i, j]  # Score associé à cette case
            row_display.append(f"{case_value}({score:.2f})")  # Affiche la valeur de la case et le score
        print(" | ".join(row_display))
        print("-" * (len(row_display) * 6 - 1))  # Juste pour séparer les lignes du plateau


def afficher_colormap(plateau, childs, selected_case=None):
    """Affiche une colormap représentant les scores des cases sur le plateau,
    marque la case sélectionnée, et affiche n_i sur chaque case."""
    scores = np.zeros_like(plateau, dtype=float)
    n_counts = np.zeros_like(plateau, dtype=int)

    # Remplir scores et n_counts à partir des enfants MCTS
    for node in childs:
        coup = node.get_coup()  # (x, y) ou None
        if coup is None:       # passe tour → pas de case à annoter
            continue
        n_i = node.get_nombre_simulation()
        w_i = node.get_value()
        score = (w_i / n_i) if n_i > 0 else 0.0
        x, y = coup
        scores[x, y] = score
        n_counts[x, y] = n_i

    # Adjust normalization to avoid over-representation of low-simulation nodes
    max_score = np.max(scores)
    im = plt.imshow(scores, cmap='viridis', interpolation='nearest', norm=Normalize(vmin=0, vmax=max_score))
    plt.colorbar(im, label='Score (w_i / n_i)')

    for (i, j), val in np.ndenumerate(n_counts):
        if val > 0:
            plt.text(j, i, str(val), ha='center', va='center', color='white', fontsize=8)

    if selected_case:
        x, y = selected_case
        plt.scatter(y, x, color='red', s=100, edgecolor='black', marker='o')

    plt.title("Colormap des scores et nombre de simulations par case")
    plt.xlabel("Colonnes")
    plt.ylabel("Lignes")
    plt.show()


def set_mcts_mode(mode="classic", iterations=20, exploration=1.0, 
                  policy_model_ext=None, value_model_ext=None, 
                  policy_path=None, value_path=None):
    """
    Configure facilement le mode MCTS à utiliser avec possibilité d'injecter des modèles.
    
    Args:
        mode: Le mode à utiliser ("classic", "policy", "value", "policy_value")
        iterations: Nombre d'itérations MCTS
        exploration: Facteur d'exploration (seulement pour POLICY)
        policy_model_ext: Modèle de politique externe à utiliser
        value_model_ext: Modèle de valeur externe à utiliser
        policy_path: Chemin vers un modèle de politique à charger
        value_path: Chemin vers un modèle de valeur à charger
    
    Returns:
        bool: True si le mode a été activé avec succès
    """
    global POLICY, VALUE, NB_ITERATIONS_MCTS, C_PUCT
    
    # Réinitialiser toutes les options
    POLICY = False
    VALUE = False
    
    # Configurer le mode demandé
    # if mode.lower() == "classic":
    #     # POLICY et VALUE sont déjà à False, rien à faire
    #     print("Mode MCTS classique activé")
    if mode.lower() == "policy":
        POLICY = True
        C_PUCT = exploration
        # print(f"Mode MCTS avec réseau de politique activé (exploration={exploration})")
        try:
            load_policy_model(policy_model_ext, policy_path)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle de politique: {e}")
            POLICY = False
            return False
    elif mode.lower() == "value":
        VALUE = True
        # print("Mode MCTS avec réseau de valeur activé")
        try:
            load_value_model(value_model_ext, value_path)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle de valeur: {e}")
            VALUE = False
            return False
    elif mode.lower() == "policy_value":
        POLICY = True
        VALUE = True
        C_PUCT = exploration
        # print(f"Mode MCTS avec réseaux de politique et valeur activés (exploration={exploration})")
        try:
            load_policy_model(policy_model_ext, policy_path)
            load_value_model(value_model_ext, value_path)
        except Exception as e:
            print(f"Erreur lors du chargement des modèles: {e}")
            POLICY = False
            VALUE = False
            return False
    
    # Configurer le nombre d'itérations
    NB_ITERATIONS_MCTS = iterations
    # print(f"Nombre d'itérations MCTS: {iterations}")
    
    return True


def enable_value_network(iterations=20, value_model_ext=None, value_path=None):
    """Active le réseau de valeur pour MCTS."""
    return set_mcts_mode("value", 
                         iterations, 
                         value_model_ext=value_model_ext, 
                         value_path=value_path)


def enable_policy_network(iterations=20, exploration=1.0, policy_model_ext=None, policy_path=None):
    """Active le réseau de politique pour MCTS."""
    return set_mcts_mode("policy", 
                         iterations, 
                         exploration, 
                         policy_model_ext=policy_model_ext, 
                         policy_path=policy_path)


def enable_policy_value_networks(iterations=20, exploration=1.0, policy_model_ext=None, 
                                 value_model_ext=None, policy_path=None, value_path=None):
    """Active à la fois les réseaux de politique et de valeur pour MCTS."""
    return set_mcts_mode("policy_value", 
                         iterations, 
                         exploration, 
                         policy_model_ext=policy_model_ext, 
                         value_model_ext=value_model_ext, 
                         policy_path=policy_path, 
                         value_path=value_path)